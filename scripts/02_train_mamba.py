"""
Same training harness as 01_train_baseline_transformer.py, swapped to
TinyMambaLM. It's near copy-paste on purpose -- keeping the harness (task
generator, optimizer, logging, metric) byte-for-byte identical between the
two is what makes the eventual comparison (training speed, phase-change
shape, final accuracy) meaningful rather than an artifact of different
setups.

Resumable: saves a full checkpoint (model + optimizer + step + history)
periodically and on a wall-clock budget, and resumes from it if present, so
a long run can be done in several foreground chunks instead of needing a
single unbroken background process.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.induction_task import InductionTaskConfig, InductionTaskGenerator, induction_accuracy
from src.models.minimal_mamba import TinyMambaLM

CKPT_PATH = "results/mamba_ckpt.pt"
MAX_WALL_SECONDS = 540  # leaves headroom under a ~600s foreground call


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_cfg = InductionTaskConfig(vocab_size=50, seq_len=64, batch_size=256, seed=0)
    gen = InductionTaskGenerator(task_cfg)

    model = TinyMambaLM(vocab_size=task_cfg.vocab_size, d_model=64, n_layers=2).to(device)

    # weight_decay=0.0 turned out to matter a lot: with AdamW's default decay
    # (0.01) the model never escapes chance-level loss even after thousands
    # of steps, on either the full task or a scaled-down debug version (short
    # seq_len, small vocab). With decay off, the scaled-down version shows a
    # clean phase transition (chance -> ~85% induction_acc) within ~1000
    # steps. Root cause not fully nailed down (weight decay continuously
    # pulling the slow-moving selective-SSM parameters, A_log/dt_proj/
    # x_to_bc_delta, back toward 0 before the multi-parameter induction
    # circuit can assemble, is the leading hypothesis) but the fix is
    # reproducible, so applying it here for the full-scale run.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    n_steps = 20000
    log_every = 50

    Path("results").mkdir(exist_ok=True)
    history = []
    start_step = 1

    if Path(CKPT_PATH).exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        history = ckpt["history"]
        print(f"resumed from {CKPT_PATH} at step {start_step}", flush=True)

    def save_checkpoint(step):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "history": history,
            },
            CKPT_PATH,
        )
        torch.save(model.state_dict(), "results/mamba_baseline.pt")
        np.save("results/mamba_train_history.npy", np.array(history))

    t0 = time.time()
    step = start_step - 1

    for step in range(start_step, n_steps + 1):
        if time.time() - t0 > MAX_WALL_SECONDS:
            print(f"wall-clock budget reached at step {step}, checkpointing and exiting", flush=True)
            break

        batch = gen.sample_batch()
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        induction_mask = batch["induction_mask"].to(device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            # Safety net on top of the A_log clamp / delta floor in
            # minimal_mamba.py: skip a step outright if a nan/inf slipped
            # through anyway, instead of letting AdamW poison every
            # parameter's moment estimates with it.
            print(f"step {step:5d}  SKIPPED (non-finite loss/grad)", flush=True)
            optimizer.zero_grad()
            continue
        optimizer.step()

        if step % log_every == 0 or step == 1:
            acc = induction_accuracy(logits, target_ids, induction_mask)
            history.append((step, loss.item(), acc))
            print(f"step {step:5d}  loss {loss.item():.4f}  induction_acc {acc:.3f}", flush=True)

        if step % 200 == 0:
            save_checkpoint(step)

    save_checkpoint(step)
    print(f"checkpoint saved at step {step}", flush=True)


if __name__ == "__main__":
    main()
