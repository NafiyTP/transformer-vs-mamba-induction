"""
Train the tiny attention-only transformer on the induction task and confirm
the classic induction-head signature before touching Mamba at all.

Success criterion (from Olsson et al.): induction_accuracy should show a
sharp jump partway through training (the "phase change"), not a smooth
improvement, and by the end at least one attention head should show the
diagonal-offset attention pattern characteristic of induction heads
(visualize with model.run_with_cache + circuitsvis or a simple imshow).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.data.induction_task import InductionTaskConfig, InductionTaskGenerator, induction_accuracy
from src.models.tiny_transformer import build_tiny_induction_transformer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_cfg = InductionTaskConfig(vocab_size=50, seq_len=64, batch_size=64, seed=0)
    gen = InductionTaskGenerator(task_cfg)

    model = build_tiny_induction_transformer(
        vocab_size=task_cfg.vocab_size, seq_len=task_cfg.seq_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_steps = 2000
    log_every = 100

    for step in range(1, n_steps + 1):
        batch = gen.sample_batch()
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        induction_mask = batch["induction_mask"].to(device)

        logits = model(input_ids)  # (b, L, vocab)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            acc = induction_accuracy(logits, target_ids, induction_mask)
            print(f"step {step:5d}  loss {loss.item():.4f}  induction_acc {acc:.3f}")

    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "results/baseline_transformer.pt")
    print("saved results/baseline_transformer.pt")

    # Next: model.run_with_cache(input_ids) on a fresh batch, inspect
    # attention patterns per head/layer, look for the diagonal-offset
    # induction pattern (see 01b_analyze_baseline.py, or the ARENA
    # induction-heads exercises for the canonical visualization code).


if __name__ == "__main__":
    main()
