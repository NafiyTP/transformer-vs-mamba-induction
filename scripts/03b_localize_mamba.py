"""
Mamba side: activation patching.

First pass (per-layer, group_size=16) showed near-zero recovery for any
single 16-channel group in isolation -- unlike the transformer, where 2
individual heads each recover ~70-80%. A follow-up whole-layer test showed
why: patching layer 0's full hidden state alone barely matters (+0.5%),
patching layer 1's full hidden state alone recovers ~52%, and patching BOTH
layers fully recovers ~100%. So the circuit is not sparse in the channel
dimension the way the transformer's is in the head dimension -- it needs a
large fraction of layer 1's ~128 channels jointly, not any single small
subgroup, and there's a secondary contribution from layer 0 that only
matters once layer 1 is already patched.

This script reproduces both findings and then sweeps group size within layer
1 (with layer 0 held clean via a full patch) to characterize how much of
layer 1 is actually needed -- i.e. how distributed the circuit really is.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.induction_task import InductionTaskConfig, InductionTaskGenerator, induction_accuracy
from src.models.minimal_mamba import TinyMambaLM
from src.analysis.patching import make_corrupted_batch


def score_of(patched_acc, clean_acc, corrupted_acc):
    return (patched_acc - corrupted_acc) / max(clean_acc - corrupted_acc, 1e-6)


def main():
    task_cfg = InductionTaskConfig(vocab_size=50, seq_len=64, batch_size=64, seed=321)
    gen = InductionTaskGenerator(task_cfg)

    model = TinyMambaLM(vocab_size=task_cfg.vocab_size, d_model=64, n_layers=2)
    model.load_state_dict(torch.load("results/mamba_baseline.pt", map_location="cpu"))
    model.eval()

    clean_batch = gen.sample_batch()
    corrupted_batch = make_corrupted_batch(clean_batch, task_cfg.vocab_size, seed=999)

    with torch.no_grad():
        clean_logits, clean_hs = model.forward_with_hidden_states(clean_batch["input_ids"])
        clean_acc = induction_accuracy(clean_logits, clean_batch["target_ids"], clean_batch["induction_mask"])
        corrupted_logits = model(corrupted_batch["input_ids"])
        corrupted_acc = induction_accuracy(
            corrupted_logits, corrupted_batch["target_ids"], corrupted_batch["induction_mask"]
        )
        print(f"clean induction_acc:     {clean_acc:.4f}")
        print(f"corrupted induction_acc: {corrupted_acc:.4f}\n")

        # --- whole-layer patches ---
        d_inner = model.blocks[0].d_inner
        results = {}
        for name, spec in [
            ("layer 0 only (full)", {0: (clean_hs[0], torch.ones(d_inner, dtype=torch.bool))}),
            ("layer 1 only (full)", {1: (clean_hs[1], torch.ones(d_inner, dtype=torch.bool))}),
            (
                "both layers (full)",
                {
                    0: (clean_hs[0], torch.ones(d_inner, dtype=torch.bool)),
                    1: (clean_hs[1], torch.ones(d_inner, dtype=torch.bool)),
                },
            ),
        ]:
            patched_logits = model(corrupted_batch["input_ids"], patch_spec=spec)
            patched_acc = induction_accuracy(
                patched_logits, corrupted_batch["target_ids"], corrupted_batch["induction_mask"]
            )
            s = score_of(patched_acc, clean_acc, corrupted_acc)
            results[name] = s
            print(f"{name:22s}: patched_acc={patched_acc:.4f}  score={s:+.4f}")

        # --- how much of layer 1 is needed, with layer 0 held clean? ---
        print("\nSweeping how much of layer 1 needs patching (layer 0 held fully clean):")
        full_mask0 = torch.ones(d_inner, dtype=torch.bool)
        fractions = [1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0]
        sweep_scores = []
        rng = torch.Generator().manual_seed(0)
        for frac in fractions:
            k = int(round(frac * d_inner))
            perm = torch.randperm(d_inner, generator=rng)
            mask1 = torch.zeros(d_inner, dtype=torch.bool)
            mask1[perm[:k]] = True
            spec = {0: (clean_hs[0], full_mask0), 1: (clean_hs[1], mask1)}
            patched_logits = model(corrupted_batch["input_ids"], patch_spec=spec)
            patched_acc = induction_accuracy(
                patched_logits, corrupted_batch["target_ids"], corrupted_batch["induction_mask"]
            )
            s = score_of(patched_acc, clean_acc, corrupted_acc)
            sweep_scores.append(s)
            print(f"  {int(frac*100):3d}% of layer-1 channels ({k:3d}/{d_inner}): score={s:+.4f}")

    Path("results").mkdir(exist_ok=True)
    np.save(
        "results/stage2_mamba_patch_scores.npy",
        {"whole_layer": results, "layer1_fraction_sweep": dict(zip(fractions, sweep_scores))},
        allow_pickle=True,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([f * 100 for f in fractions], sweep_scores, marker="o")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="full recovery")
    ax.set_xlabel("% of layer-1 channels patched (random subset)")
    ax.set_ylabel("recovered induction_acc (normalized)")
    ax.set_title("Mamba: how distributed is the circuit\nwithin layer 1?")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/stage2_mamba_patch_sweep.png", dpi=150)
    print("\nsaved results/stage2_mamba_patch_sweep.png")


if __name__ == "__main__":
    main()
