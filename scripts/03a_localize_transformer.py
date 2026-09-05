"""
Transformer side: activation-patch every (layer, head) and see which ones
causally recover induction_accuracy on a corrupted input. This should light
up the same heads that the attention-pattern scan already flagged
(01b_analyze_baseline.py) -- patching is the causal complement to that
correlational (attention-pattern) evidence, and agreement between the two is
itself a small result worth reporting.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.induction_task import InductionTaskConfig, InductionTaskGenerator
from src.models.tiny_transformer import build_tiny_induction_transformer
from src.analysis.patching import make_corrupted_batch, transformer_patch_heads


def main():
    task_cfg = InductionTaskConfig(vocab_size=50, seq_len=64, batch_size=64, seed=321)
    gen = InductionTaskGenerator(task_cfg)

    model = build_tiny_induction_transformer(vocab_size=task_cfg.vocab_size, seq_len=task_cfg.seq_len)
    model.load_state_dict(torch.load("results/baseline_transformer.pt", map_location="cpu"))
    model.eval()

    clean_batch = gen.sample_batch()
    corrupted_batch = make_corrupted_batch(clean_batch, task_cfg.vocab_size, seed=999)

    scores, clean_acc, corrupted_acc = transformer_patch_heads(model, clean_batch, corrupted_batch)

    print(f"clean induction_acc:     {clean_acc:.4f}")
    print(f"corrupted induction_acc: {corrupted_acc:.4f}")
    print("\nPatching recovery score per (layer, head), 1.0 = full recovery:")
    for layer in range(scores.shape[0]):
        for head in range(scores.shape[1]):
            print(f"  layer {layer} head {head}: {scores[layer, head].item():+.3f}")

    Path("results").mkdir(exist_ok=True)
    np.save("results/stage2_transformer_patch_scores.npy", scores.numpy())

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(scores.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(scores.shape[1]))
    ax.set_yticks(range(scores.shape[0]))
    ax.set_title("Transformer: activation patching\nrecovery score per (layer, head)")
    fig.colorbar(im, ax=ax, label="recovered induction_acc (normalized)")
    fig.tight_layout()
    fig.savefig("results/stage2_transformer_patch_heatmap.png", dpi=150)
    print("\nsaved results/stage2_transformer_patch_heatmap.png")


if __name__ == "__main__":
    main()
