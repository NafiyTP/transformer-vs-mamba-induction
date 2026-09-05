"""
Load the trained baseline transformer checkpoint, confirm induction_accuracy,
and check for the diagonal-offset attention pattern characteristic of
induction heads (per-head, per-layer scan).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.induction_task import InductionTaskConfig, InductionTaskGenerator, induction_accuracy
from src.models.tiny_transformer import build_tiny_induction_transformer


def main():
    device = "cpu"
    task_cfg = InductionTaskConfig(vocab_size=50, seq_len=64, batch_size=256, seed=123)
    gen = InductionTaskGenerator(task_cfg)

    model = build_tiny_induction_transformer(vocab_size=task_cfg.vocab_size, seq_len=task_cfg.seq_len)
    state_dict = torch.load("results/baseline_transformer.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 1) Headline metric on a fresh held-out batch (different seed than training)
    batch = gen.sample_batch()
    with torch.no_grad():
        logits = model(batch["input_ids"])
    acc = induction_accuracy(logits, batch["target_ids"], batch["induction_mask"])
    overall_acc = (logits.argmax(-1) == batch["target_ids"]).float().mean().item()
    print(f"induction_accuracy (held-out, seed=123): {acc:.4f}")
    print(f"overall next-token accuracy (mostly unpredictable by design): {overall_acc:.4f}")

    # 2) Attention pattern scan: for one example sequence, per layer/head,
    # check how much attention mass lands on "the position right after the
    # earlier occurrence of the current token" -- the induction-head signature.
    single_batch_cfg = InductionTaskConfig(vocab_size=50, seq_len=64, batch_size=1, seed=7)
    single_gen = InductionTaskGenerator(single_batch_cfg)
    ex = single_gen.sample_batch()
    input_ids = ex["input_ids"]  # (1, L)
    induction_mask = ex["induction_mask"][0]  # (L,)

    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq = input_ids[0].tolist()
    L = len(seq)

    # "ideal" induction source: for each destination position, the position
    # right after the previous occurrence of the same token (if any)
    ideal_source = [-1] * L
    for i in range(1, L):
        tok = seq[i]
        for j in range(i - 1, -1, -1):
            if seq[j] == tok:
                if j + 1 <= i:
                    ideal_source[i] = j + 1
                break

    scores = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        pattern = cache["pattern", layer][0]  # (n_heads, L, L) — dest x src
        for head in range(n_heads):
            mass = []
            for i in range(1, L):
                src = ideal_source[i]
                if src == -1 or src >= i:
                    continue
                mass.append(pattern[head, i, src].item())
            scores[layer, head] = float(np.mean(mass)) if mass else float("nan")

    print("\nInduction-pattern attention mass per (layer, head):")
    for layer in range(n_layers):
        for head in range(n_heads):
            print(f"  layer {layer} head {head}: {scores[layer, head]:.3f}")

    best_layer, best_head = np.unravel_index(np.nanargmax(scores), scores.shape)
    print(f"\nStrongest induction-head candidate: layer {best_layer}, head {best_head} "
          f"(avg attention mass on induction source: {scores[best_layer, best_head]:.3f})")

    Path("results").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(scores, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title("Induction attention mass\nper (layer, head)")
    fig.colorbar(im, ax=ax, label="avg attn on induction source")
    fig.tight_layout()
    fig.savefig("results/stage0_induction_head_scores.png", dpi=150)
    print("saved results/stage0_induction_head_scores.png")

    # Full attention pattern of the best head, to visually confirm the
    # characteristic diagonal-offset stripe.
    pattern_best = cache["pattern", best_layer][0, best_head].numpy()
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    im2 = ax2.imshow(pattern_best, cmap="viridis", vmin=0, vmax=pattern_best.max())
    ax2.set_xlabel("source position")
    ax2.set_ylabel("destination position")
    ax2.set_title(f"Attention pattern: layer {best_layer}, head {best_head}")
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    fig2.savefig("results/stage0_best_head_attention_pattern.png", dpi=150)
    print("saved results/stage0_best_head_attention_pattern.png")

    np.save("results/stage0_head_scores.npy", scores)

    return {
        "held_out_induction_acc": acc,
        "best_layer": int(best_layer),
        "best_head": int(best_head),
        "best_score": float(scores[best_layer, best_head]),
    }


if __name__ == "__main__":
    main()
