"""
Pole/eigenvalue analysis of layer 1's channels, correlated against the
per-channel importance scores from 03b_localize_mamba.py.

Hypothesis (the TSIA202b-flavored one motivating this whole project): if a
channel matters for induction, it needs to hold information across the
gap between the repeated bigram's two occurrences, so it should have a
*slower* discrete pole (closer to 1) than a channel that doesn't matter --
exactly like needing a pole near the unit circle for long memory in a
classical linear filter or a Kalman filter's state transition.

Each channel has d_state=16 poles (one per state dimension), and delta_t is
itself input-dependent, so "the pole" isn't a single fixed number per
channel -- we use each channel's *slowest* (largest-magnitude, closest to 1)
discrete pole, evaluated at delta averaged over real induction-relevant
timesteps from actual data, as a one-number summary of that channel's
achievable memory horizon.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.induction_task import InductionTaskConfig, InductionTaskGenerator
from src.models.minimal_mamba import TinyMambaLM


def main():
    task_cfg = InductionTaskConfig(vocab_size=50, seq_len=64, batch_size=64, seed=321)
    gen = InductionTaskGenerator(task_cfg)

    model = TinyMambaLM(vocab_size=task_cfg.vocab_size, d_model=64, n_layers=2)
    model.load_state_dict(torch.load("results/mamba_baseline.pt", map_location="cpu"))
    model.eval()

    block1 = model.blocks[1]
    d_inner = block1.d_inner

    # Get a representative delta_t for layer 1 by running a real batch and
    # averaging delta over induction-relevant positions (where the channel's
    # memory is actually being exercised).
    batch = gen.sample_batch()
    with torch.no_grad():
        x = model.embed(batch["input_ids"])
        out0 = model.blocks[0](x)
        x1_in = x + out0  # input to layer 1

        xz = block1.in_proj(x1_in)
        x_in, _z = xz.chunk(2, dim=-1)
        x_conv = block1.conv1d(x_in.transpose(1, 2))[..., : x1_in.shape[1]].transpose(1, 2)
        x_conv = torch.nn.functional.silu(x_conv)
        bc_delta = block1.x_to_bc_delta(x_conv)
        delta_raw, _B, _C = torch.split(
            bc_delta, [block1.dt_rank, block1.d_state, block1.d_state], dim=-1
        )
        delta = torch.nn.functional.softplus(block1.dt_proj(delta_raw)) + 1e-6  # (b, L, d_inner)

        mask = batch["induction_mask"]  # (b, L)
        # average delta over induction-relevant (batch, position) pairs, per channel
        mask_f = mask.unsqueeze(-1).float()
        avg_delta = (delta * mask_f).sum(dim=(0, 1)) / mask_f.sum(dim=(0, 1)).clamp(min=1)
        # (d_inner,)

        A = block1.effective_A()  # (d_inner, d_state)
        discrete_poles = torch.exp(avg_delta.unsqueeze(-1) * A)  # (d_inner, d_state)
        slowest_pole_per_channel = discrete_poles.max(dim=-1).values  # closest to 1 = slowest

    per_channel_scores = np.load("results/stage2_mamba_per_channel_scores.npy")

    Path("results").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(slowest_pole_per_channel.numpy(), per_channel_scores, alpha=0.6, s=20)
    ax.set_xlabel("channel's slowest discrete pole (closer to 1 = longer memory)")
    ax.set_ylabel("individual patching importance score")
    ax.set_title("Layer 1: does memory horizon predict\ncausal importance for induction?")
    fig.tight_layout()
    fig.savefig("results/stage3_pole_vs_importance.png", dpi=150)
    print("saved results/stage3_pole_vs_importance.png")

    corr = np.corrcoef(slowest_pole_per_channel.numpy(), per_channel_scores)[0, 1]
    print(f"Pearson correlation (pole, importance): {corr:+.3f}")

    # split into top-20% vs bottom-20% important channels, compare pole distributions
    order = np.argsort(per_channel_scores)
    n = len(order)
    bottom = order[: n // 5]
    top = order[-n // 5 :]
    print(f"\nmean slowest-pole, bottom 20% important channels: {slowest_pole_per_channel.numpy()[bottom].mean():.4f}")
    print(f"mean slowest-pole, top 20% important channels:    {slowest_pole_per_channel.numpy()[top].mean():.4f}")

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.hist(slowest_pole_per_channel.numpy()[bottom], bins=20, alpha=0.6, label="bottom 20% importance", density=True)
    ax2.hist(slowest_pole_per_channel.numpy()[top], bins=20, alpha=0.6, label="top 20% importance", density=True)
    ax2.set_xlabel("slowest discrete pole")
    ax2.set_ylabel("density")
    ax2.legend()
    ax2.set_title("Pole distribution: important vs unimportant channels")
    fig2.tight_layout()
    fig2.savefig("results/stage3_pole_distribution_comparison.png", dpi=150)
    print("saved results/stage3_pole_distribution_comparison.png")


if __name__ == "__main__":
    main()
