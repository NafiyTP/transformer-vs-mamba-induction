"""
Eigenvalue / pole analysis of the Mamba SSM's effective A matrix.

This is the part that most directly uses the TSIA202b background: a diagonal
linear recurrence h_t = A_bar * h_{t-1} + ... is exactly the kind of object
you analyze via pole locations for an ARMA filter or a Kalman filter's
state-transition matrix. Since MinimalMambaBlock parametrizes
A = -exp(A_log) as one real value per (channel, state-dim) pair, "eigenvalues"
here are just the diagonal entries themselves (already diagonalized by
construction) -- the interesting object is how they combine with the
per-timestep delta_t to give an *effective* discrete pole
exp(delta_t * A) at each step, i.e. a time-varying (input-selective) pole,
which is precisely what makes Mamba's "selectivity" different from a
fixed-pole linear filter.

Core question for this script: do the channels identified as causally
important for induction (from the patching results) have systematically
different pole structure (e.g. poles closer to 1, i.e. slower decay / longer
memory) than channels that patching found unimportant?
"""

import matplotlib.pyplot as plt
import torch

from src.models.minimal_mamba import MinimalMambaBlock


def continuous_poles(block: MinimalMambaBlock) -> torch.Tensor:
    """Raw continuous-time poles, shape (d_inner, d_state). These are what
    A_log parametrizes directly, before any input-dependent discretization."""
    return block.effective_A()


def effective_discrete_poles(
    block: MinimalMambaBlock, delta: torch.Tensor
) -> torch.Tensor:
    """
    delta: (d_inner,) or (batch, d_inner) -- a representative delta_t, e.g.
    averaged over a batch of induction-relevant timesteps.
    Returns exp(delta * A), the discrete-time pole actually seen at that step.
    A pole near 1 means near-unit decay, i.e. long memory; near 0 means the
    channel forgets almost immediately.
    """
    A = block.effective_A()  # (d_inner, d_state)
    if delta.dim() == 1:
        delta = delta.unsqueeze(-1)  # (d_inner, 1)
    return torch.exp(delta * A)


def plot_pole_comparison(
    poles_important: torch.Tensor,
    poles_other: torch.Tensor,
    save_path: str = "results/pole_comparison.png",
):
    """
    Scatter of pole magnitude for channels patching flagged as important for
    induction vs the rest. The hypothesis to check visually: important
    channels cluster at poles closer to 1 (longer effective memory), the way
    a resonant/near-marginally-stable pole in a classical filter is needed to
    "hold onto" information over many steps.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(
        poles_important.flatten().detach().cpu().numpy(),
        bins=30,
        alpha=0.6,
        label="induction-important channels",
        density=True,
    )
    ax.hist(
        poles_other.flatten().detach().cpu().numpy(),
        bins=30,
        alpha=0.6,
        label="other channels",
        density=True,
    )
    ax.set_xlabel("discrete pole value (closer to 1 = longer memory)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"saved {save_path}")


if __name__ == "__main__":
    # Smoke test with a freshly initialized (untrained) block, just to check
    # the plumbing -- real analysis needs a trained model and the patching
    # script's list of important channel indices.
    block = MinimalMambaBlock(d_model=64, d_state=16)
    poles = continuous_poles(block)
    print("continuous pole range:", poles.min().item(), poles.max().item())
