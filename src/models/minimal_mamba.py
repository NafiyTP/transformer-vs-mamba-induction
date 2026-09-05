"""
Minimal, pure-PyTorch, sequential (non-parallel-scan) selective SSM block,
following Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces".

Not the official CUDA-accelerated `mamba-ssm` package:
  - avoids build/CUDA-kernel headaches for a project meant to run on modest
    hardware.
  - the explicit Python loop over time keeps every intermediate hidden state
    h_t addressable, which the activation patching and eigenvalue analysis
    later on need directly.
  - at the tiny scale used here (sequences of ~64 tokens, hidden dims in the
    hundreds), the O(L) Python loop is not a real bottleneck.

Notation follows the paper:
  x_t        : input at time t, shape (d_inner,)
  A          : (d_inner, d_state), fixed (structured) transition matrix,
               parametrized as A = -exp(A_log) so eigenvalues stay negative
               real (stable, and this is exactly the pole location the
               eigenvalue analysis inspects later).
  B_t, C_t   : (d_inner, d_state) and (d_state,) -- input-dependent
               ("selective"), computed by linear projections of x_t. This
               input-dependence is Mamba's key departure from S4/vanilla SSMs.
  delta_t    : (d_inner,), input-dependent step size, softplus-activated.
  Discretization (zero-order hold, as in the paper):
    A_bar_t = exp(delta_t * A)
    B_bar_t = delta_t * B_t
  Recurrence:
    h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
    y_t = C_t . h_t + D * x_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner_mult: int = 2,
        conv_kernel: int = 4,
        dt_rank: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner_mult * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # Input projection: splits into the "main" branch (x) and the gate (z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Short causal conv before the SSM, as in the paper (local mixing)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
        )

        # Selective parameters: B, C, and delta are all *functions of x_t*
        self.x_to_bc_delta = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is NOT input-dependent (only its discretization via delta_t is) --
        # this is the object the eigenvalue analysis looks at per channel.
        A_log_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_log_init.unsqueeze(0).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # dt_proj.bias is initialized so that softplus(bias) covers a spread
        # of time constants (as in the official Mamba init), instead of all
        # channels starting with the same delta -- this matters a lot for
        # how quickly channels specialize during training.
        with torch.no_grad():
            dt_min, dt_max = 0.001, 0.1
            dt = torch.exp(
                torch.rand(self.d_inner) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
                + torch.log(torch.tensor(dt_min))
            )
            inv_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_softplus)
            # Zero-init dt_proj.weight: at initialization this makes delta_t
            # purely a function of the per-channel bias above (a clean,
            # controlled spread of time constants / memory horizons), instead
            # of being dominated by a random input-dependent term that wipes
            # out that spread. Input-dependence (the actual "selective" part
            # of selective SSMs) then grows from zero as training moves this
            # weight away from 0 -- this is what the official Mamba init does
            # too, and it matters a lot: without it, channels that should
            # have a long memory horizon get an effectively random delta at
            # every step instead, and long-range information decays within a
            # handful of timesteps regardless of the bias.
            self.dt_proj.weight.zero_()

    def effective_A(self) -> torch.Tensor:
        """A = -exp(A_log), shape (d_inner, d_state). Real negative eigenvalues
        by construction -- this is the pole/eigenvalue analysis's input.

        A_log is clamped before exponentiating as a numerical-stability guard:
        with weight_decay=0 (needed to let the induction circuit assemble --
        see the note in scripts/02_train_mamba.py), nothing pulls A_log back
        towards 0, and over enough steps it can drift high enough that
        exp(A_log) overflows to inf. Then, at any timestep where softplus
        rounds delta to exactly 0.0 for that channel, delta * A becomes
        0 * (-inf) = nan (this actually happened: loss went to nan around
        step 550 on the full-scale task before this was added). Clamping at
        20 (exp(20) ~= 4.9e8) keeps A_bar's dynamic range well inside float32
        without constraining anything the model actually needs -- learned |A|
        this large would mean sub-single-step decay anyway.
        """
        return -torch.exp(self.A_log.clamp(max=20.0))

    def forward(
        self,
        x: torch.Tensor,
        return_hidden_states: bool = False,
        hidden_patch: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    ):
        """
        x: (batch, seq_len, d_model)

        hidden_patch, if given, is (source_hidden_states, channel_mask):
          - source_hidden_states: (batch, seq_len, d_inner, d_state), typically
            the cached hidden states from a *clean* run (see
            MinimalMambaBlock(..., return_hidden_states=True) on that run).
          - channel_mask: (d_inner,) bool, which channels to overwrite at
            every timestep with the clean run's hidden state, instead of this
            run's own recurrence. This is the activation-patching primitive
            used to test which channels are causally responsible for
            induction: patch a candidate group of channels on a *corrupted*
            input and see how much induction_accuracy is recovered.

        Returns y: (batch, seq_len, d_model), and optionally the full stack of
        hidden states h_t (batch, seq_len, d_inner, d_state) for interpretability.
        """
        b, L, _ = x.shape
        xz = self.in_proj(x)  # (b, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (b, L, d_inner)

        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)  # (b, L, d_inner)

        bc_delta = self.x_to_bc_delta(x_conv)  # (b, L, dt_rank + 2*d_state)
        delta_raw, B, C = torch.split(
            bc_delta, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # small floor: softplus can round to exactly 0.0 in float32 for very
        # negative inputs, which combined with a large |A| (see effective_A's
        # clamp docstring) is the other half of the 0 * (-inf) = nan failure
        # mode -- belt-and-suspenders alongside the A_log clamp.
        delta = F.softplus(self.dt_proj(delta_raw)) + 1e-6  # (b, L, d_inner)

        A = self.effective_A()  # (d_inner, d_state)

        h = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        hidden_states = [] if return_hidden_states else None

        if hidden_patch is not None:
            source_hidden_states, channel_mask = hidden_patch
            channel_mask = channel_mask.to(x.device)

        for t in range(L):
            delta_t = delta[:, t, :]  # (b, d_inner)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A)  # (b, d_inner, d_state)
            B_t = B[:, t, :]  # (b, d_state)
            x_t = x_conv[:, t, :]  # (b, d_inner)
            B_bar_x = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)

            h = A_bar * h + B_bar_x  # (b, d_inner, d_state)

            if hidden_patch is not None:
                # overwrite the masked channels with the clean run's hidden
                # state at this same timestep, for every subsequent step too
                # (since h is recurrent, this also propagates the patch
                # forward through the rest of the sequence, which is the
                # intended causal-effect measurement).
                patched_slice = source_hidden_states[:, t]  # (b, d_inner, d_state)
                mask = channel_mask.view(1, -1, 1)
                h = torch.where(mask, patched_slice, h)

            C_t = C[:, t, :]  # (b, d_state)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t  # (b, d_inner)

            ys.append(y_t)
            if return_hidden_states:
                hidden_states.append(h.clone())

        y = torch.stack(ys, dim=1)  # (b, L, d_inner)
        y = y * F.silu(z)
        out = self.out_proj(y)  # (b, L, d_model)

        if return_hidden_states:
            return out, torch.stack(hidden_states, dim=1)  # (b, L, d_inner, d_state)
        return out


class TinyMambaLM(nn.Module):
    """Stack of MinimalMambaBlocks + embedding/unembedding, sized to match
    the tiny transformer baseline for a fair comparison."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2, d_state: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MinimalMambaBlock(d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        patch_spec: "dict[int, tuple[torch.Tensor, torch.Tensor]] | None" = None,
    ) -> torch.Tensor:
        """
        patch_spec: optional {layer_idx: (source_hidden_states, channel_mask)},
        forwarded to the matching block's `hidden_patch` argument. Used by the
        activation patching script to override one layer's hidden state with
        a clean run's, while every other layer runs normally.
        """
        x = self.embed(input_ids)
        for layer_idx, block in enumerate(self.blocks):
            patch = patch_spec.get(layer_idx) if patch_spec else None
            out = block(x, hidden_patch=patch)
            x = x + out
        x = self.norm(x)
        return self.unembed(x)

    def forward_with_hidden_states(self, input_ids: torch.Tensor):
        """Runs the model and also returns every layer's full hidden-state
        stack (list indexed by layer), for use as the "clean" reference in
        activation patching."""
        x = self.embed(input_ids)
        all_hidden_states = []
        for block in self.blocks:
            out, h = block(x, return_hidden_states=True)
            all_hidden_states.append(h)
            x = x + out
        x = self.norm(x)
        logits = self.unembed(x)
        return logits, all_hidden_states


if __name__ == "__main__":
    model = TinyMambaLM(vocab_size=50, d_model=64, n_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    dummy = torch.randint(0, 50, (2, 32))
    out = model(dummy)
    print("output shape:", out.shape)
