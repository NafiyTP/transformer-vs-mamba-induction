"""
Tiny 2-layer attention-only transformer, via TransformerLens, used as the
baseline before touching Mamba.

Using TransformerLens instead of a hand-rolled model gets run_with_cache /
run_with_hooks for free, which is what activation patching and attention
pattern inspection need. It's also what the ARENA interpretability
curriculum uses, so the standard induction-head exercises transfer directly.
"""

from transformer_lens import HookedTransformer, HookedTransformerConfig


def build_tiny_induction_transformer(
    vocab_size: int,
    seq_len: int,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    d_head: int = 32,
    attn_only: bool = True,
    seed: int = 0,
) -> HookedTransformer:
    """
    Attention-only by default: Olsson et al.'s original induction-head result
    is cleanest in attention-only transformers (no MLPs), which makes the
    attention-pattern-based circuit unambiguous. Flip attn_only=False if you
    want to check robustness with MLPs present.
    """
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=seq_len,
        d_head=d_head,
        n_heads=n_heads,
        d_vocab=vocab_size,
        act_fn="relu",
        attn_only=attn_only,
        seed=seed,
        normalization_type="LNPre",
    )
    return HookedTransformer(cfg)


if __name__ == "__main__":
    model = build_tiny_induction_transformer(vocab_size=50, seq_len=64)
    print(model.cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
