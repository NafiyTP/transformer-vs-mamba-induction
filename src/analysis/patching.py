"""
Activation patching utilities for circuit localization.

The recipe (standard interpretability pattern, adapted to both model types):
  1. Build a "clean" batch (normal induction sequences) and a "corrupted"
     batch (same sequences but with the *source* occurrence of the repeated
     chunk replaced by fresh random tokens, so the bigram no longer actually
     repeats -- induction can't fire from the corrupted input alone).
  2. Run the clean batch, cache all intermediate activations.
  3. Run the corrupted batch, but at each candidate site (a transformer head,
     or a Mamba channel/hidden-state slice), splice in the cached clean
     activation instead of the corrupted one.
  4. Measure how much of the clean induction_accuracy is recovered.
     A site whose patching recovers a lot of accuracy is causally implicated.
"""

from typing import Callable

import torch

from src.data.induction_task import induction_accuracy


def make_corrupted_batch(clean_batch: dict, vocab_size: int, seed: int = 0) -> dict:
    """
    Corrupt a clean induction batch by replacing the *source* occurrence of
    the repeated chunk with fresh random tokens, breaking the "this bigram
    was seen before" signal while keeping sequence length, targets, and the
    induction_mask identical to the clean batch -- so induction_accuracy on
    the corrupted run and the clean run are directly comparable, and any gap
    between them is attributable to the corruption itself.
    """
    rng = torch.Generator().manual_seed(seed)
    input_ids = clean_batch["input_ids"].clone()
    chunk_positions = clean_batch["chunk_positions"]

    for i, (src_start, src_end, dst_start, dst_end) in enumerate(chunk_positions):
        repeat_len = src_end - src_start
        new_tokens = torch.randint(
            0, vocab_size, (repeat_len,), generator=rng
        )
        # input_ids = original seq[:-1], so index j in input_ids is seq[j];
        # src_end-1 is guaranteed < dst_start < seq_len - repeat_len <= seq_len-1,
        # so src_end-1 is always a valid input_ids index.
        input_ids[i, src_start:src_end] = new_tokens

    corrupted_batch = dict(clean_batch)
    corrupted_batch["input_ids"] = input_ids
    return corrupted_batch


# ---------------------------------------------------------------------------
# Transformer-side patching (per attention head, via TransformerLens hooks)
# ---------------------------------------------------------------------------


@torch.no_grad()
def transformer_patch_heads(
    model,
    clean_batch: dict,
    corrupted_batch: dict,
) -> "torch.Tensor":
    """
    For each (layer, head), patch that head's mixed-value output (hook_z)
    from the clean run into the corrupted run, and record recovered
    induction_accuracy. Returns a (n_layers, n_heads) tensor of scores in
    [0, 1] where 1 = full recovery of clean-level induction_accuracy and 0 =
    no better than the corrupted run's own (typically near-chance) accuracy.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_logits, clean_cache = model.run_with_cache(clean_batch["input_ids"])
    clean_acc = induction_accuracy(
        clean_logits, clean_batch["target_ids"], clean_batch["induction_mask"]
    )

    corrupted_logits = model(corrupted_batch["input_ids"])
    corrupted_acc = induction_accuracy(
        corrupted_logits, corrupted_batch["target_ids"], corrupted_batch["induction_mask"]
    )

    scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        clean_z = clean_cache["z", layer]  # (batch, seq, n_heads, d_head)
        for head in range(n_heads):

            def hook_fn(z, hook, layer=layer, head=head):
                z = z.clone()
                z[:, :, head, :] = clean_z[:, :, head, :]
                return z

            patched_logits = model.run_with_hooks(
                corrupted_batch["input_ids"],
                fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)],
            )
            patched_acc = induction_accuracy(
                patched_logits, corrupted_batch["target_ids"], corrupted_batch["induction_mask"]
            )
            denom = max(clean_acc - corrupted_acc, 1e-6)
            scores[layer, head] = (patched_acc - corrupted_acc) / denom

    return scores, clean_acc, corrupted_acc


# ---------------------------------------------------------------------------
# Mamba-side patching (per channel group, via MinimalMambaBlock.hidden_patch)
# ---------------------------------------------------------------------------


@torch.no_grad()
def mamba_patch_channel_groups(
    model,
    clean_batch: dict,
    corrupted_batch: dict,
    group_size: int = 16,
) -> "tuple[torch.Tensor, float, float]":
    """
    For each layer and each contiguous group of `group_size` d_inner
    channels, patch that group's hidden state h_t (all timesteps) from the
    clean run into the corrupted run, and record recovered induction_accuracy.

    Grouped rather than per-channel: d_inner is typically 128, so per-channel
    patching would mean 128 x n_layers forward passes; grouping into chunks
    of `group_size` gives a fast first pass to find the interesting region,
    which can then be refined with a smaller group_size just in that region.

    Returns (scores of shape (n_layers, n_groups), clean_acc, corrupted_acc).
    """
    clean_logits, clean_hidden_states = model.forward_with_hidden_states(
        clean_batch["input_ids"]
    )
    clean_acc = induction_accuracy(
        clean_logits, clean_batch["target_ids"], clean_batch["induction_mask"]
    )

    corrupted_logits = model(corrupted_batch["input_ids"])
    corrupted_acc = induction_accuracy(
        corrupted_logits, corrupted_batch["target_ids"], corrupted_batch["induction_mask"]
    )

    n_layers = len(model.blocks)
    d_inner = model.blocks[0].d_inner
    n_groups = (d_inner + group_size - 1) // group_size

    scores = torch.zeros(n_layers, n_groups)

    for layer in range(n_layers):
        source_h = clean_hidden_states[layer]  # (b, L, d_inner, d_state)
        for g in range(n_groups):
            lo, hi = g * group_size, min((g + 1) * group_size, d_inner)
            channel_mask = torch.zeros(d_inner, dtype=torch.bool)
            channel_mask[lo:hi] = True

            patch_spec = {layer: (source_h, channel_mask)}
            patched_logits = model(corrupted_batch["input_ids"], patch_spec=patch_spec)
            patched_acc = induction_accuracy(
                patched_logits, corrupted_batch["target_ids"], corrupted_batch["induction_mask"]
            )
            denom = max(clean_acc - corrupted_acc, 1e-6)
            scores[layer, g] = (patched_acc - corrupted_acc) / denom

    return scores, clean_acc, corrupted_acc
