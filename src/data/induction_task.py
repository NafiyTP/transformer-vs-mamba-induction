"""
Synthetic induction / repeated-token task, following Olsson et al. (2022).

Random token sequences with a repeated chunk inserted somewhere in the
middle. A model that's learned induction should, on seeing the first token
of a bigram it already saw earlier in the sequence, predict whatever
followed it last time.

Only depends on numpy + torch so the exact same generator can be reused for
both models without any changes, which is what makes the transformer vs
Mamba comparison fair.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class InductionTaskConfig:
    vocab_size: int = 50          # small vocab keeps models tiny
    seq_len: int = 64             # total sequence length
    min_prefix_len: int = 8       # random tokens before the first repeat
    batch_size: int = 64
    seed: int | None = None


class InductionTaskGenerator:
    """
    Produces batches of (input_ids, target_ids) for the induction task.

    Construction of one sequence:
      1. Sample `seq_len` random tokens from [0, vocab_size).
      2. Pick a random split point after `min_prefix_len`.
      3. Copy a chunk of tokens from before the split point to after it,
         so the exact same bigram context reappears later in the sequence.
      4. Targets are the standard next-token targets (input shifted by one);
         induction accuracy is measured specifically on the positions where
         the copied chunk repeats, since that's where induction is needed
         to predict correctly (elsewhere the target is unpredictable, which
         is intentional -- it keeps the model from cheating any other way).
    """

    def __init__(self, config: InductionTaskConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def _make_one_sequence(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        cfg = self.cfg
        seq = self.rng.integers(0, cfg.vocab_size, size=cfg.seq_len)

        # Choose a repeat chunk length and where to copy it to.
        max_repeat_len = max(2, (cfg.seq_len - cfg.min_prefix_len) // 3)
        repeat_len = int(self.rng.integers(2, max_repeat_len + 1))

        src_start = int(
            self.rng.integers(0, cfg.min_prefix_len + 1)
        )  # somewhere in the prefix
        src_end = src_start + repeat_len

        dst_start = int(
            self.rng.integers(src_end + 1, cfg.seq_len - repeat_len)
        )
        dst_end = dst_start + repeat_len

        seq[dst_start:dst_end] = seq[src_start:src_end]

        # induction_mask marks positions where the *previous* token starts a
        # bigram that already occurred earlier -- i.e. positions
        # [dst_start+1, dst_end) are where induction should fire, since the
        # model has already seen "token at dst_start" followed by
        # "token at dst_start+1" back in the source chunk.
        induction_mask = np.zeros(cfg.seq_len, dtype=bool)
        induction_mask[dst_start + 1 : dst_end] = True

        return seq, induction_mask, (src_start, src_end, dst_start, dst_end)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        seqs = np.zeros((cfg.batch_size, cfg.seq_len), dtype=np.int64)
        masks = np.zeros((cfg.batch_size, cfg.seq_len), dtype=bool)
        chunk_positions = []

        for i in range(cfg.batch_size):
            seq, mask, positions = self._make_one_sequence()
            seqs[i] = seq
            masks[i] = mask
            chunk_positions.append(positions)

        input_ids = torch.from_numpy(seqs[:, :-1]).long()
        target_ids = torch.from_numpy(seqs[:, 1:]).long()
        # shift mask to align with targets (mask was defined on token positions,
        # targets are inputs shifted left by one)
        induction_mask = torch.from_numpy(masks[:, 1:]).bool()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "induction_mask": induction_mask,
            # (src_start, src_end, dst_start, dst_end) per sequence, original
            # (pre-shift) indexing. patching.py uses this to build corrupted
            # versions of a batch.
            "chunk_positions": chunk_positions,
        }


def induction_accuracy(
    logits: torch.Tensor, target_ids: torch.Tensor, induction_mask: torch.Tensor
) -> float:
    """
    Accuracy restricted to positions where induction is the only signal
    available for correct prediction. Same function for both models so the
    scores are directly comparable.
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == target_ids) & induction_mask
    denom = induction_mask.sum().clamp(min=1)
    return (correct.sum().float() / denom).item()


if __name__ == "__main__":
    cfg = InductionTaskConfig(seed=0)
    gen = InductionTaskGenerator(cfg)
    batch = gen.sample_batch()
    print("input_ids", batch["input_ids"].shape)
    print("target_ids", batch["target_ids"].shape)
    print("induction positions in first sequence:", batch["induction_mask"][0].nonzero().flatten().tolist())
