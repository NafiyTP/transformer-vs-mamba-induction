# Induction circuits: Transformer vs Mamba

Small mechanistic interpretability project comparing how a Transformer and a Mamba (selective state-space model) implement induction, the mechanism behind in-context copying ("...A B ... A" -> predict "B"). Both models solve the task about equally well, but the internal circuit turns out to be completely different: sparse and localized in the Transformer, diffuse across a whole layer in Mamba.

## Why this

Induction heads are the classic entry point into mechanistic interpretability (Olsson et al., 2022): a small, well-understood circuit that explains part of how transformers do in-context learning. Mamba replaces attention with a linear recurrence but matches transformers on the same tasks, including induction, so the natural question is whether it's doing the same thing internally or something else entirely. I hadn't seen this compared directly with activation patching on both architectures, and the state-space formulation (transfer functions, poles) is something I already had some background in from a signal processing course, so it felt like a good angle to actually dig into rather than just read about.

## What's in here

- A synthetic induction task (repeated random token sequences), used identically for both models so the comparison is fair.
- A tiny attention-only Transformer (via TransformerLens) as the baseline.
- A from-scratch, pure PyTorch implementation of a selective SSM block (Mamba), sequential rather than the parallel-scan CUDA kernel, so every intermediate hidden state stays inspectable.
- Activation patching on both models to find which part of the network is causally responsible for induction.
- Pole/eigenvalue analysis of the Mamba SSM's state transition, to test whether "how long a channel remembers" predicts "how important that channel is."

## Results

| | Transformer | Mamba |
|---|---|---|
| Induction accuracy (held-out) | 0.854 | 0.869 |
| Circuit | 2 attention heads, layer 2 (patching recovers 68-80% each) | Whole layer 2, no single channel matters (max +2%), full layer recovers +52%, both layers +100% |

Same task, comparable performance, but a qualitatively different circuit. The Transformer concentrates induction into two heads you can point at. Mamba spreads it across roughly its whole second layer: patching a random 25% of its channels recovers about 18% of the accuracy, 50% recovers about 36%, 100% recovers everything, a smooth curve with no small subset that does most of the work.

The pole analysis result was a small surprise: I expected channels with a slower pole (closer to 1, longer memory) to be the important ones for induction, by analogy with needing a resonant/near-unit pole for long memory in a classical filter. The data doesn't support that: correlation between pole speed and per-channel importance is +0.046, essentially nothing, and the average pole of the most vs least important channels is almost identical (0.992 vs 0.991). Most channels already have plenty of memory, so memory horizon isn't what makes a channel useful here, it's presumably the content of what it reads/writes (B and C), not its time constant.

Full write-up, including the debugging path to get Mamba trained at all, is in [`WRITEUP.md`](./WRITEUP.md).

## Repo structure

```
src/
  data/induction_task.py       synthetic induction task generator + accuracy metric
  models/tiny_transformer.py   attention-only baseline (TransformerLens)
  models/minimal_mamba.py      selective SSM block, from scratch
  analysis/patching.py         activation patching for both models
  analysis/eigen_analysis.py   pole/eigenvalue analysis of the Mamba SSM
scripts/
  01_train_baseline_transformer.py
  01b_analyze_baseline.py      attention pattern scan on the trained transformer
  02_train_mamba.py            resumable (checkpoints every 200 steps)
  03a_localize_transformer.py  patching, transformer side
  03b_localize_mamba.py        patching, mamba side
  04_eigen_analysis.py         pole analysis
results/                       saved checkpoints, plots, numbers
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 scripts/01_train_baseline_transformer.py
python3 scripts/01b_analyze_baseline.py
python3 scripts/02_train_mamba.py       # resumable, re-run to continue past its time budget
python3 scripts/03a_localize_transformer.py
python3 scripts/03b_localize_mamba.py
python3 scripts/04_eigen_analysis.py
```

Everything is small enough to train on CPU (a few minutes for the transformer, longer for Mamba since it's a sequential Python loop rather than a fused kernel, closer to 20-40 minutes total to reach the plateau depending on the machine).

## References

- Olsson et al., 2022, "In-context Learning and Induction Heads" (Anthropic) — the original circuit and the synthetic task this project reuses.
- Gu & Dao, 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
- TransformerLens (Neel Nanda) and the ARENA mechanistic interpretability curriculum, for the patching methodology on the transformer side.

## Limitations

The Mamba block is a simplified, sequential reimplementation, not the official `mamba-ssm` CUDA kernel, so absolute numbers (training steps needed, exact thresholds) shouldn't be taken as representative of the real Mamba implementation, only the qualitative comparison (localized vs distributed circuit) is the point. Both models are tiny (2 layers) and trained on a synthetic task, not real language.
