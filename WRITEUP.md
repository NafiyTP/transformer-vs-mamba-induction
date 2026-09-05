# Do transformers and Mamba solve induction the same way?

## Motivation

I wanted a portfolio project that was closer to actual research than to a build-something-that-works engineering project, since that's what a research internship application is supposed to demonstrate. I already had two more "product" style projects (a log anomaly detection pipeline, a multi-agent system for medical scheduling), so I was looking for something with an actual question behind it instead of a spec to implement.

Mechanistic interpretability seemed like a good fit: it's the part of ML research closest to "form a hypothesis, run a controlled experiment, look at the result critically," and it's also directly useful for me since I'm already reading about transformer internals for LLM-focused interviews. The generic version of this (reproduce induction heads on GPT-2, following the ARENA exercises) is a well-worn tutorial at this point though, so I wanted a version with an actual open question in it.

Mamba gave me that. It replaces attention with a linear state-space recurrence, and unlike most people doing interpretability, I'd already seen state-space models, transfer functions and pole placement from a signal processing course (TSIA202b), so I had a way in that most people in this space don't have. The question I settled on: does Mamba implement the same induction mechanism as a transformer, and can I explain what I find using the same math I'd use to analyze a Kalman filter or an ARMA model?

## The task

Induction, in the sense the interpretability literature uses it, is the ability to complete a pattern that already appeared earlier in the same context: see "... A B ... A", predict "B". Olsson et al. (2022) showed transformers develop a specific, well-localized circuit for this: a "previous token" head in an early layer feeds a bigram representation to an "induction head" in a later layer, which does the actual copying.

I used their synthetic task: random token sequences with a repeated chunk inserted somewhere in the middle. The model is scored only on the positions where the repeat lets it predict correctly; everywhere else is unpredictable noise on purpose, so the metric can't be gamed by anything except actually doing induction.

## Setup

Both models are tiny, two layers, trained from scratch on the same task with the same evaluation code, so the comparison is apples to apples:

- Transformer: attention-only (no MLPs), via TransformerLens, so patching and attention pattern inspection come for free.
- Mamba: I wrote the selective SSM block myself in plain PyTorch rather than using the official `mamba-ssm` package, mainly to avoid the CUDA kernel build, but also because a sequential Python loop over timesteps keeps every hidden state directly inspectable, which matters for the patching experiments later.

## Getting Mamba to actually train

This ended up being most of the work, and I'm including it here instead of hiding it because the debugging was arguably the most "research" part of the whole project.

First attempt: loss stuck exactly at chance level, no movement at all, for thousands of steps. My first guess was a slow phase transition (the transformer literature documents these taking a while too), so I just let it run longer. No change.

I checked the model could learn anything at all first: trained it to memorize a single fixed batch of random targets, no induction structure involved. It hit 98% in about 30 steps, so the architecture itself wasn't broken.

Next I went looking for why long-range information wasn't reaching the output. I perturbed one token early in a sequence and tracked how long the effect survived. It disappeared after 5-6 positions, while the task needs memory over 20-40 positions. That pointed at delta, the per-channel timestep parameter that controls how fast each SSM channel forgets. It turned out `dt_proj`'s weight was randomly initialized, which meant delta was effectively noise at every step instead of being controlled by the per-channel bias I'd set up to give a spread of memory horizons. Zero-initializing that weight (matching what the actual Mamba implementation does) fixed the memory decay, confirmed by rerunning the perturbation test.

Still stuck at chance after that fix, though. The next thing I tried was the most useful single test in the whole debugging process: train on a single fixed induction batch (not a fresh one every step) instead of the real setup. Perfect accuracy in about 100 steps. That told me the architecture could represent the solution just fine, so the actual problem was generalizing across varying inputs, not representational capacity.

That pointed at optimization, and the fix turned out to be almost embarrassingly simple: AdamW's default weight decay (0.01) was preventing the circuit from ever forming. Setting `weight_decay=0.0` and testing on a scaled-down version of the task (smaller vocabulary, shorter sequences) gave a clean phase transition, chance to 85% accuracy, in under a thousand steps. I still don't have a fully satisfying explanation for exactly why weight decay kills this particular circuit (my best guess is that it keeps pulling the slow-moving selective parameters, A, dt_proj, the B/C projections, back toward zero faster than the multi-parameter coordination needed for induction can build up), but the effect is very reproducible.

Removing weight decay opened up a different problem at full scale: the loss went to `nan` around step 550. With nothing regularizing it, the A parameter (parametrized as `-exp(A_log)`) could drift large enough to overflow, and if delta rounded to exactly 0.0 for some channel at the same time, the product became `0 * -inf = nan`. Clamping `A_log` and putting a small floor under delta fixed it, plus a general safety check that skips a step outright if the loss or gradient ever comes back non-finite.

With all of that in place, the full-scale run (vocabulary of 50, sequences of 64) went through a clean phase transition between steps 1000 and 3000, and plateaued at 86.9% held-out accuracy, matching the transformer's 85.4%.

## Where the circuit actually lives

Once both models worked, I ran activation patching: corrupt the source of the repeated chunk so the model can't do induction, then splice in activations from a clean run at specific parts of the network and see how much accuracy comes back.

For the transformer this reproduced the textbook result almost exactly. Patching either of the two heads in layer 2 individually recovers 68-80% of the accuracy gap; layer 1's heads do essentially nothing (under 1%). Attention pattern inspection agrees: both layer-2 heads put 50-56% of their attention mass on the position an induction head should attend to, versus 2-3% for layer 1.

Mamba is a different story. Patching any single group of 16 channels (out of 128) in either layer recovers at most 3.5%, nowhere near the transformer's per-head numbers. But patching all 128 channels of layer 2 at once recovers 52%, and patching both layers fully recovers essentially 100%. Sweeping what fraction of layer 2 gets patched shows a smooth, roughly proportional curve (25% of channels -> 18% recovery, 50% -> 36%, 75% -> 75%, 100% -> 100%), with no small subset that punches above its weight. Individually, no channel recovers more than about 2%.

So both models solve the same task about equally well, but the transformer does it with a circuit you can point at (two heads), while Mamba does it with something closer to a population code spread across an entire layer, where no individual unit matters much but the aggregate does.

## The pole hypothesis, and why it didn't hold up

The part of this I was most curious about going in was whether I could explain Mamba's circuit using the state-space math directly, the way you'd analyze a resonant filter's pole locations. My hypothesis: channels that matter for induction should have a slower pole (closer to 1, meaning longer memory), since they need to hold information across the whole gap between the two occurrences of the repeated chunk.

I computed each layer-2 channel's slowest discrete pole (using the actual delta values observed at induction-relevant positions on real batches) and compared it against that channel's individual patching importance score. The correlation is +0.046, essentially zero. The mean pole of the least important 20% of channels (0.991) and the most important 20% (0.992) are almost identical.

The hypothesis was wrong, or at least too simple. My read on why: by the time training has converged, nearly every channel already has more than enough memory (poles very close to 1) to span the task's gap, so memory horizon stops being the bottleneck and stops differentiating channels. What likely distinguishes an important channel from an unimportant one is the content of its B and C projections, whether it happens to write and later read a direction that actually encodes token identity, not how long it can hold onto it. I didn't have time to test that directly, but it's the obvious next step.

## Takeaways

- Comparable end performance can hide a completely different internal mechanism. That seems like a useful reminder generally, not just for this pair of architectures.
- The debugging process (three separate real bugs: initialization, weight decay, numerical overflow) mattered more than I expected going in. None of them were exotic; all three would have been easy to miss without deliberately isolating variables (fixed batch vs streaming batch, scaled-down task vs full task) one at a time.
- The one hypothesis I had real confidence in going in (pole speed predicts importance) is the one that didn't survive contact with the data, which I think is a fine outcome for a small independent project: a clean negative result beats a hand-wavy positive one.

## What I'd do with more time

- Look at the B/C projections directly for important vs unimportant channels, since that's the more likely explanation the pole analysis pointed me toward.
- Run the same comparison with the official `mamba-ssm` CUDA kernel to check the training dynamics (especially the weight decay effect) aren't an artifact of my simplified sequential implementation.
- Scale up to more layers and a larger vocabulary to see whether the "distributed" character of the Mamba circuit is a small-model effect or holds at scale.
