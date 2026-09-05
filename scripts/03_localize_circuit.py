"""
Stage 2 entry point: circuit localization via activation patching, for both
models. Left as a thin driver script -- the two real TODOs it depends on
(the corruption recipe and the Mamba hidden-state injection point) are
flagged with NotImplementedError in src/analysis/patching.py, since those are
methodological decisions worth making deliberately once Stage 0/1 results are
in hand, not defaults to lock in now.

Once patching.py is filled in, this script should:
  1. Load results/baseline_transformer.pt and results/mamba_baseline.pt.
  2. Build clean/corrupted batches from the same induction_task generator.
  3. For the transformer: patch each (layer, head) attention pattern, record
     recovered induction_accuracy -> a (layer x head) heatmap.
  4. For Mamba: patch each (layer, channel) hidden-state slice, record
     recovered induction_accuracy -> a (layer x channel) heatmap.
  5. Save both heatmaps to results/ for the write-up, and save the list of
     "important" Mamba channels for Stage 3's eigenvalue analysis.
"""

raise SystemExit(
    "Stage 2 driver -- fill in src/analysis/patching.py first (see its "
    "docstrings for the two open methodological decisions), then implement "
    "the loop described in this file's module docstring."
)
