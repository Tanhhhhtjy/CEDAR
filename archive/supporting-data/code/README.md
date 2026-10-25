# Code layout

`src/cedar_m2` contains the intervention and evaluation implementation.
`scripts/` contains the frozen runners, analyzers, and audits.  The companion
Python modules expected by those scripts are mirrored under
`datasets/m2_2b_formal/`.  `analysis/analyze_scale_contraction.py` is a
package-relative CPU-only reanalysis of the recorded option logits.

The original GPU runners require the public model and intervention assets named
in the run manifests; those weights and vectors are not redistributed.  No GPU
is needed to inspect records, rerun the contraction analysis, or verify hashes.
