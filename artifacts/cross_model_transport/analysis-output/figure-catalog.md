# Figure catalog

## `figures/e2a_family_geometry.*`

- Purpose: replace Figure 0 placeholder coordinates with 12 observed paired family means.
- Data: `e2a_family_geometry.csv` and V/F rows from `e2a_family_arm_geometry.csv`.
- Notice: family-specific movement and its heterogeneity; lower-panel units are logits.
- Decision implication: Figure 0 can use real paired family geometry without rerunning E2A.
- Caveat: target-follow percentage points are not plotted on the margin axis.

## `figures/figure2d_cross_model_transport.*`

- Purpose: show within-model overall and V-F effects with family-bootstrap CIs.
- Data: `figure2d_forest_input.csv`.
- Notice: both overall CIs are positive; both V-F intervals cross zero.
- Decision implication: numerical routing is G-A, bounded to the archived matched regime.
- Caveat: horizontal positions are within-model effects, not cross-model raw-logit levels.
