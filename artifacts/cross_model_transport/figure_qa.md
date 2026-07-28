# Figure QA

- MATLAB scripts read only `e2a_family_geometry.csv`, `e2a_family_arm_geometry.csv`, `cross_model_family_effects.csv`, and `figure2d_forest_input.csv`; estimates and coordinates are not hard-coded.
- All three figures were visually inspected at original PNG resolution.
- SVG files contain vector path elements and no embedded raster `<image>` elements.
- PDF files contain no detected `/Subtype /Image` objects; plots remain vector.
- PNG dimensions and metadata:
  - `e2a_family_geometry.png`: 3898×3038 at 600 dpi.
  - `figure2d_cross_model_transport.png`: 3764×2038 at 600 dpi.
  - `cross_model_paired_family_geometry.png`: 3915×2109 at 600 dpi.
- E2A upper panel uses 12 observed family means per condition, exact x positions, gray hollow uncapped markers, teal filled p25 markers, and family connectors.
- E2A lower panel uses family-level `delta_m_target` in logits; target-follow percentage points are not placed on this axis.
- Cross-model forest uses model-specific shapes and within-model effects only; it does not plot cross-model raw-logit levels.
- Paired family audit uses N gray, R blue-gray, V teal, F orange-red; Qwen circles, Llama diamonds; uncapped hollow and p25 filled.
- Color is redundant with marker shape, fill, panel/model labels, and connector structure.

The paired family audit panels have separate within-model axes and are explicitly labelled; they are diagnostic, not a direct comparison of the two models' raw-logit levels.
