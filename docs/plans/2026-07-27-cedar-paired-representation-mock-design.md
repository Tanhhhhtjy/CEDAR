# CEDAR Paired Representation Mock — Design

Date: 2026-07-27  
Status: approved visual mock; simulated data only

## Purpose

Test the visual form of a Figure-6-inspired representation plot before any
hidden-state experiment is run. The mock must not imply an observed CEDAR
result.

## Approved visual design

- Use a two-dimensional PCA-like coordinate system generated from a fixed
  random seed and explicitly simulated data.
- Show Valid (`V`) and Fallacy (`F`) conditions with different colors.
- Show uncapped states as hollow markers and capped states as filled markers.
- Connect each uncapped/capped pair with a low-opacity arrow.
- Add group centroids and covariance ellipses to make the aggregate displacement
  legible without relying on the dense point cloud alone.
- Include a compact displacement-summary panel rather than adding more prompt
  conditions to the main scatter.
- Label the entire artifact `SIMULATED DATA · DESIGN MOCKUP` in the figure and
  its caption/readme.

## Intended hypothetical story

The simulated layout illustrates one possible future outcome: validity-related
states remain separable while capping applies a broadly shared displacement.
This is a visual hypothesis only. Real results may instead show collapsed,
absent, or condition-specific separation, and the final figure must follow the
observed data.

## Production workflow

1. The formal-data workflow remains MATLAB-owned. For this simulated visual
   mock, Python/Matplotlib may generate the same fixed-seed geometry because the
   local MATLAB R2026a process currently crashes during startup.
2. Export SVG and PDF as vector artifacts plus PNG for quick inspection.
3. Inkscape may adjust typography and annotation placement, but must not move
   data marks or alter statistics.
4. Validate embedded fonts, absence of Type 3 fonts, grayscale legibility,
   overlap, and final AAAI double-column size.

## Deliverables

- MATLAB source script and temporary Python mock renderer
- simulated data table with a fixed seed
- editable SVG
- vector PDF
- PNG preview
- README distinguishing the mock from experimental evidence
