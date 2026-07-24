# Editable Top-Conference Figure Demo Design

Date: 2026-07-24

## Purpose

Create three locally viewable, fully editable research-figure demonstrations. They test visual quality and editability only; none represents CEDAR evidence or a paper claim.

## Shared constraints

- Deliver one `.drawio` source and one editable `.svg` source per figure.
- Export matching PDF and PNG previews.
- Target AAAI double-column width (6.975 in) with at least 9 pt text at final size.
- Use Arial/Helvetica consistently, a colorblind-safe palette, restrained borders, and no decorative gradients or shadows.
- Keep every label, box, arrow, chart mark, and legend item independently editable in Draw.io.
- Mark every figure `Illustrative example — synthetic content/data`.
- Validate overlap, clipping, whitespace, grayscale readability, SVG structure, Draw.io XML, final-size preview, and PDF font embedding where the renderer permits inspection.

## Figure A: Conceptual pipeline

Theme: **Evidence-Aware Self-Correction**.

Three-panel narrative:

1. A noisy evidence stream containing a reliable update, a distractor, and social pressure.
2. A selective verification module that scores source reliability and logical support before routing information.
3. A calibrated decision update that changes only when evidence passes both checks.

The figure demonstrates hierarchy, branching, semantic color, equation callouts, and a compact confidence trajectory.

## Figure B: Technical architecture

Theme: **Sparse Memory Router**.

The architecture shows token encoding, a router, three specialist paths, gated memory read/write, residual fusion, and a prediction head. A small training-objective strip separates task, routing, and consistency losses. The diagram demonstrates repeated modules, skip connections, tensor labels, grouped regions, and bidirectional data flow.

## Figure C: Results dashboard

Theme: **Robustness Across Shift Severity**.

The dashboard contains a main performance-versus-shift plot, a compact calibration plot, an ablation heatmap, and a resource/performance trade-off inset. Values are explicitly synthetic. The figure demonstrates data-density control, legends, direct annotations, uncertainty bands, multiple panel types, and grayscale-safe line encodings.

## File layout

```text
figures/
  README.md
  editable-topconf-demo/
    conceptual-pipeline/
    technical-architecture/
    results-dashboard/
```

Each figure directory contains `source.drawio`, `source.svg`, `preview.pdf`, and `preview.png`. Validation artifacts may be placed under `_checks/`.

## Acceptance criteria

- All three `.drawio` files open without repair warnings.
- Each visual component can be selected and edited separately.
- Draw.io export and the editable SVG are visually equivalent at review size.
- No label overlaps or clips; connectors do not cross labels.
- Minimum final-size text is 9 pt.
- The main message remains understandable in grayscale.
- Synthetic status is visible but visually subordinate.
