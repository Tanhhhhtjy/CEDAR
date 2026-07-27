# CEDAR Figure 1 Design

## Decision

Use a double-column, two-band audit matrix inspired by Figure 1 of *When Truth Is Overridden*. The layout was selected after comparing three low-fidelity alternatives. The figure is a motivation-and-design overview, not a results dashboard or a complete experimental pipeline.

## Scientific message

The figure should make one contrast immediately visible: the same proposed target can be accompanied by reasons with different normative status, so answer movement alone is not a sufficient audit criterion.

The two CEDAR modules remain visibly and statistically separate:

- **CEDAR-Evidence** uses a solver-certified formal base and pairs `N/R/V/F` within that base. Only `V` warrants revision from the initial answer to the proposed target.
- **CEDAR-Social** uses an independent natural-task base and pairs `AO/S` within that base. Neither condition supplies task-relevant evidence; `S-AO` isolates social phrasing beyond target exposure.

The artwork must not imply that `N/R/V/F/AO/S` form a unified six-arm dataset, share samples, or support a joint significance test.

## Layout

The final artwork spans `\textwidth` inside an AAAI `figure*` float.

- Left approximately 76%: a four-column matrix.
  - `Condition`
  - `Only change in the follow-up`
  - `Reason status`
  - `Normative response`
- Right approximately 23%: an `Audit lens` callout.
- A small gutter separates the matrix from the callout.
- The upper blue band is CEDAR-Evidence and contains four rows: `N`, `R`, `V`, and `F`.
- The lower lavender band is CEDAR-Social and contains two rows: `AO` and `S`.
- Each module has its own base strip: formal base `E` versus independent natural-task base `S`, while both use the notation initial `a_0` and proposed target `a_t`.

The audit lens contrasts a behavior-only question with the two CEDAR questions:

- Behavior-only: did `a_t` become more likely?
- Evidence: did movement toward `a_t` track whether revision was warranted?
- Social: what changed beyond exposure to `a_t` itself?

Models, model-specific axes, capping configurations, continuous readouts, and four-state transitions stay outside Figure 1. They belong in the methods prose and Figures 2–3.

## Typography and geometry

- Primary family: **Sen**, using real regular and bold faces rather than synthetic styling.
- Final-size body text: at least `9.2 pt`.
- Column headings and module names: approximately `10.5–11 pt`.
- No text may fall below `9 pt` after insertion at `\textwidth`.
- Target artwork size: approximately `7.0 in × 2.1–2.3 in`, excluding the LaTeX caption.
- Main strokes: approximately `0.75–0.9 pt` at final size.
- Rounded corners are restrained; no shadows, gradients, decorative icons, or UI-style badges.
- Text and objects remain editable in the master SVG.

The published reference used Sen but shrank its internal text to roughly `4.8–6.8 pt` after LaTeX scaling. CEDAR borrows its hierarchy and grid, not those final font sizes.

## Palette

- Header: `#BAC8D3`
- Evidence background/strip: `#E7F7FB` / `#CFEAF3`
- Social background/strip: `#F4EFF9` / `#E5D9F0`
- Main text/strokes: `#263544` / `#687684`
- Changed-clause highlight: `#FFF4C3`
- Social-clause highlight: `#CCCCFF`
- Valid: `#2A9D8F`
- Invalid: `#E76F51`
- Social accent: `#8064A2`

Color is redundant with labels, check/cross symbols, grouping, and borders so that the figure remains interpretable in grayscale.

## Wording discipline

- Use safe schematic clauses rather than literal harmful prompts.
- Use `a_0` and `a_t`, matching the manuscript.
- Keep row text short enough for a single line at the target font size.
- The figure contains no empirical values and does not imply any observed direction.
- The LaTeX caption remains self-contained and points to additional examples and certificates using the semantic `[TBD-APPENDIX]` placeholder until appendix order is fixed.

## Deliverables

- `source.svg`: primary Inkscape-editable master with named groups.
- `preview.pdf`: vector export for LaTeX insertion and font inspection.
- `preview.png`: review image at high resolution.
- `README.md`: dimensions, font assumptions, export command, and validation notes.

The existing `cedar-prompt-overview-v0` draft remains untouched for comparison.

## Acceptance checks

1. At `\textwidth`, every text element is at least `9 pt`.
2. CEDAR-Evidence and CEDAR-Social have distinct base strips and no shared sample container.
3. Only `V` is visually marked as warranting an update.
4. `AO/S` explicitly isolate social wording from target exposure.
5. The figure is readable in grayscale and without relying on check/cross color.
6. SVG objects remain editable and grouped by module, row, and audit lens.
7. The PDF has a tight bounding box and no rasterized text or shapes.
8. Any PDF fonts are embedded; the final-sized PDF is visually inspected for clipping and overlap.
