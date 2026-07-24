# Figure Resource Catalog Design

Date: 2026-07-24

## Purpose

Create a durable, searchable index of the figure-production resources evaluated
for CEDAR. The catalog must make the preferred MATLAB + Inkscape workflow easy
to recover later without implying that any third-party tool has been installed
or that a reusable asset is automatically suitable for publication.

## Scope

The index covers:

- MATLAB plotting, statistical visualization, and vector export;
- Inkscape extensions and figure-management utilities;
- reusable Figure 1 visuals and layout references;
- neural-network and structured-diagram generators;
- general-purpose editable SVG icon libraries;
- Draw.io and TikZ alternatives;
- known licensing, maintenance, formatting, and workflow risks.

It does not install tools, alter experimental evidence, or designate a concept
illustration as a validated paper result.

## File structure

```text
figures/
  README.md
  resources/
    RESOURCE_CATALOG.md
    MATLAB_INKSCAPE_WORKFLOW.md
    ASSET_PROVENANCE.md
  editable-topconf-demo/
```

`figures/README.md` is the short navigation entry point. The catalog is the
complete human-readable ledger. The workflow records tool responsibilities,
source-of-truth rules, directory conventions, export commands, and publication
checks. The provenance file is a template for recording every external asset
actually incorporated into a paper figure.

## Catalog schema

Each resource records, where known:

- resource name and direct URL;
- resource type and intended use;
- recommendation tier (`A`, `B`, `C`, or `Avoid`);
- license and maintenance snapshot date;
- editable input/output formats;
- local installation status;
- important risks and conditions of use.

Resources are grouped by task rather than by popularity. `A` means preferred
for the current CEDAR workflow, `B` means useful for a specific figure class,
`C` means reference-only, and `Avoid` means that a known workflow, licensing,
or publication-format conflict should be resolved before use.

## Workflow policy

- MATLAB is the only source of truth for quantitative geometry and data marks.
- Inkscape SVG is the master source for the final composite Figure 1.
- Draw.io is optional and limited to early storyboarding or relationship drafts.
- Data curves, bar heights, and sample positions are never manually changed in
  Inkscape; changed data requires regenerating and replacing the MATLAB panel.
- Third-party assets are copied only after license review and provenance entry.
- Final PDF export is checked at AAAI double-column size for minimum text size,
  font embedding, Type 3 fonts, overlap, clipping, whitespace, and grayscale
  readability.

## Acceptance criteria

- A new contributor can find the preferred tool for a figure type from the
  root README in under a minute.
- Every listed resource has a direct link, role, recommendation tier, license
  status, and explicit risk note.
- The workflow distinguishes editable sources, generated artifacts, and final
  publication exports.
- The provenance template is ready before any external asset enters a formal
  CEDAR figure.
- No third-party skill or tool is installed as part of this documentation task.
