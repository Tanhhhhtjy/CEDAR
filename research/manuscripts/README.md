# CEDAR AAAI-27 Overleaf Manuscript

This is the anonymous AAAI-27 manuscript workspace.

- `main.tex` is the main-paper submission source. Compile it with **pdfLaTeX**.
- `main_zh.tex` is the synchronized Chinese review companion. Compile it with **XeLaTeX**; it is not a submission source because `aaai2027.sty` does not support CJK.
- `supplementary.tex` is the separate anonymous supplementary source. Compile it with **pdfLaTeX**; do not append it to `main.pdf`.
- `supplementary_zh.tex` is its synchronized Chinese review companion. Compile it with **XeLaTeX**.
- `ReproducibilityChecklist.tex` remains a separate submission artifact.

## Current manuscript state

- The global section order follows `CEDAR_AAAI27_REVISED_FRAMEWORK_v2.md`.
- Sections 3--5 are drafted and synchronized in English and Chinese; Section 6 is maintained by a separate author and is not edited in this workstream.
- Figures 1--3 use font-safe vector PDFs. Editable SVG and reproducible plotting sources for Figures 2--3 are maintained in the main CEDAR repository.
- Unavailable values remain explicit, narrowly scoped `TBD` placeholders; Figure 2D is the only artwork panel with result slots still marked `TBD`.
- Citations are not added until their bibliographic records are verified.
- The main paper contains no table: complete method contracts, absolute diagnostic levels, extended estimates, representation geometry, and integrity checks are kept in the separate supplement.
- The supplement has three canonical result tables: Evidence banks, CEDAR-Social, and matched Qwen--Llama CEDAR-Evidence. Stable `\tbdcell{...}` keys identify unknown table cells while rendering compact placeholders.

## Fixed writing rule

Write every manuscript section in final-paper tense, as a completed study. Do not tell the reader that an experiment is planned, unfinished, waiting to run, or waiting to be frozen. Where the exact configuration, statistic, analysis, figure, or conclusion is unavailable, insert a narrow `TBD` placeholder in its final sentence slot.

Internal evidence labels may be used in source comments:

- `[KNOWN]`: supported by an audited artifact or primary source.
- `[TBD]`: exact content still has to be inserted.
- `[TARGET]`: desired internal outcome; never present it as an observed result.

## Fixed section order

1. Introduction
2. Related Work
3. CEDAR: Auditing Epistemic Selectivity
4. Auditing Assistant-Axis Capping
5. Diagnosing the Shift
6. Implications, Scope, and Future Control
7. Conclusion

## Scientific naming

- **CEDAR** is the reason-sensitive audit framework, not a dataset.
- **CEDAR-Evidence** is the new solver-certified paired evaluation dataset.
- **CEDAR-Social** is a paired protocol over SycoBench-600 that is analyzed separately from CEDAR-Evidence.
- N/R/V/F and AO/S are paired only within their respective modules; they are not a unified six-arm dataset.
- Qwen and Llama estimates remain model-specific and are never silently pooled.

## AAAI-27 invariants

- Keep `\documentclass[letterpaper]{article}` and `\usepackage[submission]{aaai2027}`.
- Do not modify `aaai2027.sty` or use layout-changing packages or spacing tricks.
- Maximum: 7 body pages and 9 total pages; pages 8--9 contain references only.
- Keep the review PDF anonymous.
- Export final figures as tightly cropped vector PDF with embedded fonts; do not use `trim` or `clip` in the paper.
- Do not include `.git` in anonymous supplementary material.

Before promoting any claim into the title, Abstract, Introduction, or Conclusion, update `CLAIM_EVIDENCE_REGISTER.md`.
