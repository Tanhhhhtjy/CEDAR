# CEDAR Figure 2 paper draft v2

This revision keeps the dense result-figure language of the AAAI-26 reference
while matching the manuscript's A/B/C/D panel contract.

- `make_result_bars.py` is the reproducible plotting source.
- `result-bars-paper-preview.png` is the current review image.
- `result-bars-paper-editable.svg` is editable in Inkscape.
- `result-bars-paper-vector.pdf` is the vector paper asset.

Panels A--C use only values recorded in the manuscript. Panel B shows V/F
updating and panel C shows the margin decomposition, matching the text. Panel D
contains result-neutral `TBD` slots and does not simulate cross-model effects.
The export is 7.05 x 3.30 inches and plotted text is at least 9 pt.

When the matched Qwen and Llama estimates are frozen, replace panel D's cards
with a within-model point--interval display. Whiskers must be described as
"where shown" until every displayed arm has an interval.
