# CEDAR dataset catalog — image-generation prompt

Mode: built-in image generation, followed by one targeted structural correction.

## Design prompt

Create a polished, elegant academic infographic for a double-column AAAI paper.
Present the CEDAR dataset as a directory/catalog rather than a sequential
pipeline. Use a wide landscape layout with a circular six-petal hub around a
white central medallion. Place three equal directory cards on the left and
three mirrored cards on the right, connected by short straight horizontal
leaders. Add a slim metadata ribbon below.

Center:

- CEDAR
- Core Formal Bank
- 24 families · 48 bases · 288 arms

The six central sections and their matching cards are:

1. Formal Core — Horn world, four-option query, unique gold, atomic edit.
2. Six-Arm Assay — N Neutral, R Re-check, V Valid, F Fallacious,
   AO Exposure, S Social.
3. Matched Controls — V/F shared edit and target; AO/S differ by one social
   semantic cue; exact token match.
4. Structural Difficulty — easy/medium, proof depth, necessary rules,
   distractors, countermodels.
5. Balanced Coverage — six answer transitions, A/B/C/D positions, six error
   classes, early/middle/late error positions.
6. Solver-Certified Audit — Python ↔ Prover9, Mace4 witnesses, replayable
   certificates, deterministic hashes.

Bottom ribbon:

`analysis-ready metadata · 4-way outcomes · margins · logits · entropy · reasoning tokens`

Use one consistent family of minimal line icons: proof network, six-node
hexagon, linked rings, stacked layers, balance scales, and shield check.
Use deep navy outlines with restrained teal, blue, muted purple, amber,
orange-red, and forest-green pastel fills. Typesetting should resemble Inter or
IBM Plex Sans. Use subtle flat offset backing rather than gradients or
photorealistic shadows. Keep every card equal in size and preserve mirrored
spacing. No experimental results, plots, flow arrows, agents, robots, people,
3D, dark background, watermark, legend, or unnecessary decoration.

## Structural correction

Preserve the generated aesthetic but enforce the following one-to-one spatial
mapping:

- upper-left: Formal Core;
- middle-left: Six Arms;
- lower-left: Matched Controls;
- upper-right: Structural Difficulty;
- middle-right: Balanced Coverage;
- lower-right: Solver Audit.

Every card must connect only to its identically named central section using a
short, straight, horizontal, non-crossing line.
