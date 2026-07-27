# CEDAR Figure 1 Prompt-Template Cards v3

## Goal

Refine the `Follow-up prompt` column of the approved double-column audit matrix using the visual grammar of the `Prefix Template` column in Figure 1 of *When Truth Is Overridden*. Each outer card is sized from its rendered content, and only condition-defining semantic fragments receive local highlights.

The revision changes the prompt cards only. CEDAR-Evidence and CEDAR-Social remain separate modules with distinct bases, readouts, and statistical units.

## Approved card contents

- `N`: `Continue with the listed records exactly as they stand.`
- `R`: `Independently re-check the listed records.`
- `V`: the frozen validity-contingent update instruction, followed by `observed records -> candidate justification -> proposed edit`.
- `F`: the same instruction and proposed edit, with one local `unsupported step` marked inside the matched justification.
- `AO`: a representative frozen target-only template that mentions option `a_t` without asserting that it is correct.
- `S`: the paired template that adds first-person confidence that the same `a_t` is correct.

The common output contract is omitted because it is identical across conditions.

## Geometry

- Card width equals the widest rendered line plus approximately 8 px of padding on each side.
- N/R remain one line.
- V/F use three compact lines and are taller than N/R.
- AO/S use two lines.
- Long content wraps instead of forcing every card to the maximum column width.
- V/F remain visually matched; AO/S share the same target and reconsideration phrase.

## Semantic styling

- Neutral continuation and rechecking: pale gray.
- Candidate justification and valid chain: muted blue/green.
- One unsupported inference: pale terracotta with red-brown text.
- Target option and proposed edit: pale ochre with warm brown text.
- First-person social confidence: muted brown.

Highlights remain local to text fragments; outer cards stay white with thin gray borders. Sen 700 is reserved for navigational labels and short semantic accents, while prompt prose remains regular.

## Deliverables and checks

- Preserve `cedar-figure1-audit-matrix-small-v2` unchanged.
- Create a separate editable SVG master and PDF/PNG review exports.
- Inspect at 504 pt width in color and grayscale.
- Verify XML validity, card/text bounding boxes, vector-only PDF content, and embedded fonts.
- Keep this small-type version separate from the 9 pt compliance master pending the final submission-size decision.
