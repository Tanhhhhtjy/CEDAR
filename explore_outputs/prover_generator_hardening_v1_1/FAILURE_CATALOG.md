# Failure Catalog

## Formal stress failures

None in the Amendment 01 replacement run: 48/48 attempts are valid.

## Engineering failures caught by gates

1. Pre-freeze Mace4 constant parsing: fixed globally with an explicit constant prefix.
2. First-run F schedule lacked easy/medium coverage within class: preserved at commit `b4586f7`,
   corrected by Amendment 01, and followed by a complete 48-attempt rebuild.
3. One automatic V/F length-ratio failure: fixed in the renderer only and rerun for all 12 rows.
4. A pre-review audit found missing `the claim that`, awkward replacement syntax, fixed condition
   wording, and an underspecified human gate. Amendment 02 revised renderer/review governance only;
   332 protected symbolic and proof artifacts retained their canonical hashes.

No failed sample was dropped, no seed was replaced, and no correctness gate was relaxed.
