# C1/V7 Phase 2 construction review

Status: `author_construction_review_passed_awaiting_independent_human_review`

This is an author-side, zero-model construction audit. It is not independent human review. Every item was recomputed by `verify_phase2_full48_solutions.py`; finite matching, tournament, hitting-set, permutation, and closure tasks were exhaustively enumerated. No Qwen, GPU, capping, or behavioral item selection was used.

## Aggregate gate

- Bases reviewed: 48/48
- Original unique: 48/48
- V-updated unique: 48/48
- Single changed node: 48/48
- Distractor wrong before and after V: 48/48
- External knowledge required: 0/48
- Family alignment failures: 0
- Additional reasonable answers found: 0
- Author decision: 48 pass, 0 unclear, 0 fail

## Item-by-item review

### C1V7-001 — C1V7-F01 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: Sorting the five active scores gives [12, 18, 25, 31, 44], whose third value is 25 points, so A is uniquely supported.
- V recomputation: Replacing Delta's score with 16 and re-sorting gives [12, 16, 18, 25, 44], whose third value is 18 points, so B is uniquely supported.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: Delta's one active score record, corrected from 31 points to 16 points.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-002 — C1V7-F01 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: Sorting the seven scores leaves 34 at the unique fourth rank.
- V recomputation: After replacing 53 with 15, the sorted roster is 12, 15, 19, 27, 34, 46, 61, so 27 is uniquely fourth.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The one active score record corrected from 53 to 15.
- Necessary steps: original 3; updated 3.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-003 — C1V7-F02 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The effective capacities are 72, 76, and 90, whose unique minimum is 72.
- V recomputation: With stage B's factor corrected to 0.70, the capacities are 72, 56, and 90, whose unique minimum is 56.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: Stage B's one availability-factor record, corrected from 0.95 to 0.70.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-004 — C1V7-F02 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: The four effective capacities are 96, 90, 99, and 105, so B uniquely sets throughput at 90.
- V recomputation: Correcting C's factor to 0.60 makes its capacity 66 while the others remain 96, 90, and 105; 66 is the unique minimum.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: Stage C's one availability-factor record, corrected from 0.90 to 0.60.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-005 — C1V7-F03 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: Ceiling of 52 divided by 12 is 5 containers, and 5 times 8 is 40 credits.
- V recomputation: At corrected capacity 18, ceiling of 52 divided by 18 is 3 containers, costing 24 credits.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The certified usable-capacity record, corrected from 12 to 18 parts per container.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-006 — C1V7-F03 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: Five cases hold only 120, while six hold 144, so exactly 6 cases are required and cost 66 tokens.
- V recomputation: At corrected capacity 35, three cases hold 105 but four hold 140, so exactly 4 cases are required and cost 44 tokens.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The one certified usable-capacity record, corrected from 24 to 35 samples per case.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-007 — C1V7-F04 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The serial product is 800 times 0.75 times 0.80 = 480 units.
- V recomputation: Replacing only stage 2's yield with 0.50 gives 800 times 0.75 times 0.50 = 300 units.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The stage-2 yield record, corrected from 0.80 to 0.50.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-008 — C1V7-F04 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: The stages leave 1080, then 864, then 648 units, so final output is 648.
- V recomputation: With only the middle yield corrected to 0.60, the stages leave 1080, then 648, then 486 units.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The stage-2 yield record, corrected from 0.80 to 0.60.
- Necessary steps: original 3; updated 3.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-009 — C1V7-F05 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The signed sum is 120 + 70 - 30 = 160.
- V recomputation: Reclassifying only the 70 entry as contra gives 120 - 70 - 30 = 20.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The complete formal-class record for the 70 entry: ordinary to contra.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-010 — C1V7-F05 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: Applying the classes gives 210 + 90 - 40 - 20 = 240.
- V recomputation: After only the 90 entry becomes contra, the balance is 210 - 90 - 40 - 20 = 60.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The complete formal-class record for the 90 entry: ordinary to contra.
- Necessary steps: original 3; updated 3.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-011 — C1V7-F06 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The base is 132 divided by 1.10 = 120.
- V recomputation: At the corrected 20% rate, the base is 132 divided by 1.20 = 110.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The one applicable-rate record, corrected from 10% to 20%.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-012 — C1V7-F06 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: The multiplier is 1.15, and 207 divided by 1.15 gives the unique base 180.
- V recomputation: The corrected multiplier is 1.38, and 207 divided by 1.38 gives the unique base 150.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The one applicable-rate record, corrected from 15% to 38%.
- Necessary steps: original 3; updated 3.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-013 — C1V7-F07 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The 90 km legs take 2 and 1 hours, so 180 divided by 3 is 60 km/h.
- V recomputation: At corrected distance ratio 1:2, the legs are 60 and 120 km and each takes 4/3 hours, so 180 divided by 8/3 is 67.5 km/h.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The certified two-leg distance-ratio record, corrected from 1:1 to 1:2.
- Necessary steps: original 2; updated 3.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-014 — C1V7-F07 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: Equal legs are 108 km; their times are 3 and 1.5 hours, giving 216/4.5 = 48 km/h.
- V recomputation: At corrected ratio 1:2, distances are 72 and 144 km; times are 2 and 2 hours, giving 216/4 = 54 km/h.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The certified two-leg distance-ratio record, corrected from 1:1 to 1:2.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-015 — C1V7-F08 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The first pass accepts 120 and the one rework pass accepts 40, totaling 160.
- V recomputation: A second permitted rework pass accepts half of the remaining 40, adding 20 for a total of 180.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The one permitted rework-pass-count record, corrected from 1 to 2.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-016 — C1V7-F08 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: The passes accept 80, then 20, then 15 units, for a deterministic total of 115.
- V recomputation: A third rework pass acts on the remaining 45 and accepts 11.25 more, raising the total to 126.25.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The one permitted rework-pass-count record, corrected from 2 to 3.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-017 — C1V7-F09 variant 1

- Difficulty and verification: `easy`; closed-rule formal derivation.
- Original recomputation: K, KL, and KLM all match KLMN, and the unique longest prefix KLM selects Coral.
- V recomputation: After only the KLM entry is removed, KL is the unique longest remaining match and selects Blue.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The complete most-specific routing-table entry KLM -> Coral, changed from active to removed.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-018 — C1V7-F09 variant 2

- Difficulty and verification: `medium`; closed-rule formal derivation.
- Original recomputation: All four prefixes match TANGO7, and the unique longest, TANG, selects Ochre.
- V recomputation: Replacing the TANG entry with nonmatching TARP leaves TAN as the unique longest active match, which selects Lilac.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The complete most-specific routing-table record, replaced from TANG -> Ochre to TARP -> Ochre.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-019 — C1V7-F10 variant 1

- Difficulty and verification: `easy`; closed-rule formal derivation.
- Original recomputation: The numeric cores tie, and beta is lexically later than alpha, so package Q is uniquely higher.
- V recomputation: With P's identifier corrected to rc, rc is lexically later than beta, so package P is uniquely higher.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: Package P's complete prerelease-identifier record, corrected from alpha to rc.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-020 — C1V7-F10 variant 2

- Difficulty and verification: `medium`; closed-rule formal derivation.
- Original recomputation: The cores and first identifier rc tie, then numeric 11 exceeds 9, so package Q is uniquely higher.
- V recomputation: Correcting P's prerelease record to rc.15 makes 15 exceed 11 at the first unequal identifier, so package P is higher.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: Package P's complete prerelease-identifier record, corrected from rc.9 to rc.15.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-021 — C1V7-F11 variant 1

- Difficulty and verification: `easy`; closed-rule formal derivation.
- Original recomputation: The reported interval is [50 - 1, 50 + 1] = [49, 51]. Both endpoints lie inside the closed band [48, 52], so B is uniquely supported.
- V recomputation: With half-width 3, the interval is [50 - 3, 50 + 3] = [47, 53]. It shares points with [48, 52] but is not fully contained, so A is uniquely supported.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The sensor's one uncertainty half-width record, corrected from 1 unit to 3 units.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-022 — C1V7-F11 variant 2

- Difficulty and verification: `medium`; closed-rule formal derivation.
- Original recomputation: The acceptance band is [70,80] and the sensor interval [71,77] has both endpoints inside it, so it is fully contained.
- V recomputation: At corrected half-width 7, the sensor interval is [67,81]; it intersects [70,80] but is not fully contained, so it partially overlaps.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The sensor's one uncertainty half-width record, corrected from 3 units to 7 units.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-023 — C1V7-F12 variant 1

- Difficulty and verification: `easy`; closed-rule formal derivation.
- Original recomputation: The escaped first bar stays inside field 1, so the fields are oak|pine, birch, cedar and field 2 is birch.
- V recomputation: Removing only that escape makes every bar a delimiter, producing oak, pine, birch, cedar; field 2 is pine.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The single escape marker immediately before the first delimiter, changed from present to absent.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-024 — C1V7-F12 variant 2

- Difficulty and verification: `medium`; closed-rule formal derivation.
- Original recomputation: Parsing yields fields north|west, central|hub, south, east, so field 3 is south.
- V recomputation: Escaping the delimiter between hub and south merges those parts into field 2; the fields become north|west, central|hub|south, east, so field 3 is east.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The single escape marker before the delimiter between hub and south, changed from absent to present.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-025 — C1V7-F13 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The power selector uses multiplier 10, giving 10 times 3 = 30 dB.
- V recomputation: Changing only the quantity type to amplitude selects multiplier 20, giving 20 times 3 = 60 dB.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The one quantity-type metadata record, corrected from power to amplitude.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-026 — C1V7-F13 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: The power selector gives 10*1.2041=12.041, which rounds to 12.0 dB.
- V recomputation: The corrected amplitude selector gives 20*1.2041=24.082, which rounds to 24.1 dB.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The one quantity-type metadata record, corrected from power to amplitude.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-027 — C1V7-F14 variant 1

- Difficulty and verification: `easy`; closed-rule formal derivation.
- Original recomputation: The header tuple is exactly (UDP,88), whose unique table label is Maple.
- V recomputation: Correcting only the protocol makes the tuple (TCP,88), whose unique table label is Quartz.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The packet header's one protocol component, corrected from UDP to TCP.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-028 — C1V7-F14 variant 2

- Difficulty and verification: `medium`; closed-rule formal derivation.
- Original recomputation: The port is 53, so tuple (UDP,53) uniquely selects Lumen.
- V recomputation: With only protocol corrected to TCP, the unchanged port 53 forms (TCP,53), which uniquely selects Nacre.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The packet header's one protocol component, corrected from UDP to TCP.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-029 — C1V7-F15 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: NOT 0100 leaves the requested bits 1001 effective.
- V recomputation: Correcting only the mask to 0101 blocks the last requested bit too, giving 1000.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The active four-bit mask record, corrected from 0100 to 0101.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-030 — C1V7-F15 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: Masking 11111 by 00101 gives 11010, which the exact table maps to Falcon.
- V recomputation: Correcting the mask to 01101 gives effective pattern 10010, which the table maps to Harbor.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The active five-bit mask record, corrected from 00101 to 01101 by one bit.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-031 — C1V7-F16 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: Tuesdays are days 5, 12, 19, and 26, so the fourth is day 26.
- V recomputation: If day 1 is corrected to Sunday, Tuesdays are 3, 10, 17, and 24, so the fourth is day 24.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The month's one day-1 weekday anchor, corrected from Friday to Sunday.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-032 — C1V7-F16 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: Mondays are days 6, 13, 20, and 27; the next would be 34, so no fifth Monday occurs.
- V recomputation: If day 1 is Monday, Mondays are 1, 8, 15, 22, and 29, so the fifth is day 29.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The month's one day-1 weekday anchor, corrected from Wednesday to Monday.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-033 — C1V7-F17 variant 1

- Difficulty and verification: `easy`; exhaustive finite enumeration.
- Original recomputation: Nia must take Quartz. Omar must then take Ruby, leaving Sable for Pia; therefore the unique complete assignment gives Omar Ruby and uniquely supports B.
- V recomputation: After Nia's complete record is replaced, Nia must take Sable. Pia must then take Ruby, leaving Quartz for Omar; therefore the different unique complete assignment uniquely supports A.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: Nia's complete allowed-item record, atomically replaced from {Quartz} to {Sable}; no other allowance row changes.
- Necessary steps: original 3; updated 3.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-034 — C1V7-F17 variant 2

- Difficulty and verification: `medium`; exhaustive finite enumeration.
- Original recomputation: A is forced to 1, then B to 2, C to 3, and D to 4; exhaustive checking leaves this single perfect matching.
- V recomputation: Replacing A's complete record with {4} forces A=4, D=3, C=2, and B=1; this is the single updated perfect matching.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: Agent A's complete allowed-item record, atomically replaced from {1} to {4}.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-035 — C1V7-F18 variant 1

- Difficulty and verification: `easy`; closed-rule formal derivation.
- Original recomputation: A=[3,10] and B=[6,8], so both B endpoints lie in A and A contains B.
- V recomputation: With A's duration corrected to 2, A=[3,5], whose end is less than B's start 6, so A is before B.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: Interval A's one duration record, corrected from 7 to 2.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-036 — C1V7-F18 variant 2

- Difficulty and verification: `medium`; closed-rule formal derivation.
- Original recomputation: X=[5,15] and Y=[12,18]; they intersect, X does not contain Y, and X is not before Y, so X overlaps Y.
- V recomputation: Correcting X's duration to 6 gives X=[5,11], whose end is less than Y's start 12, so X is before Y.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: Interval X's one duration record, corrected from 10 to 6.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-037 — C1V7-F19 variant 1

- Difficulty and verification: `easy`; closed-rule formal derivation.
- Original recomputation: Yes weight is 6, but veto member P votes no, so the conjunction fails and the proposal fails.
- V recomputation: Correcting only P's vote to yes gives weight 8 and satisfies the veto condition, so the proposal passes.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: Veto member P's one certified vote record, corrected from no to yes.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-038 — C1V7-F19 variant 2

- Difficulty and verification: `medium`; closed-rule formal derivation.
- Original recomputation: Yes weight is 4+4+3=11, at least 10, and the sole veto member votes yes, so the motion passes.
- V recomputation: Correcting only B's vote to no reduces yes weight to 7 while veto remains satisfied, so the threshold fails and the motion fails.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: Non-veto member B's one certified vote record, corrected from yes to no.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-039 — C1V7-F20 variant 1

- Difficulty and verification: `easy`; independent calculation.
- Original recomputation: The join waits 5 units for B, then C takes 3, so completion is 8.
- V recomputation: Correcting A's duration to 6 makes A critical; max(6,5)+3 gives completion 9.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: Task A's one duration record, corrected from 2 to 6.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-040 — C1V7-F20 variant 2

- Difficulty and verification: `medium`; independent calculation.
- Original recomputation: Branch totals are 5 and 6; the critical branch takes 6, then D adds 2, so completion is 8.
- V recomputation: Correcting B's duration to 5 makes branch 1 total 8; max(8,6)+2 gives completion 10.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: Task B's one duration record, corrected from 2 to 5.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-041 — C1V7-F21 variant 1

- Difficulty and verification: `easy`; exhaustive finite enumeration.
- Original recomputation: X defeats both Y and Z, while no other candidate defeats every opponent, so X is unique.
- V recomputation: Reversing only X versus Y gives Y victories over X and Z, so Y becomes the unique Condorcet winner.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The complete pairwise outcome record for X versus Y, reversed from X-win to Y-win.
- Necessary steps: original 2; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-042 — C1V7-F21 variant 2

- Difficulty and verification: `medium`; exhaustive finite enumeration.
- Original recomputation: W has recorded wins over X, Y, and Z, so W alone defeats all three opponents.
- V recomputation: Correcting only W versus X to X>W gives X wins over W, Y, and Z, so X alone defeats every opponent.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The complete pairwise outcome record for W versus X, reversed from W-win to X-win.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-043 — C1V7-F22 variant 1

- Difficulty and verification: `easy`; exhaustive finite enumeration.
- Original recomputation: Singleton {Q} intersects both conflicts, while {P} and {R} each miss one, so {Q} is uniquely minimum.
- V recomputation: Replacing the second conflict with {P,R} makes singleton {P} intersect both, while {Q} and {R} each miss one; {P} is uniquely minimum.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The complete second conflict-set record, atomically replaced from {Q,R} to {P,R}.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-044 — C1V7-F22 variant 2

- Difficulty and verification: `medium`; exhaustive finite enumeration.
- Original recomputation: The intersection of all three conflicts is exactly {B}, so {B} is the unique singleton hitting every set and hence uniquely minimum.
- V recomputation: Replacing only {B,E} with {A,E} makes the intersection of all three conflicts exactly {A}, so {A} is the unique minimum diagnosis.
- Original option verdicts: A=supported; B=contradicted; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The complete third conflict-set record, atomically replaced from {B,E} to {A,E}.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-045 — C1V7-F23 variant 1

- Difficulty and verification: `easy`; exhaustive finite enumeration.
- Original recomputation: P must start, block QR must follow it, and S must be after R, leaving only PQRS.
- V recomputation: Replacing only the adjacency clue with R immediately before Q leaves P first and S after R; only PRQS satisfies all clues.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: The complete directed-adjacency clue, atomically replaced from Q-before-R to R-before-Q.
- Necessary steps: original 1; updated 2.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-046 — C1V7-F23 variant 2

- Difficulty and verification: `medium`; exhaustive finite enumeration.
- Original recomputation: A is first, blocks BC and DE must follow, and C before D orders those blocks, so only ABCDE works.
- V recomputation: Replacing only BC with directed block CB, while C remains before D, forces block CB before block DE after A, yielding ACBDE.
- Original option verdicts: A=contradicted; B=supported; C=contradicted.
- Updated option verdicts: A=contradicted; B=contradicted; C=supported.
- Sole changed node: The complete directed-adjacency clue, atomically replaced from B-before-C to C-before-B.
- Necessary steps: original 3; updated 4.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-047 — C1V7-F24 variant 1

- Difficulty and verification: `easy`; exhaustive finite enumeration.
- Original recomputation: M's closure adds N and then P, while N cannot add M and P adds nothing, so {M} is the unique minimal key.
- V recomputation: Replacing only M->N with N->M leaves N->P; N's closure now adds M and P, while M and P cannot reach all attributes, so {N} is unique.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=supported; B=contradicted; C=contradicted.
- Sole changed node: One complete functional-dependency record, atomically replaced from M -> N to N -> M.
- Necessary steps: original 3; updated 3.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

### C1V7-048 — C1V7-F24 variant 2

- Difficulty and verification: `medium`; exhaustive finite enumeration.
- Original recomputation: Under R -> Q, Q -> P, P -> S, and Q -> T, the closure of {R} is {P, Q, R, S, T}; its only proper subset is empty and is not a key. The closures of {P} and {Q} omit attributes, so C is the unique minimal key.
- V recomputation: After replacing R -> Q with Q -> R, the closure of {Q} is {P, Q, R, S, T}; its only proper subset is empty and is not a key. The closures of {P} and {R} omit attributes, so B is the unique minimal key.
- Original option verdicts: A=contradicted; B=contradicted; C=supported.
- Updated option verdicts: A=contradicted; B=supported; C=contradicted.
- Sole changed node: The one functional-dependency record R -> Q, atomically replaced by Q -> R.
- Necessary steps: original 5; updated 5.
- Family alignment: yes; the answer operator and V node type match the frozen card.
- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.
- Author decision: `pass`.

## Boundary statement

These author conclusions certify construction consistency only. They do not substitute for the two independent blind reviewers required by the freeze gate. Any later human rejection requires repair and a fresh complete review packet under the runbook.
