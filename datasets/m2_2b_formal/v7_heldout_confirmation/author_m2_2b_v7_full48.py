#!/usr/bin/env python3
# ruff: noqa: E501
"""Materialize the approved C1/V7 solution-first base specifications.

This is a deterministic, zero-model authoring script.  It copies the four approved
sentinels byte-for-byte at the object level and builds only the remaining bases.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
VERSION = "m2.2b-v7-heldout-confirmation"
SYSTEM = (
    "You are solving a closed-world evidence-revision problem. You may receive "
    "an Observation and Candidate reasoning. Treat the Observation as new task "
    "information. Candidate reasoning is an argument, not additional evidence. "
    "Independently determine whether its conclusion follows from the original "
    "information and the Observation. Do not assume the Candidate reasoning is "
    "correct or incorrect."
)
CONTRACT = (
    'Return only JSON: {"answer":"A|B|C","confidence":0-100,"reason":"one concise sentence"}.'
)
STRATUM = {
    **{number: "arithmetic_accounting" for number in range(1, 9)},
    **{number: "unit_label_rule_application" for number in range(9, 17)},
    **{number: "multistep_logic" for number in range(17, 25)},
}
SENTINEL_HASHES = {
    "C1V7-001": "e861c1a3900501bbd138f0f3f8156b5e5f4b31ab1e64b7fedbd1480a61049d8e",
    "C1V7-021": "5cc7474c1e8d0c1118149a73741c4feb0da242aec62de559e76411acbf49c6ec",
    "C1V7-033": "d44f4c1804823efb60a93a5200a7fa9bf423fcc59ca5c9f5add37546a196c115",
    "C1V7-048": "a1487c0ada6dc32bb4870d30df8d6ec1f4dfa8f2244225800d34ecf772a73042",
}


class Spec(NamedTuple):
    family: int
    variant: int
    body: str
    initial_text: str
    target_text: str
    distractor_text: str
    original_rationale: str
    updated_rationale: str
    evidence: str
    original_steps: tuple[str, ...]
    updated_steps: tuple[str, ...]
    changed_node: str
    initial_reason: str


SPECS = [
    Spec(
        1,
        2,
        "A roster has seven active scores: 12, 19, 27, 34, 46, 53, and 61. Sort all active scores from smallest to largest and report the score at rank 4, the unique median rank. Which value is the median?",
        "34",
        "27",
        "19",
        "Sorting the seven scores leaves 34 at the unique fourth rank.",
        "After replacing 53 with 15, the sorted roster is 12, 15, 19, 27, 34, 46, 61, so 27 is uniquely fourth.",
        "A roster correction replaces the active score 53 with 15; the roster still has seven records and every other score is unchanged.",
        (
            "Sort the seven active scores.",
            "Identify rank 4 as the median rank.",
            "Read 34 at rank 4.",
        ),
        ("Replace only 53 with 15.", "Re-sort the seven scores.", "Read 27 at rank 4."),
        "The one active score record corrected from 53 to 15.",
        "The fourth value in the sorted seven-score roster is 34.",
    ),
    Spec(
        2,
        1,
        "A serial line can produce only as much as its least capable stage. Effective capacity equals nominal capacity times the recorded availability factor. Stage A is 90 at 0.80, stage B is 80 at 0.95, and stage C is 100 at 0.90. What is the line's feasible throughput?",
        "72 units",
        "56 units",
        "76 units",
        "The effective capacities are 72, 76, and 90, whose unique minimum is 72.",
        "With stage B's factor corrected to 0.70, the capacities are 72, 56, and 90, whose unique minimum is 56.",
        "The stage record corrects B's availability factor from 0.95 to 0.70; all nominal capacities and the other factors are unchanged.",
        (
            "Compute the three effective capacities as 72, 76, and 90.",
            "Select their unique minimum, 72.",
        ),
        (
            "Replace only B's factor and recompute B as 80 times 0.70 = 56.",
            "Select 56 as the unique minimum.",
        ),
        "Stage B's one availability-factor record, corrected from 0.95 to 0.70.",
        "The least effective stage capacity is 72 units.",
    ),
    Spec(
        2,
        2,
        "A serial line's feasible throughput is the minimum effective capacity among all stages. Effective capacity is nominal capacity multiplied by its availability factor. The stages are A: 120 at 0.80, B: 100 at 0.90, C: 110 at 0.90, and D: 150 at 0.70. What is the feasible throughput?",
        "90 units",
        "66 units",
        "96 units",
        "The four effective capacities are 96, 90, 99, and 105, so B uniquely sets throughput at 90.",
        "Correcting C's factor to 0.60 makes its capacity 66 while the others remain 96, 90, and 105; 66 is the unique minimum.",
        "The stage record corrects C's availability factor from 0.90 to 0.60; all nominal capacities and every other factor are unchanged.",
        (
            "Multiply each nominal capacity by its factor.",
            "Obtain A=96, B=90, C=99, and D=105.",
            "Take the unique minimum, 90.",
        ),
        (
            "Replace only C's factor with 0.60.",
            "Recompute C as 110 times 0.60 = 66.",
            "Compare 66 with the unchanged 96, 90, and 105.",
            "Take the unique minimum, 66.",
        ),
        "Stage C's one availability-factor record, corrected from 0.90 to 0.60.",
        "The minimum of the four effective capacities is 90 units.",
    ),
    Spec(
        3,
        1,
        "Exactly 52 parts must be packed. A container has a certified usable capacity of 12 parts, containers cannot be split, and each container costs 8 credits. The number purchased is the smallest whole number that holds every part. What is the total cost?",
        "40 credits",
        "24 credits",
        "32 credits",
        "Ceiling of 52 divided by 12 is 5 containers, and 5 times 8 is 40 credits.",
        "At corrected capacity 18, ceiling of 52 divided by 18 is 3 containers, costing 24 credits.",
        "A packing certificate corrects usable capacity per container from 12 parts to 18 parts; the 52-part requirement and 8-credit price are unchanged.",
        ("Compute ceil(52/12)=5 whole containers.", "Multiply 5 by 8 to get 40 credits."),
        ("Compute ceil(52/18)=3 whole containers.", "Multiply 3 by 8 to get 24 credits."),
        "The certified usable-capacity record, corrected from 12 to 18 parts per container.",
        "Five indivisible containers cost 40 credits.",
    ),
    Spec(
        3,
        2,
        "A shipment has 137 indivisible samples. Each certified case can hold 24 samples and costs 11 tokens. Cases cannot be bought fractionally, so purchase the least whole number whose total capacity is at least 137. What is the purchase cost?",
        "66 tokens",
        "44 tokens",
        "55 tokens",
        "Five cases hold only 120, while six hold 144, so exactly 6 cases are required and cost 66 tokens.",
        "At corrected capacity 35, three cases hold 105 but four hold 140, so exactly 4 cases are required and cost 44 tokens.",
        "The case certificate corrects usable capacity from 24 samples to 35 samples; shipment size and price per case are unchanged.",
        (
            "Check that 5 times 24 is below 137.",
            "Check that 6 times 24 reaches 137.",
            "Multiply 6 cases by 11 tokens.",
        ),
        (
            "Replace only capacity 24 with 35.",
            "Check that 3 times 35 is below 137.",
            "Check that 4 times 35 reaches 137.",
            "Multiply 4 cases by 11 tokens.",
        ),
        "The one certified usable-capacity record, corrected from 24 to 35 samples per case.",
        "Six whole cases are necessary and cost 66 tokens.",
    ),
    Spec(
        4,
        1,
        "A batch starts with 800 units and passes two serial stages. Stage yields are deterministic: 0.75 at stage 1 and 0.80 at stage 2. Final output equals the starting amount times both yields. What is the output?",
        "480 units",
        "300 units",
        "400 units",
        "The serial product is 800 times 0.75 times 0.80 = 480 units.",
        "Replacing only stage 2's yield with 0.50 gives 800 times 0.75 times 0.50 = 300 units.",
        "A quality report corrects stage 2's yield from 0.80 to 0.50; the starting batch and stage 1 yield are unchanged.",
        ("Multiply 800 by both serial yields to obtain 480.",),
        ("Replace only the second yield.", "Multiply 800 by 0.75 and 0.50 to obtain 300."),
        "The stage-2 yield record, corrected from 0.80 to 0.50.",
        "Multiplying both serial yields gives 480 units.",
    ),
    Spec(
        4,
        2,
        "A batch of 1200 units passes three serial stages with deterministic yields 0.90, 0.80, and 0.75 in that order. No other gains or losses occur. Final output is the product of the start count and all three yields. What is the output?",
        "648 units",
        "486 units",
        "540 units",
        "The stages leave 1080, then 864, then 648 units, so final output is 648.",
        "With only the middle yield corrected to 0.60, the stages leave 1080, then 648, then 486 units.",
        "A quality report corrects the stage-2 yield from 0.80 to 0.60; the start count and the stage-1 and stage-3 yields are unchanged.",
        ("Apply the 0.90 yield to get 1080.", "Apply 0.80 to get 864.", "Apply 0.75 to get 648."),
        (
            "Keep stage 1 at 0.90 to get 1080.",
            "Apply the corrected 0.60 yield to get 648.",
            "Keep stage 3 at 0.75 to get 486.",
        ),
        "The stage-2 yield record, corrected from 0.80 to 0.60.",
        "The three-stage yield product is 648 units.",
    ),
    Spec(
        5,
        1,
        "Under the supplied ledger rule, ordinary entries contribute their listed amount and contra entries subtract it. The active entries are 120 ordinary, 70 ordinary, and 30 contra. What is the signed balance?",
        "160",
        "20",
        "90",
        "The signed sum is 120 + 70 - 30 = 160.",
        "Reclassifying only the 70 entry as contra gives 120 - 70 - 30 = 20.",
        "The account master corrects the 70 entry's formal class from ordinary to contra; all amounts and other classes are unchanged.",
        ("Apply the supplied signs and compute 120+70-30=160.",),
        ("Reverse only the 70 entry's sign.", "Compute 120-70-30=20."),
        "The complete formal-class record for the 70 entry: ordinary to contra.",
        "The signed ledger balance is 160.",
    ),
    Spec(
        5,
        2,
        "A closed ledger defines ordinary entries as positive and contra entries as negative. Its active entries are 210 ordinary, 90 ordinary, 40 contra, and 20 contra. Sum all four signed contributions. What is the balance?",
        "240",
        "60",
        "150",
        "Applying the classes gives 210 + 90 - 40 - 20 = 240.",
        "After only the 90 entry becomes contra, the balance is 210 - 90 - 40 - 20 = 60.",
        "The account master corrects the 90 entry's formal class from ordinary to contra; every amount and the other three classes are unchanged.",
        (
            "Map ordinary entries to plus signs and contra entries to minus signs.",
            "Form 210+90-40-20.",
            "Evaluate the signed sum as 240.",
        ),
        (
            "Replace only the 90 entry's plus sign with a minus sign.",
            "Form 210-90-40-20.",
            "Evaluate the corrected sum as 60.",
        ),
        "The complete formal-class record for the 90 entry: ordinary to contra.",
        "The four signed contributions sum to 240.",
    ),
    Spec(
        6,
        1,
        "A total of 132 is tax-inclusive. The closed rule is gross = base times (1 + rate), and the applicable rate is 10%. What is the base amount?",
        "120",
        "110",
        "108",
        "The base is 132 divided by 1.10 = 120.",
        "At the corrected 20% rate, the base is 132 divided by 1.20 = 110.",
        "The jurisdiction record corrects the applicable rate from 10% to 20%; gross 132 and its tax-inclusive status are unchanged.",
        ("Invert gross=base times 1.10 to obtain 120.",),
        ("Replace only the rate multiplier with 1.20.", "Divide 132 by 1.20 to obtain 110."),
        "The one applicable-rate record, corrected from 10% to 20%.",
        "Dividing the inclusive gross by 1.10 gives base 120.",
    ),
    Spec(
        6,
        2,
        "An invoice gross of 207 is explicitly tax-inclusive. Use only the equation gross = base times (1 + applicable rate). The recorded applicable rate is 15%. What base amount generated the gross?",
        "180",
        "150",
        "160",
        "The multiplier is 1.15, and 207 divided by 1.15 gives the unique base 180.",
        "The corrected multiplier is 1.38, and 207 divided by 1.38 gives the unique base 150.",
        "The jurisdiction record corrects the applicable rate from 15% to 38%; gross 207 and inclusive status are unchanged.",
        (
            "Convert the 15% rate to multiplier 1.15.",
            "Rearrange the supplied equation as base=gross/1.15.",
            "Compute 207/1.15=180.",
        ),
        (
            "Replace only the rate with 38% and form multiplier 1.38.",
            "Use base=gross/1.38.",
            "Compute 207/1.38=150.",
        ),
        "The one applicable-rate record, corrected from 15% to 38%.",
        "The supplied inclusive equation yields base 180.",
    ),
    Spec(
        7,
        1,
        "A vehicle covers 180 km in two legs, half the distance at 45 km/h and half at 90 km/h. Average speed means total distance divided by total travel time. What is the trip's average speed?",
        "60 km/h",
        "67.5 km/h",
        "75 km/h",
        "The 90 km legs take 2 and 1 hours, so 180 divided by 3 is 60 km/h.",
        "At corrected distance ratio 1:2, the legs are 60 and 120 km and each takes 4/3 hours, so 180 divided by 8/3 is 67.5 km/h.",
        "A route audit corrects the distance ratio of the 45 km/h leg to the 90 km/h leg from 1:1 to 1:2; total distance and both speeds are unchanged.",
        ("Compute leg times 90/45=2 and 90/90=1.", "Divide 180 by total time 3 to get 60."),
        (
            "Use the corrected ratio to obtain 60 km and 120 km.",
            "Compute total time as 60/45+120/90=8/3 hours.",
            "Divide 180 by 8/3 to get 67.5.",
        ),
        "The certified two-leg distance-ratio record, corrected from 1:1 to 1:2.",
        "Total distance divided by the two leg times is 60 km/h.",
    ),
    Spec(
        7,
        2,
        "A 216 km route has two legs traveled at 36 km/h and 72 km/h. The recorded leg-distance ratio is 1:1. Derive each distance from the ratio, compute each travel time, then divide total distance by total time. What is the average speed?",
        "48 km/h",
        "54 km/h",
        "60 km/h",
        "Equal legs are 108 km; their times are 3 and 1.5 hours, giving 216/4.5 = 48 km/h.",
        "At corrected ratio 1:2, distances are 72 and 144 km; times are 2 and 2 hours, giving 216/4 = 54 km/h.",
        "A route audit corrects the distance ratio of the 36 km/h leg to the 72 km/h leg from 1:1 to 1:2; total distance and speeds are unchanged.",
        (
            "Split 216 equally into two 108 km legs.",
            "Compute times 108/36=3 and 108/72=1.5.",
            "Divide 216 by 4.5 to get 48.",
        ),
        (
            "Replace only the ratio and split 216 into 72 and 144.",
            "Compute times 72/36=2 and 144/72=2.",
            "Add the times to get 4 hours.",
            "Divide 216 by 4 to get 54.",
        ),
        "The certified two-leg distance-ratio record, corrected from 1:1 to 1:2.",
        "The ratio-derived leg times give an average of 48 km/h.",
    ),
    Spec(
        8,
        1,
        "A workflow starts with 200 units. The first pass accepts 60% of its input. Each permitted rework pass then accepts 50% of the units rejected by the preceding pass. Exactly one rework pass is permitted. How many units are accepted in total?",
        "160 units",
        "180 units",
        "170 units",
        "The first pass accepts 120 and the one rework pass accepts 40, totaling 160.",
        "A second permitted rework pass accepts half of the remaining 40, adding 20 for a total of 180.",
        "The workflow record corrects the permitted rework-pass count from one to two; the start count and both acceptance rates are unchanged.",
        ("Accept 120 initially and 40 on the one rework pass.", "Add them to obtain 160."),
        (
            "Keep the original accepted 160.",
            "Apply the added second pass to the remaining 40 and add 20, obtaining 180.",
        ),
        "The one permitted rework-pass-count record, corrected from 1 to 2.",
        "The initial and one rework acceptance total 160 units.",
    ),
    Spec(
        8,
        2,
        "A workflow starts with 160 units. The initial pass accepts 50% of its input. Every permitted rework pass accepts 25% of what the immediately preceding pass rejected. Exactly two rework passes are permitted, after which processing stops. How many units are accepted?",
        "115 units",
        "126.25 units",
        "120 units",
        "The passes accept 80, then 20, then 15 units, for a deterministic total of 115.",
        "A third rework pass acts on the remaining 45 and accepts 11.25 more, raising the total to 126.25.",
        "The workflow record corrects the permitted rework-pass count from two to three; the 160-unit start and both acceptance rates are unchanged.",
        (
            "Accept 80 on the initial pass, leaving 80.",
            "Accept 20 on rework 1, leaving 60.",
            "Accept 15 on rework 2 and total 80+20+15=115.",
        ),
        (
            "Preserve the first three accepted amounts totaling 115.",
            "Carry the 45 units rejected by rework 2 into the added pass.",
            "Accept 25% of 45, which is 11.25.",
            "Add 11.25 to obtain 126.25.",
        ),
        "The one permitted rework-pass-count record, corrected from 2 to 3.",
        "The three accepted amounts sum to 115 units.",
    ),
    Spec(
        9,
        1,
        "A routing table uses the unique longest active prefix of a key. Its entries are K -> Amber, KL -> Blue, and KLM -> Coral. For key KLMN, which label is selected?",
        "Coral",
        "Blue",
        "Amber",
        "K, KL, and KLM all match KLMN, and the unique longest prefix KLM selects Coral.",
        "After only the KLM entry is removed, KL is the unique longest remaining match and selects Blue.",
        "A routing-table correction removes the complete KLM -> Coral entry; key KLMN and the broader K and KL entries are unchanged.",
        ("List matching prefixes K, KL, and KLM.", "Use the longest, KLM, to select Coral."),
        ("Remove only the KLM entry.", "Use the next-longest match KL to select Blue."),
        "The complete most-specific routing-table entry KLM -> Coral, changed from active to removed.",
        "The longest matching prefix KLM selects Coral.",
    ),
    Spec(
        9,
        2,
        "A routing table selects the label attached to the unique longest active prefix of a key. Active entries are T -> Ivory, TA -> Jade, TAN -> Lilac, and TANG -> Ochre. For key TANGO7, which label is selected?",
        "Ochre",
        "Lilac",
        "Jade",
        "All four prefixes match TANGO7, and the unique longest, TANG, selects Ochre.",
        "Replacing the TANG entry with nonmatching TARP leaves TAN as the unique longest active match, which selects Lilac.",
        "A table correction replaces the complete entry TANG -> Ochre with TARP -> Ochre; the key and all broader entries are unchanged.",
        (
            "Test each active prefix against TANGO7.",
            "Order matching prefixes T, TA, TAN, and TANG by length.",
            "Use the unique longest match TANG to select Ochre.",
        ),
        (
            "Replace only the complete TANG entry with TARP.",
            "Test TARP and confirm it does not prefix TANGO7.",
            "Order the unchanged matching entries T, TA, and TAN.",
            "Use the unique longest TAN to select Lilac.",
        ),
        "The complete most-specific routing-table record, replaced from TANG -> Ochre to TARP -> Ochre.",
        "The unique longest active match is TANG, labeled Ochre.",
    ),
    Spec(
        10,
        1,
        "Compare two versions using only these rules: compare numeric core components first; equal cores proceed to prerelease identifiers; for unequal alphabetic prerelease identifiers, ordinary lexical order decides, with the later word having higher precedence. The versions are package P: 1.8.0-alpha and package Q: 1.8.0-beta. Which has higher precedence?",
        "Package Q",
        "Package P",
        "They have equal precedence",
        "The numeric cores tie, and beta is lexically later than alpha, so package Q is uniquely higher.",
        "With P's identifier corrected to rc, rc is lexically later than beta, so package P is uniquely higher.",
        "The release manifest corrects package P's prerelease identifier from alpha to rc; both numeric cores and package Q are unchanged.",
        (
            "The numeric cores tie, so compare alpha with beta.",
            "Lexical order makes beta later, selecting package Q.",
        ),
        (
            "Replace only P's identifier with rc.",
            "Lexical order makes rc later than beta, selecting package P.",
        ),
        "Package P's complete prerelease-identifier record, corrected from alpha to rc.",
        "The tied cores are decided by beta being later than alpha.",
    ),
    Spec(
        10,
        2,
        "Use only these version rules. Compare numeric core components first. If cores tie, compare dot-separated prerelease identifiers left to right; equal alphabetic identifiers advance to the next identifier, and unequal numeric identifiers are ordered numerically. Package P is 2.4.0-rc.9 and package Q is 2.4.0-rc.11. Which has higher precedence?",
        "Package Q",
        "Package P",
        "They have equal precedence",
        "The cores and first identifier rc tie, then numeric 11 exceeds 9, so package Q is uniquely higher.",
        "Correcting P's prerelease record to rc.15 makes 15 exceed 11 at the first unequal identifier, so package P is higher.",
        "The release manifest corrects package P's complete prerelease identifier from rc.9 to rc.15; numeric cores and package Q are unchanged.",
        (
            "Compare the numeric cores and find them equal.",
            "Compare the first prerelease identifiers rc and find them equal.",
            "Compare numeric identifiers 9 and 11, selecting package Q.",
        ),
        (
            "Replace only P's prerelease record with rc.15.",
            "Confirm that cores and rc still tie.",
            "Compare numeric identifiers 15 and 11.",
            "Select package P because 15 is larger.",
        ),
        "Package P's complete prerelease-identifier record, corrected from rc.9 to rc.15.",
        "After the core and rc ties, numeric 11 gives package Q higher precedence.",
    ),
    Spec(
        11,
        2,
        "A nominal value is 75 units with allowed tolerance 5 units, so the closed acceptance band is [75-5, 75+5]. A sensor reports center 74 with uncertainty half-width 3, giving [center-half-width, center+half-width]. Classify the sensor interval by exhaustive rules: fully contained if both endpoints lie in the acceptance band; fully excluded if the closed intervals share no point; otherwise partially overlaps. Which label applies?",
        "Fully contained in the acceptance band",
        "Partially overlaps the acceptance band",
        "Fully excluded from the acceptance band",
        "The acceptance band is [70,80] and the sensor interval [71,77] has both endpoints inside it, so it is fully contained.",
        "At corrected half-width 7, the sensor interval is [67,81]; it intersects [70,80] but is not fully contained, so it partially overlaps.",
        "A calibration certificate corrects the sensor's uncertainty half-width from 3 units to 7 units; center 74, nominal value, and tolerance are unchanged.",
        (
            "Compute the acceptance band [70,80].",
            "Compute the sensor interval [71,77].",
            "Apply the endpoint rule to label it fully contained.",
        ),
        (
            "Keep the acceptance band [70,80].",
            "Replace only half-width 3 with 7.",
            "Compute the corrected sensor interval [67,81].",
            "Apply the exhaustive rules to label it partial overlap.",
        ),
        "The sensor's one uncertainty half-width record, corrected from 3 units to 7 units.",
        "The interval [71,77] lies fully inside [70,80].",
    ),
    Spec(
        12,
        1,
        "Parse a record using | as delimiter. A backslash immediately before | escapes that delimiter, makes the | literal field content, and is not retained. Unescaped | ends a field. The raw record is oak\\|pine|birch|cedar. What is field 2, counting from 1?",
        "birch",
        "pine",
        "cedar",
        "The escaped first bar stays inside field 1, so the fields are oak|pine, birch, cedar and field 2 is birch.",
        "Removing only that escape makes every bar a delimiter, producing oak, pine, birch, cedar; field 2 is pine.",
        "A raw-record correction removes the one backslash immediately before the first |; all letters and all delimiter characters remain unchanged.",
        ("Parse the escaped first bar as literal content.", "Read field 2 as birch."),
        ("Remove only the first escape marker.", "Reparse and read field 2 as pine."),
        "The single escape marker immediately before the first delimiter, changed from present to absent.",
        "The escape keeps oak|pine together, making birch field 2.",
    ),
    Spec(
        12,
        2,
        "Parse left to right using | as delimiter. A backslash immediately before | escapes that delimiter, contributes a literal | to the current field, and is discarded; every unescaped | ends a field. The raw record is north\\|west|central\\|hub|south|east. What is field 3, counting from 1?",
        "south",
        "east",
        "central|hub",
        "Parsing yields fields north|west, central|hub, south, east, so field 3 is south.",
        "Escaping the delimiter between hub and south merges those parts into field 2; the fields become north|west, central|hub|south, east, so field 3 is east.",
        "A raw-record correction inserts one backslash immediately before the delimiter between hub and south; every other character is unchanged.",
        (
            "Resolve the escaped bar inside north|west.",
            "Resolve the escaped bar inside central|hub.",
            "Split at the remaining delimiters and read field 3 as south.",
        ),
        (
            "Insert only the new escape marker before the hub-south delimiter.",
            "Parse north|west as field 1.",
            "Parse central|hub|south as field 2.",
            "Read east as field 3.",
        ),
        "The single escape marker before the delimiter between hub and south, changed from absent to present.",
        "Sequential parsing makes south the third field.",
    ),
    Spec(
        13,
        1,
        "A closed encoding rule assigns 10*log10(ratio) to a power ratio and 20*log10(ratio) to an amplitude ratio. The ratio is 1000, it is recorded as a power ratio, and log10(1000)=3. What encoded value follows?",
        "30 dB",
        "60 dB",
        "20 dB",
        "The power selector uses multiplier 10, giving 10 times 3 = 30 dB.",
        "Changing only the quantity type to amplitude selects multiplier 20, giving 20 times 3 = 60 dB.",
        "A metadata correction changes the ratio's quantity type from power to amplitude; ratio 1000 and the supplied logarithm are unchanged.",
        ("Use the power multiplier 10 and compute 10*3=30 dB.",),
        (
            "Replace only the type selector with amplitude.",
            "Use multiplier 20 and compute 20*3=60 dB.",
        ),
        "The one quantity-type metadata record, corrected from power to amplitude.",
        "The power-ratio multiplier gives 30 dB.",
    ),
    Spec(
        13,
        2,
        "Use only this encoding: power ratios use 10*log10(ratio), while amplitude ratios use 20*log10(ratio). For ratio 16, use the supplied log10(16)=1.2041 and round the final result to one decimal place. The quantity is recorded as a power ratio. What is its encoded value?",
        "12.0 dB",
        "24.1 dB",
        "16.0 dB",
        "The power selector gives 10*1.2041=12.041, which rounds to 12.0 dB.",
        "The corrected amplitude selector gives 20*1.2041=24.082, which rounds to 24.1 dB.",
        "A metadata correction changes the quantity type from power to amplitude; ratio 16, supplied logarithm, and rounding rule are unchanged.",
        (
            "Select multiplier 10 for the recorded power type.",
            "Compute 10*1.2041=12.041.",
            "Round once at the end to 12.0 dB.",
        ),
        (
            "Replace only the type selector with amplitude.",
            "Select multiplier 20.",
            "Compute 20*1.2041=24.082.",
            "Round once at the end to 24.1 dB.",
        ),
        "The one quantity-type metadata record, corrected from power to amplitude.",
        "The power formula rounds to 12.0 dB.",
    ),
    Spec(
        14,
        1,
        "Use this closed lookup table: (UDP,88) -> Maple; (TCP,88) -> Quartz; (UDP,89) -> Silver. A packet header records protocol UDP and port 88. Which label does its exact ordered pair select?",
        "Maple",
        "Quartz",
        "Silver",
        "The header tuple is exactly (UDP,88), whose unique table label is Maple.",
        "Correcting only the protocol makes the tuple (TCP,88), whose unique table label is Quartz.",
        "A packet-header correction changes the protocol component from UDP to TCP; numeric port 88 and the lookup table are unchanged.",
        ("Form tuple (UDP,88) and look it up as Maple.",),
        ("Replace only UDP with TCP.", "Look up (TCP,88) as Quartz."),
        "The packet header's one protocol component, corrected from UDP to TCP.",
        "Exact tuple (UDP,88) maps to Maple.",
    ),
    Spec(
        14,
        2,
        "Use this exact ordered-pair table: (UDP,53) -> Lumen; (TCP,53) -> Nacre; (UDP,54) -> Opal. A packet records protocol UDP. Its numeric port is defined in the packet as 40+13. Which table label is selected?",
        "Lumen",
        "Nacre",
        "Opal",
        "The port is 53, so tuple (UDP,53) uniquely selects Lumen.",
        "With only protocol corrected to TCP, the unchanged port 53 forms (TCP,53), which uniquely selects Nacre.",
        "A packet-header correction changes the protocol component from UDP to TCP; the port expression 40+13 and the table are unchanged.",
        (
            "Evaluate the packet's port as 40+13=53.",
            "Form ordered pair (UDP,53).",
            "Look up that pair as Lumen.",
        ),
        (
            "Keep the evaluated port 53.",
            "Replace only the protocol with TCP.",
            "Form ordered pair (TCP,53).",
            "Look up that pair as Nacre.",
        ),
        "The packet header's one protocol component, corrected from UDP to TCP.",
        "Port 53 paired with UDP maps to Lumen.",
    ),
    Spec(
        15,
        1,
        "Requested permissions are the four-bit pattern 1101. An active mask blocks every position containing 1, so effective permissions are requested AND bitwise-NOT(mask), restricted to four bits. The mask is 0100. What effective pattern results?",
        "1001",
        "1000",
        "1100",
        "NOT 0100 leaves the requested bits 1001 effective.",
        "Correcting only the mask to 0101 blocks the last requested bit too, giving 1000.",
        "A policy correction changes the active mask from 0100 to 0101; requested pattern 1101 and the Boolean rule are unchanged.",
        ("Apply 1101 AND NOT(0100) to obtain 1001.",),
        ("Replace only the mask with 0101.", "Apply 1101 AND NOT(0101) to obtain 1000."),
        "The active four-bit mask record, corrected from 0100 to 0101.",
        "Masking 1101 by 0100 leaves 1001.",
    ),
    Spec(
        15,
        2,
        "Requested permissions are 11111. Effective bits equal requested AND bitwise-NOT(mask), restricted to five bits. A label table maps 11010 -> Falcon, 10010 -> Harbor, and 10110 -> Juniper. The active mask is 00101. Which label applies to the effective pattern?",
        "Falcon",
        "Harbor",
        "Juniper",
        "Masking 11111 by 00101 gives 11010, which the exact table maps to Falcon.",
        "Correcting the mask to 01101 gives effective pattern 10010, which the table maps to Harbor.",
        "A policy correction activates one additional mask bit, changing the complete mask record from 00101 to 01101; requested bits and label table are unchanged.",
        (
            "Complement the five-bit mask 00101.",
            "AND it with requested pattern 11111 to obtain 11010.",
            "Look up 11010 as Falcon.",
        ),
        (
            "Replace only the mask record with 01101.",
            "Complement the corrected five-bit mask.",
            "AND it with 11111 to obtain 10010.",
            "Look up 10010 as Harbor.",
        ),
        "The active five-bit mask record, corrected from 00101 to 01101 by one bit.",
        "The effective pattern 11010 maps to Falcon.",
    ),
    Spec(
        16,
        1,
        "A 30-day month has day 1 on Friday. Weekdays repeat every seven days. What calendar date is the fourth Tuesday of that month?",
        "Day 26",
        "Day 24",
        "Day 22",
        "Tuesdays are days 5, 12, 19, and 26, so the fourth is day 26.",
        "If day 1 is corrected to Sunday, Tuesdays are 3, 10, 17, and 24, so the fourth is day 24.",
        "A calendar bulletin corrects the weekday of day 1 from Friday to Sunday; month length and the requested fourth Tuesday are unchanged.",
        ("List Tuesdays as 5,12,19,26 and select the fourth.",),
        (
            "Replace only the day-1 weekday with Sunday.",
            "List Tuesdays as 3,10,17,24 and select the fourth.",
        ),
        "The month's one day-1 weekday anchor, corrected from Friday to Sunday.",
        "The fourth Tuesday falls on day 26.",
    ),
    Spec(
        16,
        2,
        "A 30-day month has day 1 on Wednesday, and weekdays repeat every seven days. If an occurrence would fall after day 30, it does not occur. What is the date of the fifth Monday?",
        "There is no fifth Monday",
        "Day 29",
        "Day 30",
        "Mondays are days 6, 13, 20, and 27; the next would be 34, so no fifth Monday occurs.",
        "If day 1 is Monday, Mondays are 1, 8, 15, 22, and 29, so the fifth is day 29.",
        "A calendar bulletin corrects the weekday of day 1 from Wednesday to Monday; the 30-day month and requested fifth Monday are unchanged.",
        (
            "Find the first Monday as day 6.",
            "Advance by seven to get 6,13,20,27.",
            "Check that the next date 34 exceeds day 30.",
        ),
        (
            "Replace only the day-1 weekday with Monday.",
            "Use day 1 as the first Monday.",
            "Advance by seven to list 1,8,15,22,29.",
            "Select day 29 as the fifth occurrence.",
        ),
        "The month's one day-1 weekday anchor, corrected from Wednesday to Monday.",
        "Only four Mondays occur within the 30-day month.",
    ),
    Spec(
        17,
        2,
        "Assign items 1, 2, 3, and 4 bijectively to agents A, B, C, and D, one item per agent and one agent per item. Complete allowed-item records are A:{1}, B:{1,2}, C:{2,3}, and D:{3,4}. Which complete assignment satisfies every record?",
        "A=1, B=2, C=3, D=4",
        "A=4, B=1, C=2, D=3",
        "A=4, B=2, C=3, D=1",
        "A is forced to 1, then B to 2, C to 3, and D to 4; exhaustive checking leaves this single perfect matching.",
        "Replacing A's complete record with {4} forces A=4, D=3, C=2, and B=1; this is the single updated perfect matching.",
        "An eligibility correction replaces agent A's complete allowed-item record {1} with {4}; the other three complete records and item set are unchanged.",
        (
            "Force A=1 from A's singleton record.",
            "Remove item 1 and force B=2.",
            "Then force C=3 and D=4, confirming the unique bijection.",
        ),
        (
            "Replace only A's complete record and force A=4.",
            "Remove item 4 and force D=3.",
            "Remove item 3 and force C=2.",
            "Assign the remaining item 1 to B and confirm the unique bijection.",
        ),
        "Agent A's complete allowed-item record, atomically replaced from {1} to {4}.",
        "The singleton cascade yields the unique assignment A=1, B=2, C=3, D=4.",
    ),
    Spec(
        18,
        1,
        "Intervals are closed and written [start, start+duration]. For this task, A contains B if both endpoints of B lie in A; A is before B if A's end is less than B's start; otherwise A overlaps B. Interval A starts at 3 with duration 7, and B starts at 6 with duration 2. What is A's relation to B?",
        "A contains B",
        "A is before B",
        "A overlaps B",
        "A=[3,10] and B=[6,8], so both B endpoints lie in A and A contains B.",
        "With A's duration corrected to 2, A=[3,5], whose end is less than B's start 6, so A is before B.",
        "A duration record is corrected from 7 to 2; both interval starts and B's duration are unchanged.",
        ("Construct A=[3,10] and B=[6,8].", "Apply the supplied endpoint rule to select contains."),
        (
            "Replace only A's duration and construct A=[3,5].",
            "Compare end 5 with B's start 6 to select before.",
        ),
        "Interval A's one duration record, corrected from 7 to 2.",
        "Both endpoints of B lie within A, so A contains B.",
    ),
    Spec(
        18,
        2,
        "Closed interval X is [start, start+duration]. Relative to Y, label X 'contains' if both Y endpoints lie in X, 'before' if X's end is less than Y's start, and 'overlaps' otherwise. X starts at 5 with duration 10; Y starts at 12 with duration 6. Which directed relation holds?",
        "X overlaps Y",
        "X is before Y",
        "X contains Y",
        "X=[5,15] and Y=[12,18]; they intersect, X does not contain Y, and X is not before Y, so X overlaps Y.",
        "Correcting X's duration to 6 gives X=[5,11], whose end is less than Y's start 12, so X is before Y.",
        "The duration record for X is corrected from 10 to 6; both starts and Y's duration are unchanged.",
        (
            "Construct X=[5,15].",
            "Construct Y=[12,18].",
            "Test contains and before, then apply the remaining overlap label.",
        ),
        (
            "Replace only X's duration and construct X=[5,11].",
            "Keep Y=[12,18].",
            "Compare X's end 11 with Y's start 12.",
            "Select the directed relation before.",
        ),
        "Interval X's one duration record, corrected from 10 to 6.",
        "The constructed intervals intersect without containment, so X overlaps Y.",
    ),
    Spec(
        19,
        1,
        "A proposal passes exactly when yes-vote weight is at least 6 and every veto-designated member votes yes. P is the sole veto member with weight 2 and votes no; Q has weight 3 and votes yes; R has weight 3 and votes yes. What is the decision?",
        "Fail",
        "Pass",
        "No decision",
        "Yes weight is 6, but veto member P votes no, so the conjunction fails and the proposal fails.",
        "Correcting only P's vote to yes gives weight 8 and satisfies the veto condition, so the proposal passes.",
        "A certified roll-call correction changes veto member P's recorded vote from no to yes; all weights, designations, and other votes are unchanged.",
        ("Sum yes weight as 3+3=6.", "Check the veto conjunct and select fail because P voted no."),
        (
            "Replace only P's vote with yes.",
            "Verify weight 8 and all veto votes yes, then select pass.",
        ),
        "Veto member P's one certified vote record, corrected from no to yes.",
        "The threshold holds but the required veto vote does not, so the decision is fail.",
    ),
    Spec(
        19,
        2,
        "A motion passes exactly if total yes-vote weight is at least 10 and every veto member votes yes. Veto member A has weight 4 and votes yes. Non-veto B has weight 4 and votes yes, C has weight 3 and votes yes, and D has weight 2 and votes no. What is the decision?",
        "Pass",
        "Fail",
        "No decision",
        "Yes weight is 4+4+3=11, at least 10, and the sole veto member votes yes, so the motion passes.",
        "Correcting only B's vote to no reduces yes weight to 7 while veto remains satisfied, so the threshold fails and the motion fails.",
        "A certified roll-call correction changes non-veto member B's recorded vote from yes to no; all weights, designations, and other votes are unchanged.",
        (
            "Sum the yes weights 4+4+3=11.",
            "Compare 11 with threshold 10.",
            "Check the sole veto vote and apply the conjunction to select pass.",
        ),
        (
            "Replace only B's vote with no.",
            "Recompute yes weight as 4+3=7.",
            "Keep the veto condition satisfied.",
            "Apply the failed threshold conjunct to select fail.",
        ),
        "Non-veto member B's one certified vote record, corrected from yes to no.",
        "Weight 11 clears the threshold and the veto condition holds.",
    ),
    Spec(
        20,
        1,
        "Tasks A and B begin together. Task C starts only after both finish. Durations are A=2, B=5, and C=3. Project completion time is max(A,B)+C. What is the completion time?",
        "8 time units",
        "9 time units",
        "10 time units",
        "The join waits 5 units for B, then C takes 3, so completion is 8.",
        "Correcting A's duration to 6 makes A critical; max(6,5)+3 gives completion 9.",
        "A schedule correction changes task A's duration from 2 to 6; all graph edges and other task durations are unchanged.",
        ("Take max(2,5)=5 and add C's 3 to get 8.",),
        ("Replace only A's duration with 6.", "Take max(6,5)+3 to get 9."),
        "Task A's one duration record, corrected from 2 to 6.",
        "The longer prerequisite plus task C gives 8 time units.",
    ),
    Spec(
        20,
        2,
        "A project has two branches that must both finish before task D. Branch 1 runs A then B, with durations A=3 and B=2. Branch 2 is task C with duration 6. Task D then takes 2. Completion is max(A+B,C)+D. What is the completion time?",
        "8 time units",
        "10 time units",
        "11 time units",
        "Branch totals are 5 and 6; the critical branch takes 6, then D adds 2, so completion is 8.",
        "Correcting B's duration to 5 makes branch 1 total 8; max(8,6)+2 gives completion 10.",
        "A schedule correction changes task B's duration from 2 to 5; graph edges and every other duration are unchanged.",
        (
            "Sum branch 1 as A+B=5.",
            "Compare it with branch 2 duration 6.",
            "Add D's 2 to the maximum to get 8.",
        ),
        (
            "Replace only B's duration and recompute branch 1 as 3+5=8.",
            "Keep branch 2 at 6.",
            "Select the new maximum 8.",
            "Add D's 2 to get 10.",
        ),
        "Task B's one duration record, corrected from 2 to 5.",
        "The six-unit branch plus D gives completion time 8.",
    ),
    Spec(
        21,
        1,
        "A Condorcet winner must defeat every other candidate in the recorded pairwise contests. The records are X defeats Y, X defeats Z, and Y defeats Z. Who is the unique Condorcet winner?",
        "X",
        "Y",
        "No Condorcet winner",
        "X defeats both Y and Z, while no other candidate defeats every opponent, so X is unique.",
        "Reversing only X versus Y gives Y victories over X and Z, so Y becomes the unique Condorcet winner.",
        "An election audit corrects the X-versus-Y outcome from X defeats Y to Y defeats X; the other two pairwise records are unchanged.",
        (
            "Check X against Y and Z and see two victories.",
            "Confirm the other candidates lack all-opponent victories.",
        ),
        ("Replace only the X-Y edge.", "Check Y's unchanged win over Z and corrected win over X."),
        "The complete pairwise outcome record for X versus Y, reversed from X-win to Y-win.",
        "X alone defeats every other candidate.",
    ),
    Spec(
        21,
        2,
        "A Condorcet winner is a candidate who defeats every other candidate pairwise. For W, X, Y, Z, the records are W>X, W>Y, W>Z, X>Y, X>Z, and Y>Z, where > means defeats. Who is the unique Condorcet winner?",
        "W",
        "X",
        "No Condorcet winner",
        "W has recorded wins over X, Y, and Z, so W alone defeats all three opponents.",
        "Correcting only W versus X to X>W gives X wins over W, Y, and Z, so X alone defeats every opponent.",
        "An election audit corrects the W-versus-X outcome from W>X to X>W; all five other pairwise outcomes are unchanged.",
        (
            "Check W's outcomes against X, Y, and Z.",
            "Find three wins for W.",
            "Check that each other candidate has at least one loss and select W.",
        ),
        (
            "Replace only the W-X edge with X>W.",
            "Combine it with unchanged X>Y and X>Z.",
            "Find that X now has three wins.",
            "Check every other candidate has a loss and select X.",
        ),
        "The complete pairwise outcome record for W versus X, reversed from W-win to X-win.",
        "W is the only candidate with a win over all three opponents.",
    ),
    Spec(
        22,
        1,
        "A diagnosis is a set that intersects every conflict set; a minimum diagnosis has the smallest cardinality. The conflict sets are {P,Q} and {Q,R}. Which is the unique minimum diagnosis?",
        "{Q}",
        "{P}",
        "{R}",
        "Singleton {Q} intersects both conflicts, while {P} and {R} each miss one, so {Q} is uniquely minimum.",
        "Replacing the second conflict with {P,R} makes singleton {P} intersect both, while {Q} and {R} each miss one; {P} is uniquely minimum.",
        "A diagnostic correction replaces the complete second conflict set {Q,R} with {P,R}; the first conflict set and universe are unchanged.",
        ("Test all singleton diagnoses and find only {Q} hits both sets.",),
        (
            "Replace only the second complete conflict set.",
            "Test all singletons and find only {P} hits both sets.",
        ),
        "The complete second conflict-set record, atomically replaced from {Q,R} to {P,R}.",
        "Only singleton {Q} intersects both conflicts.",
    ),
    Spec(
        22,
        2,
        "A diagnosis intersects every conflict set, and a minimum diagnosis has least cardinality. The active conflicts are {A,B,C}, {A,B,D}, and {B,E}. Considering the universe {A,B,C,D,E}, which is the unique minimum diagnosis?",
        "{B}",
        "{A}",
        "{C}",
        "The intersection of all three conflicts is exactly {B}, so {B} is the unique singleton hitting every set and hence uniquely minimum.",
        "Replacing only {B,E} with {A,E} makes the intersection of all three conflicts exactly {A}, so {A} is the unique minimum diagnosis.",
        "A diagnostic correction replaces the complete third conflict set {B,E} with {A,E}; the first two sets and universe are unchanged.",
        (
            "Intersect the first two conflicts to obtain {A,B}.",
            "Intersect {A,B} with {B,E} to obtain {B}.",
            "Verify the singleton hits every set and is uniquely minimum.",
        ),
        (
            "Replace only the third complete conflict set with {A,E}.",
            "Keep the first-two intersection {A,B}.",
            "Intersect it with {A,E} to obtain {A}.",
            "Verify singleton {A} hits every set and is uniquely minimum.",
        ),
        "The complete third conflict-set record, atomically replaced from {B,E} to {A,E}.",
        "The only element common to every conflict is B.",
    ),
    Spec(
        23,
        1,
        "Arrange P, Q, R, S once each. P is first; Q is immediately before R; and S is after R. Which complete sequence uniquely satisfies the constraints?",
        "PQRS",
        "PRQS",
        "PSRQ",
        "P must start, block QR must follow it, and S must be after R, leaving only PQRS.",
        "Replacing only the adjacency clue with R immediately before Q leaves P first and S after R; only PRQS satisfies all clues.",
        "A clue correction replaces the complete adjacency record 'Q immediately before R' with 'R immediately before Q'; the other clues are unchanged.",
        ("Place P first and block QR next; S must follow R, yielding PQRS.",),
        (
            "Replace only block QR with block RQ.",
            "Place P first and enforce S after R, yielding PRQS.",
        ),
        "The complete directed-adjacency clue, atomically replaced from Q-before-R to R-before-Q.",
        "The endpoint and directed-adjacency constraints force PQRS.",
    ),
    Spec(
        23,
        2,
        "Arrange A, B, C, D, E once each. A is first; B is immediately before C; D is immediately before E; and C is before D. Which complete sequence uniquely satisfies all constraints?",
        "ABCDE",
        "ACBDE",
        "ABDCE",
        "A is first, blocks BC and DE must follow, and C before D orders those blocks, so only ABCDE works.",
        "Replacing only BC with directed block CB, while C remains before D, forces block CB before block DE after A, yielding ACBDE.",
        "A clue correction replaces the complete adjacency record 'B immediately before C' with 'C immediately before B'; all other clues are unchanged.",
        (
            "Treat BC and DE as directed blocks.",
            "Use C before D to order BC before DE.",
            "Place A first to obtain ABCDE.",
        ),
        (
            "Replace only block BC with directed block CB.",
            "Keep directed block DE.",
            "Use C before D to order CB before DE.",
            "Place A first to obtain ACBDE.",
        ),
        "The complete directed-adjacency clue, atomically replaced from B-before-C to C-before-B.",
        "The ordered blocks after A force the sequence ABCDE.",
    ),
    Spec(
        24,
        1,
        "For attributes {M,N,P}, a set is a candidate key if its closure under the listed functional dependencies contains all attributes and no proper subset does. The only dependencies are M -> N and N -> P. Which singleton is the unique minimal key?",
        "{M}",
        "{N}",
        "{P}",
        "M's closure adds N and then P, while N cannot add M and P adds nothing, so {M} is the unique minimal key.",
        "Replacing only M->N with N->M leaves N->P; N's closure now adds M and P, while M and P cannot reach all attributes, so {N} is unique.",
        "A schema correction replaces the complete dependency M -> N with N -> M; attribute universe and dependency N -> P are unchanged.",
        (
            "Compute M+ by adding N then P.",
            "Compute N+ and P+ and see that each lacks M.",
            "Check the empty set is not a key and select {M}.",
        ),
        (
            "Replace only M->N with N->M and compute N+ as {N,M,P}.",
            "Compute M+ and P+ and see they are incomplete.",
            "Check minimality and select {N}.",
        ),
        "One complete functional-dependency record, atomically replaced from M -> N to N -> M.",
        "Only M closes to all three attributes.",
    ),
]


def canonical_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def migration_for(family: int, variant: int) -> tuple[str, str]:
    patterns = (
        (("A", "B"), ("A", "C")),
        (("B", "A"), ("B", "C")),
        (("C", "A"), ("C", "B")),
    )
    return patterns[(family - 1) % 3][variant - 1]


def make_base(spec: Spec) -> dict[str, object]:
    number = 2 * (spec.family - 1) + spec.variant
    base_id = f"C1V7-{number:03d}"
    family_id = f"C1V7-F{spec.family:02d}"
    initial, target = migration_for(spec.family, spec.variant)
    distractor = next(label for label in ("A", "B", "C") if label not in {initial, target})
    options = {
        initial: spec.initial_text,
        target: spec.target_text,
        distractor: spec.distractor_text,
    }
    prompt = (
        f"{spec.body}\n\n(A) {options['A']}\n(B) {options['B']}\n(C) {options['C']}\n\n{CONTRACT}"
    )
    assistant = json.dumps(
        {"answer": initial, "confidence": 95, "reason": spec.initial_reason},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    evidence = spec.evidence
    v_followup = (
        f"Observation:\n{evidence}\n\n"
        f"Using the original rules and all unchanged information, recompute the answer. {CONTRACT}"
    )
    original_status = {
        label: "supported" if label == initial else "contradicted" for label in ("A", "B", "C")
    }
    updated_status = {
        label: "supported" if label == target else "contradicted" for label in ("A", "B", "C")
    }
    return {
        "dataset_version": VERSION,
        "base_id": base_id,
        "family_id": family_id,
        "variant": spec.variant,
        "stratum": STRATUM[spec.family],
        "initial_answer": initial,
        "target_answer": target,
        "distractor_answer": distractor,
        "answer_migration": f"{initial}_to_{target}",
        "options": options,
        "original_gold_rationale": spec.original_rationale,
        "updated_gold_rationale": spec.updated_rationale,
        "messages_common": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "v_new_evidence": evidence,
        "v_followup": v_followup,
        "construction_certificate": {
            "construction_method": "solution_first_closed_world_minimal_update",
            "difficulty_band": "easy" if spec.variant == 1 else "medium",
            "original_derivation_steps": list(spec.original_steps),
            "updated_derivation_steps": list(spec.updated_steps),
            "original_option_status": original_status,
            "updated_option_status": updated_status,
            "v_changed_node": spec.changed_node,
            "v_changed_node_count": 1,
            "external_knowledge_required": False,
            "distractor_wrong_before": True,
            "distractor_wrong_after": True,
        },
        "new_structure_attestation": True,
    }


def main() -> None:
    if len(SPECS) != 44 or len({(spec.family, spec.variant) for spec in SPECS}) != 44:
        raise RuntimeError("authoring specs must cover exactly 44 distinct non-sentinels")
    sentinels = [
        json.loads(line)
        for line in (HERE / "phase2_sentinel4_draft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for sentinel in sentinels:
        base_id = str(sentinel["base_id"])
        expected = SENTINEL_HASHES.get(base_id)
        if expected is None or canonical_sha256(sentinel) != expected:
            raise RuntimeError(f"approved sentinel content changed: {base_id}")
    occupied = {(int(row["family_id"][-2:]), int(row["variant"])) for row in sentinels}
    if occupied & {(spec.family, spec.variant) for spec in SPECS}:
        raise RuntimeError("authoring specs overlap approved sentinels")
    bases = sentinels + [make_base(spec) for spec in SPECS]
    bases.sort(key=lambda row: str(row["base_id"]))
    if [row["base_id"] for row in bases] != [f"C1V7-{i:03d}" for i in range(1, 49)]:
        raise RuntimeError("full base IDs are incomplete")
    output = HERE / "m2_2b_v7_heldout48_base.jsonl"
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in bases
        ),
        encoding="utf-8",
        newline="\n",
    )
    for sentinel in sentinels:
        base_id = str(sentinel["base_id"])
        rebuilt = next(row for row in bases if row["base_id"] == base_id)
        if canonical_sha256(rebuilt) != SENTINEL_HASHES[base_id]:
            raise RuntimeError(f"sentinel changed during assembly: {base_id}")
    print(f"WROTE {output} ({len(bases)} bases; 4 sentinels unchanged)")


if __name__ == "__main__":
    main()
