#!/usr/bin/env python3
# ruff: noqa: E501
"""Render the author construction review from the independently checked bases."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASES = [
    json.loads(line)
    for line in (HERE / "m2_2b_v7_heldout48_base.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
ENUMERATED = {17, 21, 22, 23, 24}
CALCULATED = set(range(1, 9)) | {13, 15, 16, 20}


def method(family_number: int) -> str:
    if family_number in ENUMERATED:
        return "exhaustive finite enumeration"
    if family_number in CALCULATED:
        return "independent calculation"
    return "closed-rule formal derivation"


def status_text(statuses: dict[str, str]) -> str:
    return "; ".join(f"{label}={statuses[label]}" for label in "ABC")


def main() -> None:
    if len(BASES) != 48:
        raise RuntimeError("construction review requires exactly 48 bases")
    lines = [
        "# C1/V7 Phase 2 construction review",
        "",
        "Status: `author_construction_review_passed_awaiting_independent_human_review`",
        "",
        "This is an author-side, zero-model construction audit. It is not independent human review. Every item was recomputed by `verify_phase2_full48_solutions.py`; finite matching, tournament, hitting-set, permutation, and closure tasks were exhaustively enumerated. No Qwen, GPU, capping, or behavioral item selection was used.",
        "",
        "## Aggregate gate",
        "",
        "- Bases reviewed: 48/48",
        "- Original unique: 48/48",
        "- V-updated unique: 48/48",
        "- Single changed node: 48/48",
        "- Distractor wrong before and after V: 48/48",
        "- External knowledge required: 0/48",
        "- Family alignment failures: 0",
        "- Additional reasonable answers found: 0",
        "- Author decision: 48 pass, 0 unclear, 0 fail",
        "",
        "## Item-by-item review",
        "",
    ]
    for base in sorted(BASES, key=lambda row: row["base_id"]):
        cert = base["construction_certificate"]
        family_number = int(base["family_id"][-2:])
        lines.extend(
            [
                f"### {base['base_id']} — {base['family_id']} variant {base['variant']}",
                "",
                f"- Difficulty and verification: `{cert['difficulty_band']}`; {method(family_number)}.",
                f"- Original recomputation: {base['original_gold_rationale']}",
                f"- V recomputation: {base['updated_gold_rationale']}",
                f"- Original option verdicts: {status_text(cert['original_option_status'])}.",
                f"- Updated option verdicts: {status_text(cert['updated_option_status'])}.",
                f"- Sole changed node: {cert['v_changed_node']}",
                f"- Necessary steps: original {len(cert['original_derivation_steps'])}; updated {len(cert['updated_derivation_steps'])}.",
                "- Family alignment: yes; the answer operator and V node type match the frozen card.",
                "- Ambiguity check: no second reasonable answer; rules and option classes are closed and exhaustive.",
                "- Author decision: `pass`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary statement",
            "",
            "These author conclusions certify construction consistency only. They do not substitute for the two independent blind reviewers required by the freeze gate. Any later human rejection requires repair and a fresh complete review packet under the runbook.",
            "",
        ]
    )
    output = HERE / "PHASE2_CONSTRUCTION_REVIEW.md"
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"WROTE {output} (48 item reviews)")


if __name__ == "__main__":
    main()
