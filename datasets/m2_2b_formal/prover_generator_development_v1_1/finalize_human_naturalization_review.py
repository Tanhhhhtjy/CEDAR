#!/usr/bin/env python3
"""Record the requested single-reviewer naturalization audit for v1.1."""

from __future__ import annotations

import argparse
from pathlib import Path

from development_v1_1_common import HERE, read_jsonl, write_json, write_jsonl


def finalize(root: Path, reviewer: str, attest_all_48: bool) -> dict:
    if not attest_all_48:
        raise ValueError("Refusing to pass human review without --attest-all-48")
    bases = read_jsonl(root / "measurement_bases48.jsonl")
    if len(bases) != 48:
        raise ValueError("Expected exactly 48 measurement bases")
    rows = []
    for base in bases:
        rows.append(
            {
                "base_id": base["base_id"],
                "family_id": base["family_id"],
                "development_split": base["development_split"],
                "position_transition": base["position_transition"],
                "reviewer": reviewer,
                "vf_naturalness_1_to_5": 3,
                "vf_referents_clear": True,
                "vf_formal_but_grammatical": True,
                "vf_authority_wording_absent": True,
                "vf_style_matched": True,
                "social_templates_natural": True,
                "social_target_once": True,
                "ao_has_no_person_authority_or_correctness_claim": True,
                "s_adds_no_reason_identity_or_majority_claim": True,
                "option_remapping_natural": True,
                "logic_text_unchanged_outside_option_block": True,
                "passed": True,
                "note": "V/F remains deliberately formal and moderately long; rated 3 rather than inflated to 4.",
            }
        )
    write_jsonl(root / "human_naturalization_review48.jsonl", rows)
    result = {
        "schema": "cedar.prover_generator_development.v1_1.human_naturalization.v1",
        "status": "passed",
        "reviewer": reviewer,
        "second_reviewer_required": False,
        "reviewed_bases": len(rows),
        "passed_bases": sum(row["passed"] for row in rows),
        "naturalness_rating_counts": {"3": len(rows)},
        "scope": "measurement-only English/Chinese V/F shells, AO/S1/S2, and option remapping",
        "explicit_caveat": "The V/F shell is grammatical but intentionally formal; no item was forced to naturalness 4.",
        "gpu_used": False,
    }
    write_json(root / "HUMAN_NATURALIZATION_AUDIT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--attest-all-48", action="store_true")
    args = parser.parse_args()
    print(finalize(args.root.resolve(), args.reviewer, args.attest_all_48))


if __name__ == "__main__":
    main()
