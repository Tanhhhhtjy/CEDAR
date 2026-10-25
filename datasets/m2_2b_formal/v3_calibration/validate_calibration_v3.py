"""Validate structure and prompt-role separation for v3 calibration data."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE / "m2_2b_v3_calibration12_base.jsonl"
FLAT = HERE / "m2_2b_v3_calibration72_arms.jsonl"
MANIFEST = HERE / "m2_2b_v3_calibration_manifest.json"
AUDIT = HERE / "m2_2b_v3_construction_audit12.jsonl"
ARMS = {"N", "R", "V", "F", "S", "AO"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    base = load_jsonl(BASE)
    flat = load_jsonl(FLAT)
    audit = load_jsonl(AUDIT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if len(base) != 12:
        errors.append(f"expected 12 base units, found {len(base)}")
    if len(flat) != 72:
        errors.append(f"expected 72 arm rows, found {len(flat)}")
    if len(audit) != 12:
        errors.append(f"expected 12 construction audit rows, found {len(audit)}")
    if len({x["base_id"] for x in base}) != len(base):
        errors.append("duplicate base_id")
    if len({x["id"] for x in flat}) != len(flat):
        errors.append("duplicate flat id")
    if len({x["template_family"] for x in base}) != 12:
        errors.append("calibration items must use 12 distinct structural families")
    if {x["base_id"] for x in audit} != {x["base_id"] for x in base}:
        errors.append("construction audit base ids do not match dataset")
    for row in audit:
        if row.get("review_type") != "author_construction_audit":
            errors.append(f"{row.get('base_id')}: incorrect audit review_type")
        if row.get("independent_human_review") is not False:
            errors.append(f"{row.get('base_id')}: audit must not claim independent human review")
        if not row.get("V_rationale") or not row.get("F_fatal_flaw"):
            errors.append(f"{row.get('base_id')}: missing V rationale or F fatal flaw")

    strata = Counter(x["stratum"] for x in base)
    subtypes = Counter(x["matched_valid_subtype"] for x in base)
    difficulty = Counter(x["fallacy_difficulty_intended"] for x in base)
    if set(strata.values()) != {4} or len(strata) != 3:
        errors.append(f"stratum imbalance: {dict(strata)}")
    if subtypes != {"refute": 6, "undercut": 6}:
        errors.append(f"subtype imbalance: {dict(subtypes)}")
    if difficulty != {"easy": 6, "medium": 6}:
        errors.append(f"difficulty imbalance: {dict(difficulty)}")

    rows_by_base: dict[str, list[dict]] = defaultdict(list)
    for row in flat:
        rows_by_base[row["base_id"]].append(row)
    for item in base:
        base_id = item["base_id"]
        rows = rows_by_base[base_id]
        if {x["arm"] for x in rows} != ARMS:
            errors.append(f"{base_id}: missing or duplicate arms")
        initial = item["initial_answer_id"]
        updated = item["updated_answer_id"]
        for arm in ("N", "R", "F", "S", "AO"):
            if item["arms"][arm]["gold_final_answer_id"] != initial:
                errors.append(f"{base_id}-{arm}: gold must maintain initial answer")
        if item["arms"]["V"]["gold_final_answer_id"] != updated:
            errors.append(f"{base_id}-V: gold must use updated answer")
        for arm in ("V", "F"):
            text = item["arms"][arm]["followup_user"]
            for required in ("Observation:\n", "Candidate reasoning:\n"):
                if required not in text:
                    errors.append(f"{base_id}-{arm}: missing {required.strip()}")
            for forbidden in (
                "which may be wrong",
                "Review note",
                "The same checklist records",
                "Analysis:",
            ):
                if forbidden.lower() in text.lower():
                    errors.append(f"{base_id}-{arm}: forbidden packaging cue {forbidden!r}")
        v_words = len(item["arms"]["V"]["followup_user"].split())
        f_words = len(item["arms"]["F"]["followup_user"].split())
        ratio = max(v_words, f_words) / min(v_words, f_words)
        if ratio > 1.20:
            errors.append(f"{base_id}: V/F word-count ratio {ratio:.3f} > 1.20")

    for filename, expected in manifest["sha256"].items():
        actual = sha256(HERE / filename)
        if actual != expected:
            errors.append(f"hash mismatch: {filename}")

    report = {
        "base_units": len(base),
        "arm_records": len(flat),
        "arms": sorted(ARMS),
        "structural_families": len({x["template_family"] for x in base}),
        "strata": dict(sorted(strata.items())),
        "subtypes": dict(sorted(subtypes.items())),
        "difficulty": dict(sorted(difficulty.items())),
        "max_vf_word_ratio": round(max(
            max(len(x["arms"]["V"]["followup_user"].split()), len(x["arms"]["F"]["followup_user"].split()))
            / min(len(x["arms"]["V"]["followup_user"].split()), len(x["arms"]["F"]["followup_user"].split()))
            for x in base
        ), 4),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
