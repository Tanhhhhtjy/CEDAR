"""Static validation for the M2.2b five-arm pilot dataset."""

from __future__ import annotations

import json
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "m2_2b_pilot60_base.jsonl"
ARM_PATH = HERE / "m2_2b_pilot300_arms.jsonl"
MANIFEST_PATH = HERE / "m2_2b_manifest.json"
ARMS = {"N", "R", "V", "F", "S"}


def load(path: Path) -> list[dict]:
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
    base = load(BASE_PATH)
    flat = load(ARM_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in flat:
        by_base[row["base_id"]].append(row)

    if len(base) != 60:
        errors.append(f"expected 60 base units, got {len(base)}")
    if len(flat) != 300:
        errors.append(f"expected 300 arm records, got {len(flat)}")
    if len(by_base) != 60:
        errors.append(f"expected 60 flattened families, got {len(by_base)}")
    if manifest.get("base_units") != len(base) or manifest.get("arm_records") != len(flat):
        errors.append("manifest counts do not match generated files")
    for name, expected_hash in manifest.get("sha256", {}).items():
        actual_hash = sha256(HERE / name)
        if actual_hash != expected_hash:
            errors.append(f"manifest hash mismatch for {name}")

    strata = Counter(x["stratum"] for x in base)
    if strata != Counter({"arithmetic_accounting": 20, "unit_label": 20, "multistep_logic": 20}):
        errors.append(f"stratum imbalance: {dict(strata)}")

    subtype = Counter(x["matched_valid_subtype"] for x in base)
    if subtype != Counter({"refute": 30, "undercut": 30}):
        errors.append(f"subtype imbalance: {dict(subtype)}")
    for stratum in strata:
        local = Counter(x["matched_valid_subtype"] for x in base if x["stratum"] == stratum)
        if local != Counter({"refute": 10, "undercut": 10}):
            errors.append(f"subtype imbalance in {stratum}: {dict(local)}")

    initial_positions = Counter(x["initial_answer_id"] for x in base)
    updated_positions = Counter(x["updated_answer_id"] for x in base)
    if initial_positions != Counter({"A": 20, "B": 20, "C": 20}):
        errors.append(f"initial answer positions not balanced: {dict(initial_positions)}")
    if updated_positions != Counter({"A": 20, "B": 20, "C": 20}):
        errors.append(f"updated answer positions not balanced: {dict(updated_positions)}")

    for item in base:
        bid = item["base_id"]
        if set(item["arms"]) != ARMS:
            errors.append(f"{bid}: missing arms {ARMS - set(item['arms'])}")
        if item["initial_answer_id"] == item["updated_answer_id"]:
            errors.append(f"{bid}: initial and updated labels are identical")
        if item["vf_length_ratio"] > 1.15:
            errors.append(
                f"{bid}: V/F word-count ratio {item['vf_length_ratio']} exceeds 1.15 "
                f"({item['vf_word_counts']})"
            )
        try:
            initial_json = json.loads(item["initial_assistant"])
        except json.JSONDecodeError as exc:
            errors.append(f"{bid}: invalid initial_assistant JSON: {exc}")
        else:
            if initial_json.get("answer") != item["initial_answer_id"]:
                errors.append(f"{bid}: initial prefill answer mismatch")

        rows = by_base.get(bid, [])
        if {x["arm"] for x in rows} != ARMS:
            errors.append(f"{bid}: flattened family does not contain five arms")
            continue
        shared_fields = ("system", "initial_user", "initial_assistant", "options", "initial_answer_id")
        for field in shared_fields:
            values = {json.dumps(x[field], sort_keys=True) for x in rows}
            if len(values) != 1:
                errors.append(f"{bid}: {field} differs across arms")
        for row in rows:
            expected = item["updated_answer_id"] if row["arm"] == "V" else item["initial_answer_id"]
            if row["gold_final_answer_id"] != expected:
                errors.append(f"{row['id']}: wrong final gold label")
            if row["normative_should_update"] != (row["arm"] == "V"):
                errors.append(f"{row['id']}: wrong normative_should_update")
            if row["arm"] != "V" and row["valid_subtype"] is not None:
                errors.append(f"{row['id']}: valid_subtype leaked outside V")
            if row["arm"] != "F" and row["fallacy_type"] is not None:
                errors.append(f"{row['id']}: fallacy_type leaked outside F")

    duplicate_prompts = [p for p, n in Counter(x["initial_user"] for x in base).items() if n > 1]
    if duplicate_prompts:
        errors.append(f"duplicate initial prompts: {len(duplicate_prompts)}")

    summary = {
        "base_units": len(base),
        "arm_records": len(flat),
        "strata": dict(strata),
        "subtypes": dict(subtype),
        "initial_answer_positions": dict(initial_positions),
        "updated_answer_positions": dict(updated_positions),
        "max_vf_length_ratio": max(x["vf_length_ratio"] for x in base),
        "fallacy_types": dict(Counter(x["arms"]["F"]["fallacy_type"] for x in base)),
        "dual_human_validity_review": dict(Counter(x["dual_human_validity_review"] for x in base)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        print("\nVALIDATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
