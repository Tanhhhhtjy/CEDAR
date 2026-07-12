"""Validate the v3 social-pressure strength calibration set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "m2_2b_v3_s_calibration36.jsonl"
AUDIT = HERE / "m2_2b_v3_s_design_audit12.jsonl"
MANIFEST = HERE / "m2_2b_v3_s_calibration_manifest.json"
SOURCE = HERE.parent / "v3_calibration" / "m2_2b_v3_calibration12_base.jsonl"
STRENGTHS = {"S1_tentative", "S2_confident_preference", "S3_coercive"}


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
    rows = load_jsonl(DATA)
    audit = load_jsonl(AUDIT)
    source = {x["base_id"]: x for x in load_jsonl(SOURCE)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if len(rows) != 36:
        errors.append(f"expected 36 rows, found {len(rows)}")
    if len(audit) != 12:
        errors.append(f"expected 12 audit rows, found {len(audit)}")
    if len({x["id"] for x in rows}) != len(rows):
        errors.append("duplicate id")
    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)
    if set(by_base) != set(source):
        errors.append("base ids do not match source calibration set")
    for base_id, group in by_base.items():
        if {x["social_pressure_strength"] for x in group} != STRENGTHS:
            errors.append(f"{base_id}: strength set mismatch")
        src = source[base_id]
        for row in group:
            if row["system"] != src["system"] or row["initial_user"] != src["initial_user"] or row["initial_assistant"] != src["initial_assistant"]:
                errors.append(f"{row['id']}: shared transcript differs from source")
            if row["gold_final_answer_id"] != row["initial_answer_id"]:
                errors.append(f"{row['id']}: social pressure must maintain initial gold")
            if row["induced_target_answer_id"] != row["updated_answer_id"]:
                errors.append(f"{row['id']}: target must match paired updated answer")
            if "No new task information is supplied." not in row["followup_user"]:
                errors.append(f"{row['id']}: missing no-new-information statement")
        strong = next(x for x in group if x["social_pressure_strength"] == "S3_coercive")
        if strong["followup_user"] != src["arms"]["S"]["followup_user"]:
            errors.append(f"{base_id}: S3 must exactly replicate prior S")

    if {x["base_id"] for x in audit} != set(source):
        errors.append("audit ids do not match source")
    for row in audit:
        if row.get("independent_human_review") is not False:
            errors.append(f"{row.get('base_id')}: design audit must not claim independent review")

    counts = Counter(x["social_pressure_strength"] for x in rows)
    if any(counts[x] != 12 for x in STRENGTHS):
        errors.append(f"strength imbalance: {dict(counts)}")
    for filename, expected in manifest["sha256"].items():
        if sha256(HERE / filename) != expected:
            errors.append(f"hash mismatch: {filename}")

    report = {
        "rows": len(rows),
        "base_units": len(by_base),
        "records_per_strength": dict(sorted(counts.items())),
        "strata": dict(sorted(Counter(x["stratum"] for x in source.values()).items())),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()

