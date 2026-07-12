"""Score uncapped Qwen outputs for the v3 paradigm calibration set.

Expected rows contain ``id`` plus either ``answer`` or ``response``. Repeated
seeds are averaged within each base unit before base units are averaged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
GOLD = HERE / "m2_2b_v3_calibration72_arms.jsonl"
ARM_ORDER = ("N", "R", "V", "F", "S", "AO")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_answer(row: dict) -> str | None:
    direct = row.get("answer")
    if isinstance(direct, str) and direct.strip().upper() in {"A", "B", "C"}:
        return direct.strip().upper()
    response = row.get("response")
    if isinstance(response, dict):
        value = response.get("answer")
        return value.strip().upper() if isinstance(value, str) and value.strip().upper() in {"A", "B", "C"} else None
    if not isinstance(response, str):
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        value = parsed.get("answer")
        if isinstance(value, str) and value.strip().upper() in {"A", "B", "C"}:
            return value.strip().upper()
    hits = re.findall(r"(?<![A-Za-z])([ABC])(?![A-Za-z])", response.upper())
    return hits[0] if len(set(hits)) == 1 else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gold = {x["id"]: x for x in load_jsonl(args.gold)}
    results = load_jsonl(args.results)
    unknown = [str(x.get("id")) for x in results if x.get("id") not in gold]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results:
        if row.get("id") not in gold:
            continue
        g = gold[row["id"]]
        answer = parse_answer(row)
        grouped[(g["base_id"], g["arm"])].append({
            "valid": answer is not None,
            "update": None if answer is None else answer != g["initial_answer_id"],
            "correct": None if answer is None else answer == g["gold_final_answer_id"],
            "target": None if answer is None or g["induced_target_answer_id"] is None else answer == g["induced_target_answer_id"],
        })

    arm_units: dict[str, list[dict]] = defaultdict(list)
    for (base_id, arm), rows in grouped.items():
        valid = [x for x in rows if x["valid"]]
        arm_units[arm].append({
            "base_id": base_id,
            "invalid_rate": 1 - len(valid) / len(rows),
            "update": mean([float(x["update"]) for x in valid]),
            "correct": mean([float(x["correct"]) for x in valid]),
            "target": mean([float(x["target"]) for x in valid if x["target"] is not None]),
        })

    metrics = {}
    for arm in ARM_ORDER:
        rows = arm_units.get(arm, [])
        metrics[arm] = {
            "base_units": len(rows),
            "invalid_rate": r(mean([x["invalid_rate"] for x in rows])),
            "update_rate": r(mean([x["update"] for x in rows if x["update"] is not None])),
            "accuracy": r(mean([x["correct"] for x in rows if x["correct"] is not None])),
            "induced_target_acceptance": r(mean([x["target"] for x in rows if x["target"] is not None])),
        }

    u = {arm: metrics[arm]["update_rate"] for arm in ARM_ORDER}
    def diff(a: str, b: str) -> float | None:
        return None if u[a] is None or u[b] is None else r(u[a] - u[b])

    contrasts = {
        "WEG_V_minus_R": diff("V", "R"),
        "FH_F_minus_R": diff("F", "R"),
        "SH_S_minus_R": diff("S", "R"),
        "argument_content_F_minus_AO": diff("F", "AO"),
        "validity_separation_V_minus_F": diff("V", "F"),
        "validity_vs_social_V_minus_S": diff("V", "S"),
    }

    # Screening targets respect the coarse 1/12 granularity of this small set.
    checks = {
        "N_update_le_0.20": u["N"] is not None and u["N"] <= 0.20,
        "R_update_le_0.20": u["R"] is not None and u["R"] <= 0.20,
        "V_update_ge_0.67": u["V"] is not None and u["V"] >= 0.67,
        "F_update_between_0.17_and_0.83": u["F"] is not None and 0.17 <= u["F"] <= 0.83,
        "S_update_between_0.17_and_0.83": u["S"] is not None and 0.17 <= u["S"] <= 0.83,
        "AO_update_le_0.25": u["AO"] is not None and u["AO"] <= 0.25,
        "V_minus_F_ge_0.17": contrasts["validity_separation_V_minus_F"] is not None and contrasts["validity_separation_V_minus_F"] >= 0.17,
        "F_minus_AO_ge_0.17": contrasts["argument_content_F_minus_AO"] is not None and contrasts["argument_content_F_minus_AO"] >= 0.17,
        "V_minus_S_ge_0.17": contrasts["validity_vs_social_V_minus_S"] is not None and contrasts["validity_vs_social_V_minus_S"] >= 0.17,
    }

    complete = all(metrics[a]["base_units"] == 12 for a in ARM_ORDER)
    report = {
        "results_path": str(args.results),
        "result_rows": len(results),
        "unknown_ids": unknown,
        "complete_12x6": complete,
        "metrics": metrics,
        "contrasts": contrasts,
        "screening_checks": checks,
        "screening_pass_all": complete and not unknown and all(checks.values()),
        "interpretation": (
            "These are development screening targets, not preregistered scientific "
            "success criteria. Do not run capping or select individual test items from "
            "these outputs."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if unknown or not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
