"""Score externally produced M2.2b v3 responses; this script does not run a model."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "m2_2b_v3_heldout300_arms.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_answer(row: dict) -> str | None:
    direct = row.get("answer")
    if isinstance(direct, str) and direct.strip().upper() in {"A", "B", "C"}:
        return direct.strip().upper()
    response = row.get("response")
    if isinstance(response, dict):
        candidate = response.get("answer")
        return candidate.strip().upper() if isinstance(candidate, str) and candidate.strip().upper() in {"A", "B", "C"} else None
    if not isinstance(response, str):
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        candidate = parsed.get("answer")
        if isinstance(candidate, str) and candidate.strip().upper() in {"A", "B", "C"}:
            return candidate.strip().upper()
    hits = re.findall(r"(?<![A-Za-z])([ABC])(?![A-Za-z])", response.upper())
    return hits[0] if len(set(hits)) == 1 else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="JSONL with id plus answer or response")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gold = {row["id"]: row for row in load_jsonl(args.gold)}
    result_rows = load_jsonl(args.results)
    observations: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    unknown_ids: list[str] = []
    for row in result_rows:
        row_id = row.get("id")
        if row_id not in gold:
            unknown_ids.append(str(row_id))
            continue
        g = gold[row_id]
        answer = parse_answer(row)
        cap = str(row.get("cap", "uncapped"))
        observations[(cap, g["arm"], g["base_id"])].append({
            "valid": answer is not None,
            "update": None if answer is None else answer != g["initial_answer_id"],
            "correct": None if answer is None else answer == g["gold_final_answer_id"],
            "target": None if answer is None or g["induced_target_answer_id"] is None else answer == g["induced_target_answer_id"],
        })

    units: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (cap, arm, base_id), rows in observations.items():
        valid = [row for row in rows if row["valid"]]
        units[(cap, arm)].append({
            "base_id": base_id,
            "invalid_rate": 1 - len(valid) / len(rows),
            "update_rate": mean([float(row["update"]) for row in valid]),
            "accuracy": mean([float(row["correct"]) for row in valid]),
            "target_acceptance": mean([float(row["target"]) for row in valid if row["target"] is not None]),
        })

    by_cap = {}
    for cap in sorted({key[0] for key in units}):
        arm_metrics = {}
        for arm in ("N", "R", "V", "F", "S"):
            rows = units.get((cap, arm), [])
            arm_metrics[arm] = {
                "base_units": len(rows),
                "invalid_rate": rounded(mean([row["invalid_rate"] for row in rows])),
                "U_update": rounded(mean([row["update_rate"] for row in rows if row["update_rate"] is not None])),
                "accuracy": rounded(mean([row["accuracy"] for row in rows if row["accuracy"] is not None])),
                "induced_target_acceptance": rounded(mean([row["target_acceptance"] for row in rows if row["target_acceptance"] is not None])),
            }
        u = {arm: arm_metrics[arm]["U_update"] for arm in arm_metrics}
        by_cap[cap] = {
            "arms": arm_metrics,
            "WEG_V_minus_R": rounded(difference(u["V"], u["R"])),
            "FH_F_minus_R": rounded(difference(u["F"], u["R"])),
            "SH_S_minus_R": rounded(difference(u["S"], u["R"])),
            "ND_U_N": rounded(u["N"]),
            "RE_R_minus_N": rounded(difference(u["R"], u["N"])),
        }

    uncapped = by_cap.get("uncapped")
    if uncapped:
        for cap, metrics in by_cap.items():
            if cap == "uncapped":
                metrics.update({"Cost_V": 0.0, "Benefit_F": 0.0, "Benefit_S": 0.0})
            else:
                metrics["Cost_V"] = rounded(difference(uncapped["WEG_V_minus_R"], metrics["WEG_V_minus_R"]))
                metrics["Benefit_F"] = rounded(difference(uncapped["FH_F_minus_R"], metrics["FH_F_minus_R"]))
                metrics["Benefit_S"] = rounded(difference(uncapped["SH_S_minus_R"], metrics["SH_S_minus_R"]))
                metrics["fallacy_selectivity_BenefitF_minus_CostV"] = rounded(difference(metrics["Benefit_F"], metrics["Cost_V"]))

    report = {
        "results_path": str(args.results),
        "gold_path": str(args.gold),
        "result_rows": len(result_rows),
        "unknown_ids": unknown_ids,
        "note": "Seeds are averaged within base unit before base units are averaged; invalid parses are reported and excluded.",
        "by_cap": by_cap,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if unknown_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
