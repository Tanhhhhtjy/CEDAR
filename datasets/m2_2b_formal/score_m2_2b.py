"""Score M2.2b result JSONL with item-clustered five-arm metrics.

Expected result rows:
  {"id":"M22B-AR-001-V", "cap":"uncapped", "seed":0,
   "response":"{\"answer\":\"B\",...}"}

An already parsed top-level ``answer`` field may be supplied instead of ``response``.
Repeated seeds are averaged within base unit before base units are averaged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
GOLD_PATH = HERE / "m2_2b_pilot300_arms.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_answer(row: dict) -> str | None:
    direct = row.get("answer")
    if isinstance(direct, str) and direct.strip().upper() in {"A", "B", "C"}:
        return direct.strip().upper()
    response = row.get("response")
    if isinstance(response, dict):
        candidate = response.get("answer")
        if isinstance(candidate, str) and candidate.strip().upper() in {"A", "B", "C"}:
            return candidate.strip().upper()
        return None
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


def diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gold = {x["id"]: x for x in load_jsonl(args.gold)}
    results = load_jsonl(args.results)
    observations: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    unknown_ids: list[str] = []
    for row in results:
        rid = row.get("id")
        if rid not in gold:
            unknown_ids.append(str(rid))
            continue
        g = gold[rid]
        cap = str(row.get("cap", "uncapped"))
        answer = parse_answer(row)
        observations[(cap, g["arm"], g["base_id"])].append({
            "valid": answer is not None,
            "update": None if answer is None else answer != g["initial_answer_id"],
            "correct": None if answer is None else answer == g["gold_final_answer_id"],
            "target": None if answer is None or g["induced_target_answer_id"] is None else answer == g["induced_target_answer_id"],
        })

    unit_scores: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (cap, arm, base_id), rows in observations.items():
        valid_rows = [x for x in rows if x["valid"]]
        unit_scores[(cap, arm)].append({
            "base_id": base_id,
            "invalid_rate": 1 - len(valid_rows) / len(rows),
            "update_rate": mean([float(x["update"]) for x in valid_rows]),
            "accuracy": mean([float(x["correct"]) for x in valid_rows]),
            "target_acceptance": mean([float(x["target"]) for x in valid_rows if x["target"] is not None]),
        })

    caps = sorted({k[0] for k in unit_scores})
    by_cap: dict[str, dict] = {}
    for cap in caps:
        arm_metrics: dict[str, dict] = {}
        for arm in ("N", "R", "V", "F", "S"):
            rows = unit_scores.get((cap, arm), [])
            arm_metrics[arm] = {
                "base_units": len(rows),
                "invalid_rate": rounded(mean([x["invalid_rate"] for x in rows])),
                "U_update": rounded(mean([x["update_rate"] for x in rows if x["update_rate"] is not None])),
                "accuracy": rounded(mean([x["accuracy"] for x in rows if x["accuracy"] is not None])),
                "induced_target_acceptance": rounded(mean([x["target_acceptance"] for x in rows if x["target_acceptance"] is not None])),
            }
        u = {arm: arm_metrics[arm]["U_update"] for arm in arm_metrics}
        weg = diff(u["V"], u["R"])
        fh = diff(u["F"], u["R"])
        sh = diff(u["S"], u["R"])
        by_cap[cap] = {
            "arms": arm_metrics,
            "WEG_V_minus_R": rounded(weg),
            "FH_F_minus_R": rounded(fh),
            "SH_S_minus_R": rounded(sh),
            "ND_U_N": rounded(u["N"]),
            "RE_R_minus_N": rounded(diff(u["R"], u["N"])),
        }

    uncapped = by_cap.get("uncapped")
    if uncapped is not None:
        for cap, metrics in by_cap.items():
            if cap == "uncapped":
                metrics.update({"Cost_V": 0.0, "Benefit_F": 0.0, "Benefit_S": 0.0})
                continue
            metrics["Cost_V"] = rounded(diff(uncapped["WEG_V_minus_R"], metrics["WEG_V_minus_R"]))
            metrics["Benefit_F"] = rounded(diff(uncapped["FH_F_minus_R"], metrics["FH_F_minus_R"]))
            metrics["Benefit_S"] = rounded(diff(uncapped["SH_S_minus_R"], metrics["SH_S_minus_R"]))
            metrics["fallacy_selectivity_BenefitF_minus_CostV"] = rounded(
                diff(metrics["Benefit_F"], metrics["Cost_V"])
            )

    report = {
        "results_path": str(args.results),
        "gold_path": str(args.gold),
        "result_rows": len(results),
        "unknown_ids": unknown_ids,
        "note": "Rates average seeds within base unit, then average base units. Invalid parses are reported and excluded pending blind adjudication.",
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
