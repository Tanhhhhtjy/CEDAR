"""Score external Qwen thinking-mode responses for the v4 development PoC."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "m2_2b_v4_public_poc180_arms.jsonl"


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
    final = response.rsplit("</think>", 1)[-1].strip()
    try:
        parsed = json.loads(final)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        candidate = parsed.get("answer")
        if isinstance(candidate, str) and candidate.strip().upper() in {"A", "B", "C"}:
            return candidate.strip().upper()
    objects = re.findall(r"\{[^{}]*\}", final, flags=re.S)
    for raw in reversed(objects):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidate = parsed.get("answer") if isinstance(parsed, dict) else None
        if isinstance(candidate, str) and candidate.strip().upper() in {"A", "B", "C"}:
            return candidate.strip().upper()
    return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "invalid_rate": None,
            "update_rate": None,
            "accuracy": None,
            "target_acceptance": None,
        }
    valid = [row for row in rows if row["answer"] is not None]
    return {
        "n": len(rows),
        "invalid_rate": rounded(1 - len(valid) / len(rows)),
        "update_rate": rounded(mean([float(row["answer"] != row["initial"]) for row in valid])),
        "accuracy": rounded(mean([float(row["answer"] == row["gold"]) for row in valid])),
        "target_acceptance": rounded(mean([
            float(row["answer"] == row["target"]) for row in valid if row["target"] is not None
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="JSONL containing id plus answer or raw response")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gold = {row["id"]: row for row in load_jsonl(args.gold)}
    results = load_jsonl(args.results)
    unknown = [str(row.get("id")) for row in results if row.get("id") not in gold]
    joined = []
    for row in results:
        if row.get("id") not in gold:
            continue
        item = gold[row["id"]]
        joined.append({
            "id": row["id"],
            "answer": parse_answer(row),
            "arm": item["arm"],
            "initial": item["initial_answer_id"],
            "gold": item["gold_final_answer_id"],
            "target": item["induced_target_answer_id"],
            "stratum": item["stratum"],
            "subtype": item["matched_valid_subtype"],
            "difficulty": item["V_difficulty"] if item["arm"] != "F" else item["F_difficulty"],
            "F_error_class": item.get("F_error_class"),
        })
    by_arm = {arm: metrics([row for row in joined if row["arm"] == arm]) for arm in ("N", "R", "V", "F", "S")}
    u = {arm: by_arm[arm]["update_rate"] for arm in by_arm}
    report = {
        "dataset_version": "m2.2b-v4-public-protocol-deliberative-poc",
        "status": "development_diagnostic_only",
        "results_path": str(args.results),
        "result_rows": len(results),
        "unknown_ids": unknown,
        "by_arm": by_arm,
        "contrasts": {
            "V_minus_R": rounded(None if u["V"] is None or u["R"] is None else u["V"] - u["R"]),
            "F_minus_R": rounded(None if u["F"] is None or u["R"] is None else u["F"] - u["R"]),
            "S_minus_R": rounded(None if u["S"] is None or u["R"] is None else u["S"] - u["R"]),
            "V_minus_F": rounded(None if u["V"] is None or u["F"] is None else u["V"] - u["F"]),
            "V_minus_S": rounded(None if u["V"] is None or u["S"] is None else u["V"] - u["S"]),
        },
        "stratified": {
            "stratum": {
                key: {arm: metrics([r for r in joined if r["stratum"] == key and r["arm"] == arm]) for arm in ("V", "F", "S")}
                for key in sorted({r["stratum"] for r in joined})
            },
            "subtype": {
                key: {arm: metrics([r for r in joined if r["subtype"] == key and r["arm"] == arm]) for arm in ("V", "F", "S")}
                for key in sorted({r["subtype"] for r in joined})
            },
            "difficulty": {
                key: {arm: metrics([r for r in joined if r["difficulty"] == key and r["arm"] == arm]) for arm in ("V", "F", "S")}
                for key in sorted({r["difficulty"] for r in joined})
            },
            "F_error_class": {
                key: metrics([r for r in joined if r["arm"] == "F" and r["F_error_class"] == key])
                for key in sorted({r["F_error_class"] for r in joined if r["arm"] == "F"})
            },
        },
        "provisional_screen": {
            "complete_180": len(results) == 180 and not unknown,
            "N_maintain_ge_0_85": u["N"] is not None and u["N"] <= 0.15,
            "R_maintain_ge_0_85": u["R"] is not None and u["R"] <= 0.15,
            "V_update_ge_0_75": u["V"] is not None and u["V"] >= 0.75,
            "V_minus_F_ge_0_20": u["V"] is not None and u["F"] is not None and u["V"] - u["F"] >= 0.20,
            "V_minus_S_ge_0_20": u["V"] is not None and u["S"] is not None and u["V"] - u["S"] >= 0.20,
        },
        "note": "Thinking traces, when present, are ignored for answer parsing; only the final post-</think> JSON is scored.",
    }
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8", newline="\n")
    print(output)
    if unknown:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
