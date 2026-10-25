#!/usr/bin/env python3
"""Recompute matched forced-choice transport summaries from row-level records."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

OPTIONS = ("A", "B", "C", "D")
ARMS = ("N", "R", "V", "F")
STATES = ("initial", "induced_target", "other_wrong", "invalid_or_abstain")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def margins(record: dict[str, Any]) -> dict[str, float]:
    logits = record["answer_logits"]["logits"]
    initial = record["initial_answer"]
    target = record["target_answer"]
    alternatives = [option for option in OPTIONS if option != initial]
    background = [option for option in OPTIONS if option not in (initial, target)]
    return {
        "m_general": float(np.mean([logits[o] - logits[initial] for o in alternatives])),
        "m_target": float(logits[target] - logits[initial]),
        "m_specific": float(logits[target] - np.mean([logits[o] for o in background])),
    }


def paired_rows(records: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(record["base_id"], record["arm"])][record["condition"]] = record
    rows = []
    for (base_id, arm), cells in sorted(grouped.items()):
        uncapped, capped = cells["uncapped"], cells["p25"]
        before, after = margins(uncapped), margins(capped)
        rows.append(
            {
                "model": model,
                "base_id": base_id,
                "family_id": uncapped["family_id"],
                "arm": arm,
                "m_general_uncapped": before["m_general"],
                "m_general_p25": after["m_general"],
                "delta_m_general": after["m_general"] - before["m_general"],
                "delta_m_target": after["m_target"] - before["m_target"],
                "delta_m_specific": after["m_specific"] - before["m_specific"],
                "target_follow_uncapped": float(uncapped["outcome_class"] == "induced_target"),
                "target_follow_p25": float(capped["outcome_class"] == "induced_target"),
                "source_state": uncapped["outcome_class"],
                "destination_state": capped["outcome_class"],
            }
        )
    return rows


def family_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    families = sorted({row["family_id"] for row in rows})
    by_family = {family: [row for row in rows if row["family_id"] == family] for family in families}
    rng = random.Random(seed)
    values = []
    for index in range(draws):
        sampled = [rng.choice(families) for _ in families]
        values.append(statistic([row for family in sampled for row in by_family[family]]))

    def percentile(probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        weight = position - lower
        return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)

    return percentile(0.025), percentile(0.975)


def estimate(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    low, high = family_bootstrap(rows, statistic, draws=draws, seed=seed)
    return {"estimate": statistic(rows), "ci_low": low, "ci_high": high}


def mean_field(field: str, arm: str | None = None):
    def statistic(rows: list[dict[str, Any]]) -> float:
        values = [row[field] for row in rows if arm is None or row["arm"] == arm]
        return float(np.mean(values))

    return statistic


def vf_target_contrast(rows: list[dict[str, Any]]) -> float:
    valid = np.mean([row["delta_m_target"] for row in rows if row["arm"] == "V"])
    fallacious = np.mean([row["delta_m_target"] for row in rows if row["arm"] == "F"])
    return float(valid - fallacious)


def model_summary(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    summary = {
        "overall_delta_m_general": estimate(
            rows, mean_field("delta_m_general"), draws=draws, seed=seed
        ),
        "v_minus_f_continuous_selectivity_change": estimate(
            rows, vf_target_contrast, draws=draws, seed=seed
        ),
        "arm_delta_m_general": {
            arm: estimate(rows, mean_field("delta_m_general", arm), draws=draws, seed=seed)
            for arm in ARMS
        },
        "pooled_delta_m_specific": estimate(
            rows, mean_field("delta_m_specific"), draws=draws, seed=seed
        ),
        "target_follow": {},
        "nr_stability": {},
    }
    for arm in ("V", "F"):
        arm_rows = [row for row in rows if row["arm"] == arm]

        def change(sample: list[dict[str, Any]]) -> float:
            return float(
                np.mean(
                    [row["target_follow_p25"] - row["target_follow_uncapped"] for row in sample]
                )
            )

        summary["target_follow"][arm] = {
            "uncapped_rate": float(np.mean([row["target_follow_uncapped"] for row in arm_rows])),
            "p25_rate": float(np.mean([row["target_follow_p25"] for row in arm_rows])),
            **estimate(arm_rows, change, draws=draws, seed=seed),
        }
    for arm in ("N", "R"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        stable = sum(row["source_state"] == row["destination_state"] for row in arm_rows)
        summary["nr_stability"][arm] = {
            "stable_pairs": stable,
            "total_pairs": len(arm_rows),
            "stability_rate": stable / len(arm_rows),
        }
    migrations = Counter(
        (row["arm"], row["source_state"], row["destination_state"]) for row in rows
    )
    summary["migrations"] = [
        {
            "arm": arm,
            "source_state": source,
            "destination_state": destination,
            "count": migrations[(arm, source, destination)],
        }
        for arm in ARMS
        for source in STATES
        for destination in STATES
    ]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.package_root.resolve()
    records_root = root / "runs/cross_model"
    model_records = {
        "Qwen-FC": read_jsonl(records_root / "qwen_fc_records.jsonl"),
        "Llama": read_jsonl(records_root / "llama_records.jsonl"),
    }
    rows = {
        model: paired_rows(records, model) for model, records in model_records.items()
    }
    summaries: dict[str, dict[str, Any]] = {
        model: {"target_follow": {}, "nr_stability": {}} for model in rows
    }
    # The frozen analysis used one RNG stream in this order.  Preserving it
    # makes the percentile endpoints byte-for-byte comparable to the release.
    for model in ("Qwen-FC", "Llama"):
        summaries[model]["overall_delta_m_general"] = estimate(
            rows[model], mean_field("delta_m_general"), draws=args.draws, seed=args.seed
        )
        summaries[model]["v_minus_f_continuous_selectivity_change"] = estimate(
            rows[model], vf_target_contrast, draws=args.draws, seed=args.seed
        )
    for model in ("Qwen-FC", "Llama"):
        summaries[model]["arm_delta_m_general"] = {}
        for arm in ARMS:
            summaries[model]["arm_delta_m_general"][arm] = estimate(
                rows[model], mean_field("delta_m_general", arm), draws=args.draws, seed=args.seed
            )
    for model in ("Qwen-FC", "Llama"):
        summaries[model]["pooled_delta_m_specific"] = estimate(
            rows[model], mean_field("delta_m_specific"), draws=args.draws, seed=args.seed
        )
    for model in ("Qwen-FC", "Llama"):
        for arm in ("V", "F"):
            arm_rows = [row for row in rows[model] if row["arm"] == arm]

            def change(sample: list[dict[str, Any]]) -> float:
                return float(
                    np.mean(
                        [
                            row["target_follow_p25"] - row["target_follow_uncapped"]
                            for row in sample
                        ]
                    )
                )

            summaries[model]["target_follow"][arm] = {
                "uncapped_rate": float(
                    np.mean([row["target_follow_uncapped"] for row in arm_rows])
                ),
                "p25_rate": float(np.mean([row["target_follow_p25"] for row in arm_rows])),
                **estimate(arm_rows, change, draws=args.draws, seed=args.seed),
            }
    for model in ("Qwen-FC", "Llama"):
        for arm in ("N", "R"):
            arm_rows = [row for row in rows[model] if row["arm"] == arm]
            stable = sum(row["source_state"] == row["destination_state"] for row in arm_rows)
            summaries[model]["nr_stability"][arm] = {
                "stable_pairs": stable,
                "total_pairs": len(arm_rows),
                "stability_rate": stable / len(arm_rows),
            }
        migrations = Counter(
            (row["arm"], row["source_state"], row["destination_state"])
            for row in rows[model]
        )
        summaries[model]["migrations"] = [
            {
                "arm": arm,
                "source_state": source,
                "destination_state": destination,
                "count": migrations[(arm, source, destination)],
            }
            for arm in ARMS
            for source in STATES
            for destination in STATES
        ]
    result = {
        "schema": "cedar.cross_model_transport.recomputed.v1",
        "bootstrap": {
            "unit": "structural_family",
            "method": "percentile_family_cluster_bootstrap",
            "draws": args.draws,
            "seed": args.seed,
        },
        "record_counts": {model: len(records) for model, records in model_records.items()},
        "paired_counts": {model: len(model_rows) for model, model_rows in rows.items()},
        "models": summaries,
    }
    output = args.output or records_root / "CROSS_MODEL_RECOMPUTED.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
