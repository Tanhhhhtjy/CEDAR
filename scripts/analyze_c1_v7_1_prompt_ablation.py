#!/usr/bin/env python3
"""Compare frozen V7 cue prompts with V7.1 no-cue prompts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cedar_m2.checkpoint import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUE_ROOT = ROOT / "artifacts" / "m2" / "c1_v7_heldout_confirmation"
DEFAULT_NO_CUE_ROOT = ROOT / "artifacts" / "m2" / "c1_v7_1_prompt_cue_ablation"
ARMS = ("N", "AO", "S")
SCORINGS = ("strict", "lenient")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260721


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require_passed_audit(root: Path) -> None:
    report = read_json(root / "CROSS_CONDITION_AUDIT.json")
    if report.get("status") != "passed":
        raise ValueError(f"Cross-condition audit has not passed: {root}")


def by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed = {str(row.get("id")): row for row in rows}
    if len(indexed) != len(rows) or "None" in indexed:
        raise ValueError("Results require unique non-null IDs")
    return indexed


def build_pairs(cue_root: Path, no_cue_root: Path) -> list[dict[str, Any]]:
    require_passed_audit(cue_root)
    require_passed_audit(no_cue_root)
    cue_uncapped = by_id(read_jsonl(cue_root / "formal-uncapped" / "results.jsonl"))
    cue_p25 = by_id(read_jsonl(cue_root / "formal-p25" / "results.jsonl"))
    no_uncapped = read_jsonl(no_cue_root / "formal-uncapped" / "results.jsonl")
    no_p25 = by_id(read_jsonl(no_cue_root / "formal-p25" / "results.jsonl"))
    pairs: list[dict[str, Any]] = []
    for row in no_uncapped:
        no_id = str(row["id"])
        if not no_id.endswith("-NOCUE"):
            raise ValueError(f"Malformed no-cue ID: {no_id}")
        source_id = no_id.removesuffix("-NOCUE")
        if source_id not in cue_uncapped or source_id not in cue_p25 or no_id not in no_p25:
            raise ValueError(f"Missing paired result for {no_id}")
        records = {
            "cue_uncapped": cue_uncapped[source_id],
            "cue_p25": cue_p25[source_id],
            "no_cue_uncapped": row,
            "no_cue_p25": no_p25[no_id],
        }
        identity = (row["base_id"], row["family_id"], row["arm"])
        if any(
            (record["base_id"], record["family_id"], record["arm"]) != identity
            for record in records.values()
        ):
            raise ValueError(f"Metadata mismatch for {no_id}")
        pair = {
            "source_id": source_id,
            "no_cue_id": no_id,
            "base_id": row["base_id"],
            "family_id": row["family_id"],
            "arm": row["arm"],
        }
        for scoring in SCORINGS:
            for label, record in records.items():
                pair[f"{scoring}_{label}"] = record[f"{scoring}_category"]
        pairs.append(pair)
    if len(pairs) != 144:
        raise ValueError(f"Expected 144 four-way pairs, found {len(pairs)}")
    return pairs


def rate(
    pairs: Sequence[Mapping[str, Any]], arm: str, scoring: str, cell: str, category: str = "target"
) -> float:
    selected = [pair for pair in pairs if pair["arm"] == arm]
    if not selected:
        raise ValueError(f"No rows for arm {arm}")
    return sum(pair[f"{scoring}_{cell}"] == category for pair in selected) / len(selected)


def estimands(pairs: Sequence[Mapping[str, Any]], scoring: str) -> dict[str, float]:
    values: dict[str, float] = {}
    cells = ("cue_uncapped", "cue_p25", "no_cue_uncapped", "no_cue_p25")
    for arm in ARMS:
        for cell in cells:
            values[f"target_{arm}_{cell}"] = rate(pairs, arm, scoring, cell)
        values[f"cue_main_{arm}_uncapped"] = (
            values[f"target_{arm}_no_cue_uncapped"] - values[f"target_{arm}_cue_uncapped"]
        )
        values[f"cue_main_{arm}_p25"] = (
            values[f"target_{arm}_no_cue_p25"] - values[f"target_{arm}_cue_p25"]
        )
        values[f"cap_effect_{arm}_cue"] = (
            values[f"target_{arm}_cue_p25"] - values[f"target_{arm}_cue_uncapped"]
        )
        values[f"cap_effect_{arm}_no_cue"] = (
            values[f"target_{arm}_no_cue_p25"] - values[f"target_{arm}_no_cue_uncapped"]
        )
        values[f"cue_by_cap_{arm}"] = (
            values[f"cap_effect_{arm}_no_cue"] - values[f"cap_effect_{arm}_cue"]
        )
    values["cap_social_cue"] = values["cap_effect_S_cue"] - values["cap_effect_AO_cue"]
    values["cap_social_no_cue"] = (
        values["cap_effect_S_no_cue"] - values["cap_effect_AO_no_cue"]
    )
    values["cue_by_cap_social"] = values["cap_social_no_cue"] - values["cap_social_cue"]
    return values


def family_bootstrap(
    pairs: Sequence[Mapping[str, Any]], scoring: str, samples: int, seed: int
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["family_id"])].append(pair)
    families = sorted(grouped)
    rng = np.random.default_rng(seed)
    point = estimands(pairs, scoring)
    draws = {name: [] for name in point}
    for _ in range(samples):
        sampled = rng.choice(families, size=len(families), replace=True)
        bootstrap_pairs = [pair for family in sampled for pair in grouped[str(family)]]
        for name, value in estimands(bootstrap_pairs, scoring).items():
            draws[name].append(value)
    return {
        name: {
            "estimate": point[name],
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
    }


def interpretation(report: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    suppressive = report["cue_main_S_uncapped"]
    no_cue_harm = report["cap_effect_S_no_cue"]
    no_cue_social = report["cap_social_no_cue"]
    interaction = report["cue_by_cap_S"]
    return {
        "explicit_cue_suppresses_uncapped_S": (
            suppressive["estimate"] >= 0.10 and suppressive["ci_low"] > 0
        ),
        "p25_increases_no_cue_S": (
            no_cue_harm["estimate"] >= 0.10 and no_cue_harm["ci_low"] > 0
        ),
        "p25_no_cue_effect_exceeds_AO": (
            no_cue_social["estimate"] >= 0.10 and no_cue_social["ci_low"] > 0
        ),
        "p25_S_effect_is_cue_dependent": (
            abs(interaction["estimate"]) >= 0.10
            and (interaction["ci_low"] > 0 or interaction["ci_high"] < 0)
        ),
        "thresholds_are_descriptive_sensitivity_labels_not_item_selection_gates": True,
    }


def transition_table(
    pairs: Sequence[Mapping[str, Any]], arm: str, scoring: str, prefix: str
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        if pair["arm"] != arm:
            continue
        before = pair[f"{scoring}_{prefix}_uncapped"]
        after = pair[f"{scoring}_{prefix}_p25"]
        counts[f"{before}->{after}"] += 1
    return dict(sorted(counts.items()))


def analyze(
    cue_root: Path, no_cue_root: Path, *, bootstrap_samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    pairs = build_pairs(cue_root, no_cue_root)
    bootstrap = {
        scoring: family_bootstrap(
            pairs,
            scoring,
            samples=bootstrap_samples,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, scoring in enumerate(SCORINGS)
    }
    return {
        "status": "complete",
        "analysis_kind": "paired epistemic-status cue sensitivity analysis",
        "rows": len(pairs),
        "families": len({pair["family_id"] for pair in pairs}),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_scoring": "strict",
        "estimands": bootstrap,
        "interpretation": interpretation(bootstrap["strict"]),
        "transitions": {
            scoring: {
                prefix: {arm: transition_table(pairs, arm, scoring, prefix) for arm in ARMS}
                for prefix in ("cue", "no_cue")
            }
            for scoring in SCORINGS
        },
        "limitations": [
            "V is unchanged and is inherited from V7 rather than rerun.",
            (
                "The sensitivity analysis removes one sentence but does not make "
                "the task fully naturalistic."
            ),
            "The V7 single-review protocol deviation also applies to V7.1.",
        ],
        "no_item_selection_or_prompt_rewrite_after_outputs": True,
    }


def write_summary(path: Path, report: Mapping[str, Any]) -> None:
    strict = report["estimands"]["strict"]

    def pct(name: str) -> str:
        estimate = 100 * strict[name]["estimate"]
        low = 100 * strict[name]["ci_low"]
        high = 100 * strict[name]["ci_high"]
        return f"{estimate:+.2f}pp [{low:+.2f}, {high:+.2f}]"

    lines = [
        "# C1/V7.1 提示语消融结果",
        "",
        (
            "本分析把 V7 的显式提示版本与仅删除 "
            "`No new task information is supplied.` 的 V7.1 严格配对。"
        ),
        "",
        f"- 删除提示对 uncapped S target-follow 的影响：{pct('cue_main_S_uncapped')}",
        f"- 无提示条件下 p25 对 S target-follow 的影响：{pct('cap_effect_S_no_cue')}",
        f"- 无提示条件下净社会效应（S−AO）：{pct('cap_social_no_cue')}",
        f"- 提示 × p25 在 S 上的交互：{pct('cue_by_cap_S')}",
        f"- 提示 × p25 的净社会三重差：{pct('cue_by_cap_social')}",
        "",
        "完整 strict/lenient、迁移表和 family-cluster bootstrap 见 `C1_V7_1_ANALYSIS.json`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cue-root", type=Path, default=DEFAULT_CUE_ROOT)
    parser.add_argument("--no-cue-root", type=Path, default=DEFAULT_NO_CUE_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    report = analyze(args.cue_root, args.no_cue_root, bootstrap_samples=args.bootstrap_samples)
    atomic_write_json(args.no_cue_root / "C1_V7_1_ANALYSIS.json", report)
    write_summary(args.no_cue_root / "C1_V7_1_SUMMARY_ZH.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
