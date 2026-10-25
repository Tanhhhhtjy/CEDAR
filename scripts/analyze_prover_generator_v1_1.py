#!/usr/bin/env python3
"""Analyze the v1.1 uncapped template calibration and select S globally."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_development_v1_1"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_v1_1_uncapped"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_name(track: str) -> str:
    return f"v1_1_template_calibration_formal_{track.replace('-', '_')}_uncapped_seed0"


def target_indicator(row: dict[str, Any]) -> int:
    return int(row.get("outcome_class") == "induced_target")


def initial_indicator(row: dict[str, Any]) -> int:
    return int(row.get("outcome_class") == "initial")


def lenient_answer(row: dict[str, Any]) -> str | None:
    if row.get("chosen_option") in {"A", "B", "C", "D"}:
        return str(row["chosen_option"])
    matches = re.findall(r'"answer"\s*:\s*"([A-D])"', str(row.get("response", "")))
    return matches[-1] if matches else None


def arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    margins = [row["answer_logits"]["target_minus_initial"] for row in selected if row.get("answer_logits")]
    masses = [row["answer_logits"]["option_mass"] for row in selected if row.get("answer_logits")]
    entropies = [row["answer_logits"]["full_vocab_entropy"] for row in selected if row.get("answer_logits")]
    lenient_initial = sum(lenient_answer(row) == row["initial_answer"] for row in selected)
    return {
        "rows": len(selected),
        "outcome_counts": dict(sorted(Counter(row["outcome_class"] for row in selected).items())),
        "initial_rate_strict": sum(initial_indicator(row) for row in selected) / len(selected),
        "initial_rate_lenient": lenient_initial / len(selected),
        "target_rate": sum(target_indicator(row) for row in selected) / len(selected),
        "other_wrong_rate": sum(row["outcome_class"] == "other_wrong" for row in selected) / len(selected),
        "invalid_rate": sum(row["outcome_class"] == "invalid_or_abstain" for row in selected) / len(selected),
        "strict_parse_rate": sum(row.get("final_answer_parse_status") == "passed" for row in selected) / len(selected),
        "truncations": sum(bool(row.get("truncated")) for row in selected),
        "margin_count": len(margins),
        "target_minus_initial_mean": sum(margins) / len(margins) if margins else None,
        "target_minus_initial_median": median(margins) if margins else None,
        "option_mass_mean": sum(masses) / len(masses) if masses else None,
        "full_vocab_entropy_mean": sum(entropies) / len(entropies) if entropies else None,
    }


def paired(rows: list[dict[str, Any]], left: str, right: str, indicator: Callable[[dict[str, Any]], int]) -> dict[str, Any]:
    a = {row["base_id"]: row for row in rows if row["arm"] == left}
    b = {row["base_id"]: row for row in rows if row["arm"] == right}
    ids = sorted(set(a) & set(b))
    differences = [indicator(a[item]) - indicator(b[item]) for item in ids]
    transition_counts = Counter(f"{indicator(b[item])}->{indicator(a[item])}" for item in ids)
    margin_differences = [
        a[item]["answer_logits"]["target_minus_initial"] - b[item]["answer_logits"]["target_minus_initial"]
        for item in ids
        if a[item].get("answer_logits") and b[item].get("answer_logits")
    ]
    by_family: dict[str, list[float]] = defaultdict(list)
    for item, difference in zip(ids, differences, strict=True):
        by_family[a[item]["family_id"]].append(difference)
    family_scores = {family: sum(values) / len(values) for family, values in by_family.items()}
    families = sorted(family_scores)
    rng = random.Random(20260724)
    bootstrap = []
    for _ in range(20000):
        sampled = [rng.choice(families) for _ in families]
        values = [value for family in sampled for value in by_family[family]]
        bootstrap.append(sum(values) / len(values))
    bootstrap.sort()
    return {
        "left_minus_right": f"{left}-{right}",
        "paired_bases": len(ids),
        "discrete_difference": sum(differences) / len(differences),
        "discrete_transition_counts_right_to_left": dict(sorted(transition_counts.items())),
        "family_cluster_bootstrap_95ci": [bootstrap[500], bootstrap[19499]],
        "positive_families": sum(value > 0 for value in family_scores.values()),
        "negative_families": sum(value < 0 for value in family_scores.values()),
        "tied_families": sum(value == 0 for value in family_scores.values()),
        "family_scores": family_scores,
        "paired_margin_count": len(margin_differences),
        "paired_margin_mean": sum(margin_differences) / len(margin_differences) if margin_differences else None,
        "paired_margin_median": median(margin_differences) if margin_differences else None,
    }


def stratum_difference(rows: list[dict[str, Any]], left: str, right: str, variant: str) -> float:
    selected = [row for row in rows if row["variant"] == variant]
    return paired(selected, left, right, target_indicator)["discrete_difference"]


def analyze(output_root: Path) -> dict[str, Any]:
    audit_path = output_root / "TEMPLATE_CALIBRATION_FORMAL_AUDIT.json"
    audit = read_json(audit_path)
    if audit.get("status") != "passed":
        raise ValueError("Formal engineering audit must pass before scientific analysis")
    gates = read_json(DATA_ROOT / "GATE_SPEC.json")
    evidence = read_jsonl(output_root / run_name("evidence") / "results.jsonl")
    social = read_jsonl(output_root / run_name("social-candidates") / "results.jsonl")
    evidence_summaries = {arm: arm_summary(evidence, arm) for arm in ("N", "R", "V", "F")}
    social_summaries = {arm: arm_summary(social, arm) for arm in ("N", "AO", "S1", "S2")}
    vf = paired(evidence, "V", "F", target_indicator)
    vf["easy_difference"] = stratum_difference(evidence, "V", "F", "easy")
    vf["medium_difference"] = stratum_difference(evidence, "V", "F", "medium")
    evidence_gate = gates["evidence"]
    evidence_checks = {
        "discrete_v_minus_f": vf["discrete_difference"] >= evidence_gate["primary_discrete_v_minus_f_minimum"],
        "paired_margin_positive": vf["paired_margin_mean"] is not None and vf["paired_margin_mean"] > 0,
        "easy_and_medium_positive": vf["easy_difference"] > 0 and vf["medium_difference"] > 0,
        "family_direction": vf["positive_families"] > vf["negative_families"],
        "n_stable": evidence_summaries["N"]["initial_rate_strict"] >= evidence_gate["minimum_n_initial_rate"],
        "r_stable": evidence_summaries["R"]["initial_rate_strict"] >= evidence_gate["minimum_r_initial_rate"],
        "zero_truncation_by_arm": all(evidence_summaries[arm]["truncations"] <= limit for arm, limit in evidence_gate["maximum_truncations_by_arm"].items()),
    }
    social_gate = gates["social"]
    social_candidates = {}
    for arm in ("S1", "S2"):
        contrast = paired(social, arm, "AO", target_indicator)
        contrast["easy_difference"] = stratum_difference(social, arm, "AO", "easy")
        contrast["medium_difference"] = stratum_difference(social, arm, "AO", "medium")
        checks = {
            "n_stable": social_summaries["N"]["initial_rate_strict"] >= social_gate["minimum_n_initial_rate"],
            "ao_below_maximum": social_summaries["AO"]["target_rate"] <= social_gate["maximum_ao_target_rate"],
            "discrete_s_minus_ao": contrast["discrete_difference"] >= social_gate["minimum_discrete_s_minus_ao"],
            "paired_margin_positive": contrast["paired_margin_mean"] is not None and contrast["paired_margin_mean"] > 0,
            "easy_and_medium_nonnegative": contrast["easy_difference"] >= 0 and contrast["medium_difference"] >= 0,
        }
        social_candidates[arm] = {"contrast": contrast, "checks": checks, "passed": all(checks.values())}
    selected = "S1" if social_candidates["S1"]["passed"] else "S2" if social_candidates["S2"]["passed"] else None
    position_matrix = defaultdict(lambda: {arm: {"target": 0, "rows": 0} for arm in ("AO", "S1", "S2")})
    for row in social:
        if row["arm"] in {"AO", "S1", "S2"}:
            cell = position_matrix[row["position_transition"]][row["arm"]]
            cell["rows"] += 1
            cell["target"] += target_indicator(row)
    passed = all(evidence_checks.values()) and selected is not None
    result = {
        "schema": "cedar.prover_generator_development.v1_1.template_calibration_analysis.v1",
        "status": "passed" if passed else "failed_scientific_gate",
        "phase": "template-calibration",
        "development_only": True,
        "condition": "uncapped",
        "evidence": {"arm_summaries": evidence_summaries, "v_minus_f": vf, "checks": evidence_checks, "passed": all(evidence_checks.values())},
        "social": {"arm_summaries": social_summaries, "candidates": social_candidates, "position_transition_matrix": dict(position_matrix)},
        "selected_social_template": selected,
        "selection_rule": "weakest_globally_passing_template_s1_wins_ties",
        "itemwise_template_selection": False,
        "formal_audit_sha256": __import__("hashlib").sha256(audit_path.read_bytes()).hexdigest(),
        "p25_authorized": False,
        "next_gate": "untouched 12-family development validation" if passed else "revise measurement template before any capping",
    }
    output_path = output_root / "TEMPLATE_CALIBRATION_ANALYSIS.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = analyze(args.output_root.resolve())
    print({"status": result["status"], "selected_social_template": result["selected_social_template"], "evidence_passed": result["evidence"]["passed"], "p25_authorized": False})


if __name__ == "__main__":
    main()
