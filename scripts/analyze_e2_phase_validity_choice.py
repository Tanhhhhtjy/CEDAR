#!/usr/bin/env python3
"""Family-clustered E2A analysis and mechanism-routing Gate."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "e2_phase_validity_choice"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from e2_common import FORMAL_CONDITIONS, read_json, read_jsonl, write_json  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "e2_phase_validity_choice"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260725


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def family_metric(
    rows: Sequence[dict[str, Any]], indicator: Callable[[Mapping[str, Any]], float]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["family_id"]].append(float(indicator(row)))
    return {family: mean(values) for family, values in grouped.items()}


def bootstrap_family_contrast(scores: Mapping[str, float]) -> dict[str, Any]:
    families = sorted(scores)
    if not families:
        return {
            "status": "not_estimable",
            "estimate": None,
            "family_cluster_bootstrap_95ci": None,
            "probability_positive": None,
            "probability_negative": None,
            "families": 0,
            "family_scores": {},
            "draws": [],
        }
    observed = mean([scores[family] for family in families])
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [mean([scores[rng.choice(families)] for _ in families]) for _ in range(BOOTSTRAP_DRAWS)]
    return {
        "estimate": observed,
        "family_cluster_bootstrap_95ci": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "probability_positive": sum(value > 0 for value in draws) / len(draws),
        "probability_negative": sum(value < 0 for value in draws) / len(draws),
        "families": len(families),
        "family_scores": dict(sorted(scores.items())),
        "draws": draws,
    }


def paired_condition_scores(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    indicator: Callable[[Mapping[str, Any]], float],
) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] in {"uncapped", condition}]
    by_cell = {(row["id"], row["condition"]): row for row in selected}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        if row["condition"] != "uncapped":
            continue
        treated = by_cell[(row["id"], condition)]
        grouped[row["family_id"]].append(indicator(treated) - indicator(row))
    return {family: mean(values) for family, values in grouped.items()}


def target_follow(row: Mapping[str, Any]) -> float:
    return float(row.get("outcome_class") == "induced_target")


def sufficient(row: Mapping[str, Any]) -> float:
    return float(row.get("chosen_semantic") == "sufficient")


def answer_margin(row: Mapping[str, Any]) -> float:
    value = row.get("answer_logits", {}).get("target_minus_initial")
    if value is None:
        raise ValueError(f"Missing Answer target margin: {row.get('id')}")
    return float(value)


def geometry(row: Mapping[str, Any]) -> dict[str, float]:
    logits = row.get("answer_logits", {}).get("logits")
    if not isinstance(logits, dict) or set(logits) != {"A", "B", "C", "D"}:
        raise ValueError(f"Missing A/B/C/D logits: {row.get('id')}")
    initial = row["initial_answer"]
    target = row["target_answer"]
    other = [label for label in ("A", "B", "C", "D") if label not in {initial, target}]
    m_target = float(logits[target]) - float(logits[initial])
    m_other = [float(logits[label]) - float(logits[initial]) for label in other]
    m_background = mean(m_other)
    return {
        "m_target": m_target,
        "m_other1": m_other[0],
        "m_other2": m_other[1],
        "m_background": m_background,
        "m_general": (m_target + sum(m_other)) / 3,
        "m_specific": m_target - mean(m_other),
    }


def reasoning_tokens(row: Mapping[str, Any]) -> float:
    value = row.get("reasoning_token_count")
    if value is None:
        raise ValueError(f"Missing reasoning-token count: {row.get('id')}")
    return float(value)


def answer_telemetry(row: Mapping[str, Any], field: str) -> float:
    value = row.get("answer_logits", {}).get(field)
    if value is None:
        raise ValueError(f"Missing Answer telemetry {field}: {row.get('id')}")
    return float(value)


def decision_value(row: Mapping[str, Any]) -> str:
    if row["assay"] == "answer":
        return str(row.get("outcome_class"))
    return str(row.get("chosen_semantic"))


def paired_condition_scores_by_decision_change(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    indicator: Callable[[Mapping[str, Any]], float],
    changed: bool,
) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] in {"uncapped", condition}]
    by_cell = {(row["id"], row["condition"]): row for row in selected}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        if row["condition"] != "uncapped":
            continue
        treated = by_cell[(row["id"], condition)]
        if (decision_value(treated) != decision_value(row)) != changed:
            continue
        grouped[row["family_id"]].append(indicator(treated) - indicator(row))
    return {family: mean(values) for family, values in grouped.items()}


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def four_state_transitions(
    rows: Sequence[dict[str, Any]], *, condition: str
) -> dict[str, Any]:
    states = ("initial", "induced_target", "other_wrong", "invalid_or_abstain")
    selected = [row for row in rows if row["condition"] in {"uncapped", condition}]
    by_cell = {(row["id"], row["condition"]): row for row in selected}
    result: dict[str, Any] = {}
    for arm in ("N", "R", "V", "F"):
        counts = {source: {target: 0 for target in states} for source in states}
        n = 0
        for row in selected:
            if row["condition"] != "uncapped" or row["arm"] != arm:
                continue
            treated = by_cell[(row["id"], condition)]
            source = str(row["outcome_class"])
            target = str(treated["outcome_class"])
            counts[source][target] += 1
            n += 1
        result[arm] = {"n": n, "counts": counts}
    return result


def post_hoc_secondary(
    answer: Sequence[dict[str, Any]], validity: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    assays = {"answer": list(answer), "validity": list(validity)}
    result: dict[str, Any] = {
        "scope": "post_hoc_secondary",
        "stratification_note": (
            "Same/changed-decision strata are descriptive post-treatment strata; "
            "families without observations in a stratum are omitted, and no causal "
            "interpretation is permitted."
        ),
        "reasoning_length_tokens": {},
        "answer_step_telemetry": {},
        "answer_four_state_transitions_vs_uncapped": {},
    }
    for assay, assay_rows in assays.items():
        raw = {
            condition: descriptive(
                [reasoning_tokens(row) for row in assay_rows if row["condition"] == condition]
            )
            for condition in FORMAL_CONDITIONS
        }
        contrasts = {}
        for condition in FORMAL_CONDITIONS[1:]:
            all_scores = paired_condition_scores(
                assay_rows, condition=condition, indicator=reasoning_tokens
            )
            same_scores = paired_condition_scores_by_decision_change(
                assay_rows,
                condition=condition,
                indicator=reasoning_tokens,
                changed=False,
            )
            changed_scores = paired_condition_scores_by_decision_change(
                assay_rows,
                condition=condition,
                indicator=reasoning_tokens,
                changed=True,
            )
            contrasts[condition] = {
                "all": bootstrap_family_contrast(all_scores),
                "same_decision": bootstrap_family_contrast(same_scores),
                "changed_decision": bootstrap_family_contrast(changed_scores),
            }
            for summary in contrasts[condition].values():
                summary.pop("draws", None)
        result["reasoning_length_tokens"][assay] = {
            "raw_by_condition": raw,
            "paired_contrasts_vs_uncapped": contrasts,
        }

        result["answer_step_telemetry"][assay] = {}
        for field in ("full_vocab_entropy", "option_mass"):
            field_contrasts = {}
            for condition in FORMAL_CONDITIONS[1:]:
                scores = paired_condition_scores(
                    assay_rows,
                    condition=condition,
                    indicator=lambda row, name=field: answer_telemetry(row, name),
                )
                summary = bootstrap_family_contrast(scores)
                summary.pop("draws", None)
                field_contrasts[condition] = summary
            result["answer_step_telemetry"][assay][field] = field_contrasts

    for condition in FORMAL_CONDITIONS[1:]:
        result["answer_four_state_transitions_vs_uncapped"][condition] = (
            four_state_transitions(answer, condition=condition)
        )
    return result


def discrimination(
    rows: Sequence[dict[str, Any]],
    condition: str,
    indicator: Callable[[Mapping[str, Any]], float],
    arm_field: str,
) -> float:
    v = [indicator(row) for row in rows if row["condition"] == condition and row[arm_field] == "V"]
    f = [indicator(row) for row in rows if row["condition"] == condition and row[arm_field] == "F"]
    return mean(v) - mean(f)


def discrimination_family_scores(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    indicator: Callable[[Mapping[str, Any]], float],
    arm_field: str,
) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["condition"] in {"uncapped", condition}:
            grouped[row["family_id"]][f"{row[arm_field]}::{row['condition']}"].append(
                indicator(row)
            )
    scores = {}
    for family, cells in grouped.items():
        uncapped = mean(cells["V::uncapped"]) - mean(cells["F::uncapped"])
        treated = mean(cells[f"V::{condition}"]) - mean(cells[f"F::{condition}"])
        scores[family] = treated - uncapped
    return scores


def fit_hierarchical_model(rows: Sequence[dict[str, Any]], *, assay: str) -> dict[str, Any]:
    """Fit the frozen family/base Bayesian logistic mixed model."""
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    records = []
    for row in rows:
        condition = row["condition"]
        source_arm = row["source_arm"] if assay == "validity" else row["arm"]
        if source_arm not in {"V", "F"}:
            continue
        record = {
            "response": sufficient(row) if assay == "validity" else target_follow(row),
            "validity": int(source_arm == "V"),
            "prefill_cap": int(condition in {"prefill_only", "full_p25"}),
            "decode_cap": int(condition in {"decode_only", "full_p25"}),
            "variant": row["variant"],
            "position_transition": row["position_transition"],
            "family_id": row["family_id"],
            "base_id": row["base_id"],
        }
        if assay == "validity":
            record["label_mapping"] = row["sufficient_label"]
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    formula = "response ~ validity * prefill_cap * decode_cap + C(variant) + C(position_transition)"
    if assay == "validity":
        formula += " + C(label_mapping)"
    model = BinomialBayesMixedGLM.from_formula(
        formula,
        {
            "family": "0 + C(family_id)",
            "base": "0 + C(base_id)",
        },
        frame,
    )
    fitted = model.fit_vb(fit_method="BFGS", minim_opts={"maxiter": 2000})
    return {
        "method": "statsmodels BinomialBayesMixedGLM variational Bayes",
        "formula": formula,
        "rows": len(frame),
        "fixed_effects": {
            name: {"posterior_mean": float(value), "posterior_sd": float(sd)}
            for name, value, sd in zip(model.exog_names, fitted.fe_mean, fitted.fe_sd, strict=True)
        },
        "variance_components": {
            name: {"posterior_log_sd_mean": float(value), "posterior_log_sd_sd": float(sd)}
            for name, value, sd in zip(model.vcp_names, fitted.vcp_mean, fitted.vcp_sd, strict=True)
        },
        "optimizer": {
            "success": bool(fitted.optim_retvals.get("success", False)),
            "message": str(fitted.optim_retvals.get("message", "")),
            "iterations": int(fitted.optim_retvals.get("nit", -1)),
        },
    }


def summarize_rates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answer = [row for row in rows if row["assay"] == "answer"]
    validity = [row for row in rows if row["assay"] == "validity"]
    summary: dict[str, Any] = {"answer": {}, "validity": {}}
    for condition in FORMAL_CONDITIONS:
        summary["answer"][condition] = {
            arm: {
                "n": len(values),
                "target_rate": mean([target_follow(row) for row in values]),
                "initial_rate": mean(
                    [float(row.get("outcome_class") == "initial") for row in values]
                ),
                "invalid_rate": mean(
                    [float(row.get("outcome_class") == "invalid_or_abstain") for row in values]
                ),
            }
            for arm in ("N", "R", "V", "F")
            if (
                values := [
                    row for row in answer if row["condition"] == condition and row["arm"] == arm
                ]
            )
        }
        summary["answer"][condition]["selectivity"] = discrimination(
            answer, condition, target_follow, "arm"
        )
        v = summary["answer"][condition]["V"]["target_rate"]
        f = summary["answer"][condition]["F"]["target_rate"]
        summary["answer"][condition]["common_update"] = (v + f) / 2
        summary["validity"][condition] = {
            arm: {
                "n": len(values),
                "sufficient_rate": mean([sufficient(row) for row in values]),
                "invalid_rate": mean(
                    [float(row.get("final_answer_parse_status") != "passed") for row in values]
                ),
            }
            for arm in ("V", "F")
            if (
                values := [
                    row
                    for row in validity
                    if row["condition"] == condition and row["source_arm"] == arm
                ]
            )
        }
        summary["validity"][condition]["discrimination"] = discrimination(
            validity, condition, sufficient, "source_arm"
        )
        v = summary["validity"][condition]["V"]["sufficient_rate"]
        f = summary["validity"][condition]["F"]["sufficient_rate"]
        summary["validity"][condition]["overall_sufficient_rate"] = (v + f) / 2
    return summary


def analyze(output_root: Path, *, reuse_frozen_hierarchical_models: bool = False) -> dict[str, Any]:
    audit = read_json(output_root / "E2A_AUDIT.json")
    if audit.get("status") != "passed":
        raise ValueError("E2A audit must pass before scientific analysis")
    rows = read_jsonl(output_root / "formal-all-seed0" / "results.jsonl")
    if len(rows) != 576:
        raise ValueError("Expected 576 E2A rows")
    answer = [row for row in rows if row["assay"] == "answer"]
    validity = [row for row in rows if row["assay"] == "validity"]
    rates = summarize_rates(rows)
    hierarchical_model_source = "refit_from_formal_rows"
    if reuse_frozen_hierarchical_models:
        previous = read_json(output_root / "E2A_ANALYSIS.json")
        if previous.get("status") != "complete" or previous.get("rows") != 576:
            raise ValueError("A complete 576-row frozen analysis is required for hygiene reuse")
        hierarchical_models = previous.get("hierarchical_models")
        if not isinstance(hierarchical_models, dict) or set(hierarchical_models) != {
            "answer",
            "validity",
        }:
            raise ValueError("Frozen hierarchical models are missing or malformed")
        hierarchical_model_source = (
            "reused_unchanged_from_e2a_result_commit_"
            "0c8517bb5fb4640f7d5e4d51356d0cf167046643"
        )
    else:
        hierarchical_models = {
            "validity": fit_hierarchical_model(validity, assay="validity"),
            "answer": fit_hierarchical_model(answer, assay="answer"),
        }
    contrasts: dict[str, Any] = {
        "validity_discrimination": {},
        "answer_selectivity": {},
        "common_update": {},
        "geometry": {},
    }
    for condition in FORMAL_CONDITIONS[1:]:
        contrasts["validity_discrimination"][condition] = bootstrap_family_contrast(
            discrimination_family_scores(
                validity, condition=condition, indicator=sufficient, arm_field="source_arm"
            )
        )
        contrasts["answer_selectivity"][condition] = bootstrap_family_contrast(
            discrimination_family_scores(
                answer, condition=condition, indicator=target_follow, arm_field="arm"
            )
        )
        common_scores: dict[str, float] = {}
        for family in sorted({row["family_id"] for row in answer}):
            family_rows = [
                row for row in answer if row["family_id"] == family and row["arm"] in {"V", "F"}
            ]
            treated = mean(
                [target_follow(row) for row in family_rows if row["condition"] == condition]
            )
            untreated = mean(
                [target_follow(row) for row in family_rows if row["condition"] == "uncapped"]
            )
            common_scores[family] = treated - untreated
        contrasts["common_update"][condition] = bootstrap_family_contrast(common_scores)
        contrasts["geometry"][condition] = {}
        for metric in ("m_target", "m_background", "m_general", "m_specific"):
            scores = paired_condition_scores(
                answer,
                condition=condition,
                indicator=lambda row, name=metric: geometry(row)[name],
            )
            contrasts["geometry"][condition][metric] = bootstrap_family_contrast(scores)

    full_validity = contrasts["validity_discrimination"]["full_p25"]
    validity_draws = full_validity.pop("draws")
    validity_equivalence_probability = sum(abs(value) <= 0.10 for value in validity_draws) / len(
        validity_draws
    )
    validity_damage_probability = sum(value <= -0.10 for value in validity_draws) / len(
        validity_draws
    )
    full_common = contrasts["common_update"]["full_p25"]
    common_draws = full_common.pop("draws")
    common_direction_probability = sum(value > 0 for value in common_draws) / len(common_draws)
    for family in ("answer_selectivity",):
        for condition in FORMAL_CONDITIONS[1:]:
            contrasts[family][condition].pop("draws", None)
    for condition in FORMAL_CONDITIONS[1:]:
        if condition != "full_p25":
            contrasts["validity_discrimination"][condition].pop("draws", None)
            contrasts["common_update"][condition].pop("draws", None)
        for metric in contrasts["geometry"][condition].values():
            metric.pop("draws", None)

    full_general = contrasts["geometry"]["full_p25"]["m_general"]
    general_by_arm = {}
    for arm in ("N", "R", "V", "F"):
        arm_rows = [row for row in answer if row["arm"] == arm]
        summary = bootstrap_family_contrast(
            paired_condition_scores(
                arm_rows,
                condition="full_p25",
                indicator=lambda row: geometry(row)["m_general"],
            )
        )
        summary.pop("draws", None)
        general_by_arm[arm] = summary
    contrasts["geometry"]["full_p25"]["m_general_by_arm"] = general_by_arm
    baseline_specific = family_metric(
        [row for row in answer if row["condition"] == "uncapped"],
        lambda row: geometry(row)["m_specific"],
    )
    rope = 0.25 * statistics.stdev(baseline_specific.values())
    specific_scores = paired_condition_scores(
        answer,
        condition="full_p25",
        indicator=lambda row: geometry(row)["m_specific"],
    )
    specific_bootstrap = bootstrap_family_contrast(specific_scores)
    specific_equivalence_probability = (
        sum(abs(value) <= rope for value in specific_bootstrap.pop("draws")) / BOOTSTRAP_DRAWS
    )
    n_r_stable = all(
        rates["answer"][condition][arm]["initial_rate"] == 1.0
        for condition in FORMAL_CONDITIONS
        for arm in ("N", "R")
    )
    common_family_scores = list(full_common["family_scores"].values())
    family_direction_ok = (
        sum(value >= 0 for value in common_family_scores) >= 8
        and sum(value > 0 for value in common_family_scores) >= 4
        and all(
            mean(common_family_scores[:index] + common_family_scores[index + 1 :]) >= 0
            for index in range(len(common_family_scores))
        )
    )
    completion_ok = True
    for assay_name, arms in (("answer", ("N", "R", "V", "F")), ("validity", ("V", "F"))):
        key = "invalid_rate"
        for arm in arms:
            baseline = rates[assay_name]["uncapped"][arm][key]
            for condition in FORMAL_CONDITIONS[1:]:
                if abs(rates[assay_name][condition][arm][key] - baseline) > 0.05:
                    completion_ok = False
    gate_t = (
        validity_equivalence_probability >= 0.80
        and common_direction_probability >= 0.90
        and family_direction_ok
        and completion_ok
        and n_r_stable
    )
    gate_e = validity_damage_probability >= 0.80
    gate_c = (
        full_general["probability_positive"] >= 0.95
        and sum(item["estimate"] > 0 for item in general_by_arm.values()) >= 3
        and specific_equivalence_probability >= 0.80
    )
    models_converged = all(model["optimizer"]["success"] for model in hierarchical_models.values())
    gates = [name for name, passed in (("T", gate_t), ("E", gate_e), ("C", gate_c)) if passed]
    if not models_converged:
        gates = []
    result = {
        "schema": "cedar.e2_phase_validity_choice.analysis.v1_1",
        "status": "complete",
        "scope": "exploratory_mechanism_assay",
        "unit_of_analysis": "12 families; 24 bases nested within family",
        "rows": len(rows),
        "rates": rates,
        "hierarchical_models": hierarchical_models,
        "hierarchical_model_source": hierarchical_model_source,
        "contrasts_vs_uncapped": contrasts,
        "routing_probabilities": {
            "validity_equivalence_within_10pp": validity_equivalence_probability,
            "validity_damage_at_least_10pp": validity_damage_probability,
            "common_update_positive": common_direction_probability,
            "m_specific_rope": rope,
            "m_specific_equivalence": specific_equivalence_probability,
            "common_update_family_direction_ok": family_direction_ok,
            "completion_difference_within_5pp": completion_ok,
        },
        "post_hoc_secondary": post_hoc_secondary(answer, validity),
        "route_gate": gates if gates else ["F"],
        "route_gate_is_manuscript_confirmation": False,
        "analysis_hygiene": {
            "raw_rows_changed": False,
            "primary_estimands_changed": False,
            "routing_thresholds_changed": False,
            "post_hoc_secondary_added": True,
            "criterion_renamed_to_overall_sufficient_rate": True,
        },
        "hierarchical_model_status": "complete" if models_converged else "nonconverged",
        "limitations": [
            "These 24 bases are an exploratory mechanism assay, not a new confirmation set.",
            (
                "Bootstrap routing probabilities are sensitivity analyses around the "
                "hierarchical model."
            ),
            "Mechanism language requires E2B or bidirectional patching after a non-F route.",
            (
                "Reasoning length, answer-step telemetry, and four-state transition analyses "
                "were added after E2A completion and are post-hoc secondary analyses."
            ),
        ],
    }
    write_json(output_root / "E2A_ANALYSIS.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reuse-frozen-hierarchical-models",
        action="store_true",
        help=(
            "Reuse the unchanged fitted model payload from the frozen E2A result while "
            "rebuilding hygiene-only summaries."
        ),
    )
    args = parser.parse_args()
    result = analyze(
        args.output_root,
        reuse_frozen_hierarchical_models=args.reuse_frozen_hierarchical_models,
    )
    print(
        json.dumps({"status": result["status"], "route_gate": result["route_gate"]}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
