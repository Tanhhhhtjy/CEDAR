from __future__ import annotations

from scripts import analyze_c3_sycobench_bidirectional as analysis


def result_record(initial: str, target: str, margin: float, answer: str, correction: float) -> dict:
    return {
        "answer_logits": {
            "target_minus_initial": margin,
            "option_argmax": answer,
            "option_mass": 0.95,
            "full_vocab_entropy": 1.0,
            "top_token_is_option": True,
        },
        "cap_stats": {
            "pooled": {
                "mean_projection_correction": correction,
                "trigger_rate": 0.5,
            },
            "by_layer": {
                str(layer): {
                    "mean_projection_correction": correction,
                    "trigger_rate": 0.5,
                }
                for layer in range(46, 54)
            },
        },
    }


def synthetic_pairs() -> list[dict]:
    pairs = []
    domains = ["d0", "d1", "d2"]
    difficulties = ["easy", "medium", "hard"]
    for index in range(600):
        base_id = f"B{index:04d}"
        gold = "D"
        foil = "A"
        common = {
            "base_id": base_id,
            "source_id": f"source-{index}",
            "normalized_stem_id": f"stem-{index // 3}",
            "exact_mcq_id": f"mcq-{index}",
            "domain": domains[(index // 75) % len(domains)],
            "difficulty": difficulties[index % 3],
            "template_id": index % 4,
            "gold_answer": gold,
            "foil_answer": foil,
        }
        for direction in analysis.DIRECTIONS:
            initial, target = (gold, foil) if direction == "harmful" else (foil, gold)
            for arm in analysis.ARMS:
                delta = {
                    ("harmful", "AO"): 0.1,
                    ("harmful", "S"): 0.4,
                    ("helpful", "AO"): 0.1,
                    ("helpful", "S"): 0.6,
                }[(direction, arm)]
                pairs.append(
                    {
                        "id": f"{base_id}-{direction}-{arm}",
                        "direction_id": f"{base_id}-{direction}",
                        **common,
                        "direction": direction,
                        "arm": arm,
                        "initial_answer": initial,
                        "target_answer": target,
                        "initial_position": initial,
                        "target_position": target,
                        "uncapped": result_record(initial, target, -1.0, initial, 0.0),
                        "p25": result_record(initial, target, -1.0 + delta, target, 0.2),
                    }
                )
    return pairs


def test_base_metrics_recover_mirrored_selectivity() -> None:
    bases = analysis.build_base_rows(synthetic_pairs())
    assert len(bases) == 600
    assert all(abs(row["cap_social_harmful"] - 0.3) < 1e-12 for row in bases)
    assert all(abs(row["cap_social_helpful"] - 0.5) < 1e-12 for row in bases)
    assert all(abs(row["selectivity_change"] - 0.2) < 1e-12 for row in bases)
    assert all(row["target_change_harmful_S"] == 1.0 for row in bases)


def test_cluster_bootstrap_is_deterministic_and_keeps_stems_intact() -> None:
    rows = [
        {
            "normalized_stem_id": "s1",
            "domain": "d",
            "difficulty": "easy",
            "effect": 1.0,
        },
        {
            "normalized_stem_id": "s1",
            "domain": "d",
            "difficulty": "hard",
            "effect": 3.0,
        },
        {
            "normalized_stem_id": "s2",
            "domain": "d",
            "difficulty": "easy",
            "effect": 2.0,
        },
        {
            "normalized_stem_id": "s2",
            "domain": "d",
            "difficulty": "hard",
            "effect": 4.0,
        },
    ]
    first = analysis.cluster_stratified_bootstrap(rows, ["effect"], samples=200, seed=7)
    second = analysis.cluster_stratified_bootstrap(rows, ["effect"], samples=200, seed=7)
    assert first == second
    assert first["effect"]["estimate"] == 2.5
    assert first["effect"]["clusters"] == 2
    assert first["effect"]["domain_by_difficulty_counts_preserved_each_draw"] is True


def test_transition_matrix_reports_all_cells() -> None:
    transitions = analysis.transition_rows(synthetic_pairs())
    assert len(transitions) == 2 * 2 * 4 * 4
    indexed = {
        (row["direction"], row["arm"], row["from"], row["to"]): row["count"] for row in transitions
    }
    assert indexed[("harmful", "S", "initial", "target")] == 600
    assert indexed[("helpful", "AO", "initial", "target")] == 600


def test_correction_analysis_reports_cells_layers_contrasts_and_quartiles() -> None:
    pairs = synthetic_pairs()
    bases = analysis.build_base_rows(pairs)
    report = analysis.correction_amount_analysis(bases, pairs)
    assert len(report["cells"]) == 4
    assert set(report["by_layer"]["harmful"]["S"]) == {str(layer) for layer in range(46, 54)}
    assert "S_minus_AO_helpful" in report["contrasts"]
    assert len(report["quartile_descriptive"]) == 16
    assert report["causal_interpretation_allowed"] is False


def test_equivalence_and_reliability_require_predefined_thresholds() -> None:
    inside = {"estimate": 0.0, "ci_low": -0.1, "ci_high": 0.1, "ci90_low": -0.08, "ci90_high": 0.08}
    assert analysis.equivalence_result(inside, 0.2)["equivalent"] is True
    assert analysis.reliability_label(inside, 0.2) == "inconclusive"
    small = {"estimate": 0.1, "ci_low": 0.01, "ci_high": 0.19}
    assert analysis.reliability_label(small, 0.2) == "small_reliable_effect"
    substantive = {"estimate": -0.3, "ci_low": -0.5, "ci_high": -0.1}
    assert analysis.reliability_label(substantive, 0.2) == "reliable_and_substantive"


def test_gate_decision_separates_selectivity_from_distribution_failure() -> None:
    rows = [
        {
            "cap_shift_harmful_S": -0.2,
            "cap_shift_helpful_S": 0.3,
            "cap_shift_harmful_AO": 0.0,
            "cap_shift_helpful_AO": 0.0,
            "cap_social_harmful": -0.2,
            "cap_social_helpful": 0.3,
            "target_change_harmful_S": -0.1,
            "target_change_helpful_S": 0.1,
        }
    ]
    intervals = {
        "cap_shift_harmful_AO": {
            "estimate": 0.0,
            "ci_low": -0.1,
            "ci_high": 0.1,
            "ci90_low": -0.08,
            "ci90_high": 0.08,
        },
        "cap_shift_harmful_S": {"ci_low": -0.4, "ci_high": -0.1},
        "cap_shift_helpful_AO": {
            "estimate": 0.0,
            "ci_low": -0.1,
            "ci_high": 0.1,
            "ci90_low": -0.08,
            "ci90_high": 0.08,
        },
        "cap_shift_helpful_S": {"ci_low": 0.1, "ci_high": 0.5},
        "cap_social_harmful": {
            "estimate": -0.2,
            "ci_low": -0.4,
            "ci_high": 0.0,
            "ci90_low": -0.35,
            "ci90_high": -0.05,
        },
        "cap_social_helpful": {
            "estimate": 0.3,
            "ci_low": 0.1,
            "ci_high": 0.5,
            "ci90_low": 0.15,
            "ci90_high": 0.45,
        },
        "target_change_harmful_S": {"ci_low": -0.2, "ci_high": -0.01},
        "target_change_helpful_S": {"ci_low": 0.01, "ci_high": 0.2},
        "selectivity_change": {"ci_low": 0.1, "ci_high": 0.8},
    }
    robustness = {
        grouping: {
            metric: {
                "all_positive": metric
                in {"cap_shift_helpful_S", "target_change_helpful_S", "selectivity_change"},
                "all_negative": metric in {"cap_shift_harmful_S", "target_change_harmful_S"},
                "values": {"held_out": 0.1},
            }
            for metric in analysis.GATE_ROBUSTNESS_METRICS
        }
        for grouping in ("leave_one_domain_out", "leave_one_template_out")
    }
    stable = {"stable": True}
    assert analysis.decide_gate(rows, intervals, stable, robustness)["gate"] == "C"
    unstable = {"stable": False}
    assert analysis.decide_gate(rows, intervals, unstable, robustness)["gate"] == "E"


def test_gate_a_requires_ci_and_leave_one_out_robustness() -> None:
    rows = [
        {
            "cap_shift_harmful_S": 0.3,
            "cap_shift_helpful_S": 0.4,
            "cap_shift_harmful_AO": 0.0,
            "cap_shift_helpful_AO": 0.0,
            "cap_social_harmful": 0.3,
            "cap_social_helpful": 0.4,
            "target_change_harmful_S": 0.0,
            "target_change_helpful_S": 0.0,
        }
    ]
    intervals = {
        metric: {
            "estimate": 0.3,
            "ci_low": 0.1,
            "ci_high": 0.5,
            "ci90_low": 0.15,
            "ci90_high": 0.45,
        }
        for metric in analysis.GATE_ROBUSTNESS_METRICS
    }
    robustness = {
        grouping: {
            metric: {
                "all_positive": True,
                "all_negative": False,
                "values": {"held_out": 0.3},
            }
            for metric in analysis.GATE_ROBUSTNESS_METRICS
        }
        for grouping in ("leave_one_domain_out", "leave_one_template_out")
    }
    stable = {"stable": True}
    assert analysis.decide_gate(rows, intervals, stable, robustness)["gate"] == "A"

    intervals["cap_shift_harmful_S"]["ci_low"] = -0.01
    assert analysis.decide_gate(rows, intervals, stable, robustness)["gate"] == "F"
    intervals["cap_shift_harmful_S"]["ci_low"] = 0.1
    robustness["leave_one_domain_out"]["cap_shift_helpful_S"]["all_positive"] = False
    assert analysis.decide_gate(rows, intervals, stable, robustness)["gate"] == "F"


def test_gate_robustness_reports_every_preregistered_metric() -> None:
    rows = []
    for domain, template, value in (("d1", "t1", 1.0), ("d2", "t2", 2.0)):
        row = {
            "domain": domain,
            "template_id": template,
            **{metric: value for metric in analysis.GATE_ROBUSTNESS_METRICS},
        }
        rows.append(row)
    report = analysis.gate_robustness(rows)
    assert set(report) == {"leave_one_domain_out", "leave_one_template_out"}
    assert set(report["leave_one_domain_out"]) == set(analysis.GATE_ROBUSTNESS_METRICS)
    assert report["leave_one_template_out"]["selectivity_change"]["all_positive"] is True
