from __future__ import annotations

from scripts import analyze_c2_v8_natural_confirmation as analysis


def direct_record(initial: str, target: str, margin: float, argmax: str) -> dict:
    return {
        "answer_logits": {
            "target_minus_initial": margin,
            "option_argmax": argmax,
            "option_mass": 0.9,
            "full_vocab_entropy": 1.0,
            "top_token_is_option": True,
        }
    }


def synthetic_direct_bases() -> tuple[list[dict], list[dict]]:
    pairs = []
    for index in range(12):
        base = f"B{index:02d}"
        common = {
            "base_id": base,
            "source_id": f"source-{index}",
            "domain": f"D{index % 3}",
            "difficulty": ("easy", "medium", "hard")[index % 3],
            "normalized_stem_id": f"stem-{index // 2}",
            "exact_mcq_id": f"mcq-{index}",
            "template_id": index % 4,
            "initial_answer": "C",
            "target_answer": "A",
        }
        for arm in analysis.ARMS:
            delta = 0.1 if arm == "AO" else 0.4
            pairs.append(
                {
                    "id": f"{base}-{arm}",
                    **common,
                    "arm": arm,
                    "uncapped": direct_record("C", "A", -1.0, "C"),
                    "p25": direct_record("C", "A", -1.0 + delta, "A" if arm == "S" else "C"),
                }
            )
    return pairs, analysis.build_base_pairs(pairs)


def test_direct_social_margin_is_paired_within_base() -> None:
    pairs, bases = synthetic_direct_bases()
    assert len(bases) == 12
    assert all(abs(analysis.direct_social(base) - 0.3) < 1e-12 for base in bases)
    bootstrap = analysis.stratified_bootstrap(
        bases,
        analysis.mean_statistic(analysis.direct_social),
        samples=200,
        seed=7,
    )
    assert abs(bootstrap["estimate"] - 0.3) < 1e-12
    assert bootstrap["ci_low"] > 0
    transitions = analysis.transition_counts(pairs, "S", "direct")
    assert transitions["initial->target"] == 12


def test_missingness_bounds_keep_invalid_outcomes_unresolved() -> None:
    pairs = []
    for index, (left, right) in enumerate(
        [
            ("target", "target"),
            ("invalid_or_truncated", "target"),
            ("initial", "initial"),
            ("initial", "invalid_or_truncated"),
        ]
    ):
        pairs.append(
            {
                "id": str(index),
                "arm": "S",
                "uncapped": {"strict_category": left},
                "p25": {"strict_category": right},
            }
        )
    bounds = analysis.missingness_bounds(pairs, "S", "strict")
    assert bounds["lower"] == 0.0
    assert bounds["upper"] == 0.5


def test_reasoning_stance_is_external_and_conservative() -> None:
    assert analysis.reasoning_stance("<think>The correct answer is C.</think>{}") == "C"
    assert analysis.reasoning_stance("<think>No explicit label.</think>{}") is None
    assert analysis.reasoning_stance("missing tags") is None


def test_decision_matrix_respects_direct_primary_and_integrity() -> None:
    def interval(estimate, low, high):
        return {"estimate": estimate, "ci_low": low, "ci_high": high}
    direct = {
        "primary_social_margin_effect": interval(0.3, 0.1, 0.5),
        "arm_cap_effects": {
            "AO": interval(0.0, -0.1, 0.1),
            "S": interval(0.3, 0.1, 0.5),
        },
        "leave_one_domain_out": {"all_positive": True},
        "leave_one_template_out": {"all_positive": True},
        "output_integrity_social_interactions": {
            "option_mass": interval(0.0, -0.01, 0.01),
            "full_vocab_entropy": interval(0.0, -0.01, 0.01),
            "top_token_is_option": interval(0.0, -0.01, 0.01),
        },
    }
    free = {
        "scoring": {
            "strict": {
                "outcomes": {
                    "target": {"social_excess": interval(0.0, -0.1, 0.1)},
                    "completion": {"social_excess": interval(0.0, -0.1, 0.1)},
                    "thinking_tokens": {"social_excess": interval(0.0, -10, 10)},
                }
            }
        }
    }
    result = analysis.decision_matrix(direct, free)
    assert result["label"] == "robust_social_pressure_specific_margin_shift"
    direct["output_integrity_social_interactions"]["option_mass"] = interval(
        -0.2, -0.3, -0.1
    )
    result = analysis.decision_matrix(direct, free)
    assert result["label"] == "distribution_perturbation_not_clean_social_shift"
