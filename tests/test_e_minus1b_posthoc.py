from __future__ import annotations

import math
from pathlib import Path

from scripts import analyze_e_minus1b_posthoc as audit

ROOT = Path(__file__).resolve().parents[1]


def test_exact_mcnemar_matches_frozen_12_vs_25_counts() -> None:
    result = audit.exact_mcnemar(12, 25)
    expected = min(
        1.0,
        2 * sum(math.comb(37, value) for value in range(13)) / (2**37),
    )
    assert result["discordant"] == 37
    assert result["p_value"] == round(expected, 10)
    assert result["odds_ratio_non_to_syc_over_syc_to_non"] == round(25 / 12, 10)


def test_transition_names_cover_all_four_migrations() -> None:
    assert audit.transition_name(True, True) == "stable_sycophantic"
    assert audit.transition_name(True, False) == "sycophantic_to_non_sycophantic"
    assert audit.transition_name(False, True) == "non_sycophantic_to_sycophantic"
    assert audit.transition_name(False, False) == "stable_non_sycophantic"


def _telemetry_layer(correction: float, triggered: int = 2) -> dict:
    return {
        "count": 2,
        "triggered": triggered,
        "trigger_rate": triggered / 2,
        "correction_sum": correction,
    }


def _actual_layer(correction: float, triggered: int = 2) -> dict:
    return {
        "total_positions": 2,
        "triggered_positions": triggered,
        "trigger_rate": triggered / 2,
        "correction_sum": correction,
    }


def test_normalized_exposure_uses_correction_sum_without_double_trigger_weighting() -> None:
    row = {
        "prompt_token_count": 2,
        "axis_telemetry": {
            "phases": {
                "prefill": {
                    "reference_cap_exposure": {
                        "p25": {
                            str(layer): _telemetry_layer(float(layer - 45), triggered=1)
                            for layer in audit.LAYERS
                        }
                    }
                }
            }
        },
        "actual_cap_stats": {
            "by_layer": {
                str(layer): _actual_layer(float(layer - 45), triggered=1)
                for layer in audit.LAYERS
            }
        },
    }
    counterfactual = audit.normalized_exposure(row, actual=False)
    assert counterfactual["pooled"]["correction_sum"] == 36.0
    assert counterfactual["pooled"]["total_positions"] == 16
    assert counterfactual["pooled"]["trigger_rate"] == 0.5
    assert counterfactual["pooled"]["normalized_correction"] == 36 / 16
    assert counterfactual["pooled"]["mean_correction_per_token"] == 18.0
    assert counterfactual["layer_46"]["normalized_correction"] == 0.5
    assert counterfactual["layers_50_53"]["correction_sum"] == 26.0
    assert audit.normalized_exposure(row, actual=True)["pooled"] == counterfactual["pooled"]


def test_frozen_e_minus1b_inputs_pass_hash_and_pair_audit() -> None:
    rows = audit.read_jsonl(audit.SOURCE_DIR / "row_results.jsonl")
    result = audit.validate_inputs(audit.SOURCE_DIR, rows)
    assert result["status"] == "passed"
    assert result["counts"]["condition_rows"] == 1800
    assert result["counts"]["prompt_condition_pairs"] == 900
    assert result["counts"]["original_swapped_clusters"] == 450


def test_transition_builder_recovers_frozen_counts_and_position_unavailability() -> None:
    rows = audit.read_jsonl(audit.SOURCE_DIR / "row_results.jsonl")
    transitions, exposures = audit.build_transition_rows(rows)
    counts = {
        name: sum(row["transition"] == name for row in transitions)
        for name in audit.TRANSITIONS
    }
    assert counts == {
        "stable_sycophantic": 750,
        "sycophantic_to_non_sycophantic": 12,
        "non_sycophantic_to_sycophantic": 25,
        "stable_non_sycophantic": 113,
    }
    assert all(
        row["position_level_availability"]["last_input_token"] == "not_available"
        for row in exposures
    )


def test_posthoc_analyzer_has_no_model_or_cuda_execution_path() -> None:
    source = (ROOT / "scripts" / "analyze_e_minus1b_posthoc.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert ".generate(" not in source
    assert "load_qwen_model" not in source
    assert '"gpu_or_model_execution": False' in source
