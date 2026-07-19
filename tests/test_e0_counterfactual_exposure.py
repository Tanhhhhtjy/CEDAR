from __future__ import annotations

from pathlib import Path

from scripts import analyze_e0_counterfactual_exposure as e0

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_git_inputs_have_exact_hashes_and_coverage() -> None:
    sources = e0.load_sources()
    manifest = e0.input_manifest(sources)
    assert manifest["status"] == "passed"
    assert manifest["counts"] == {
        "dataset_rows": 144,
        "formal_result_rows": 1152,
        "strict_score_rows": 1152,
        "lenient_score_rows": 1152,
    }


def test_uncapped_counterfactual_telemetry_supports_coarse_E0_only() -> None:
    availability = e0.telemetry_availability(e0.load_sources())
    assert availability["status"] == "passed_with_coarse_phase_limit"
    assert availability["uncapped_records_audited"] == 576
    assert availability["phase_layer_records_audited"] == 576 * 2 * 8
    assert availability["uncapped_p25_counterfactual_exposure_complete"] is True
    assert availability["not_available"]["token_position_exposure_arrays"] is True
    assert availability["not_available"]["answer_logits_or_initial_margin"] is True


def test_transition_classifier_is_exhaustive_without_losing_other_wrong_paths() -> None:
    assert e0.classify_transition("initial", "target", False) == "initial_to_target"
    assert e0.classify_transition("target", "initial", False) == "target_to_initial"
    assert e0.classify_transition("initial", "other_wrong", False) == "initial_to_other_wrong"
    assert e0.classify_transition("target", "other_wrong", False) == "target_to_other_wrong"
    assert e0.classify_transition("initial", "initial", False) == "stable_initial"
    assert e0.classify_transition("target", "target", False) == "stable_target"
    assert e0.classify_transition("invalid", "target", False) == "invalid_or_truncated"
    assert e0.classify_transition("initial", "initial", True) == "invalid_or_truncated"
    assert (
        e0.classify_transition("other_wrong", "target", False)
        == "other_transition:other_wrong_to_target"
    )


def test_reconstructed_correction_sum_is_mean_times_count_only() -> None:
    sources = e0.load_sources()
    row = sources["results"][("thinking_greedy", "uncapped")][0]
    raw = row["axis_telemetry"]["phases"]["decode"]["reference_cap_exposure"]["p25"][
        "52"
    ]
    reconstructed = e0._layer_exposure(row, "decode", 52)
    assert reconstructed["correction_sum"] == raw["mean_correction"] * raw["count"]
    assert reconstructed["trigger_rate"] == raw["triggered"] / raw["count"]
    assert reconstructed["correction_sum"] != (
        raw["mean_correction"] * raw["count"] * raw["trigger_rate"]
    )


def test_primary_transition_counts_match_frozen_strict_scores() -> None:
    sources = e0.load_sources()
    transitions = e0.build_transitions(sources)
    selected = [
        row for row in transitions if row["mode"] == "thinking_greedy" and row["arm"] == "S"
    ]
    assert sum(row["strict"]["transition"] == "initial_to_target" for row in selected) == 5
    assert sum(row["strict"]["transition"] == "stable_initial" for row in selected) == 21
    assert sum(row["strict"]["transition"] == "target_to_initial" for row in selected) == 1
    assert sum(row["strict"]["transition"] == "invalid_or_truncated" for row in selected) == 1


def test_primary_family_cluster_contrast_uses_all_frozen_phase_groups() -> None:
    sources = e0.load_sources()
    transitions = e0.build_transitions(sources)
    exposures = e0.build_exposures(sources, transitions)
    report = e0.family_cluster_contrast(
        transitions,
        exposures,
        mode="thinking_greedy",
        arm="S",
        replicates=100,
    )
    assert report["status"] == "estimated"
    assert report["left_n"] == 5
    assert report["right_n"] == 21
    assert report["family_clusters"] == 18
    assert set(report["features"]) == set(e0.feature_names())


def test_E0_analyzer_has_no_model_or_cuda_execution_path() -> None:
    source = (ROOT / "scripts" / "analyze_e0_counterfactual_exposure.py").read_text(
        encoding="utf-8"
    )
    assert "import torch" not in source
    assert ".generate(" not in source
    assert "load_qwen_model" not in source
    assert '"gpu_or_model_execution": False' in source
