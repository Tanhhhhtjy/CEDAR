from __future__ import annotations

import functools
import hashlib
from collections import Counter
from pathlib import Path

from scripts import analyze_c1_v7_1_completion_addendum as analysis


@functools.lru_cache(maxsize=1)
def frozen_inputs() -> dict:
    return analysis.load_and_validate_inputs()


def directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_frozen_git_commits_and_result_hashes() -> None:
    loaded = frozen_inputs()
    provenance = loaded["input_provenance"]
    assert analysis.CUE_COMMIT == "1336f9dda660876dcfccb39842569eb93946c8f3"
    assert analysis.NO_CUE_COMMIT == "deb4932318b11906c1008849e8c97ec960fb1d90"
    for label, expected in analysis.EXPECTED_RESULT_SHA256.items():
        assert provenance[label]["sha256"] == expected
        assert len(provenance[label]["git_blob_id"]) == 40


def test_all_144_pairs_are_exhaustive_and_prompt_ablation_is_exact() -> None:
    pairs = frozen_inputs()["pairs"]
    assert len(pairs) == 144
    assert Counter(pair["arm"] for pair in pairs) == {"N": 48, "AO": 48, "S": 48}
    assert len({pair["family_id"] for pair in pairs}) == 24
    for pair in pairs:
        records = pair["records"]
        assert (
            records["cue_uncapped"]["prompt"].replace(analysis.CUE_SENTENCE, "")
            == records["no_cue_uncapped"]["prompt"]
        )
        for scoring in analysis.SCORINGS:
            for record in records.values():
                assert record[f"{scoring}_category"] in analysis.CATEGORIES


def test_no_cue_s_frozen_counts_and_complete_case_transitions() -> None:
    pairs = frozen_inputs()["pairs"]
    s_pairs = [pair for pair in pairs if pair["arm"] == "S"]
    uncapped = Counter(
        analysis.category_for(pair, "no_cue", "uncapped", "strict") for pair in s_pairs
    )
    p25 = Counter(
        analysis.category_for(pair, "no_cue", "p25", "strict") for pair in s_pairs
    )
    assert uncapped == {"target": 23, "initial": 19, "invalid_or_truncated": 6}
    assert p25 == {"target": 26, "initial": 22}
    report, rows = analysis.complete_case_analysis(pairs, samples=50)
    assert len(rows) == 42
    assert report["uncapped_target_n"] == 23
    assert report["p25_target_n"] == 21
    assert report["initial_to_target"] == 7
    assert report["target_to_initial"] == 9


def test_missingness_bound_formula_and_frozen_points() -> None:
    formula = analysis.partial_identification_bounds(
        n=48, uncapped_target=23, uncapped_invalid=6, p25_target=26
    )
    assert abs(formula["lower"] + 0.0625) < 1e-12
    assert abs(formula["upper"] - 0.0625) < 1e-12
    bounds = analysis.analyze_missingness_bounds(frozen_inputs()["pairs"], samples=50)
    no_cue = bounds["strict"]["point_bounds"]["no_cue"]
    cue = bounds["strict"]["point_bounds"]["cue"]
    assert abs(no_cue["lower"] + 0.0625) < 1e-12
    assert abs(no_cue["upper"] - 0.0625) < 1e-12
    assert abs(cue["lower"] - 5 / 48) < 1e-12
    assert abs(cue["upper"] - 9 / 48) < 1e-12


def test_family_bootstrap_is_deterministic() -> None:
    pairs = frozen_inputs()["pairs"]

    def statistic(rows: list[dict]) -> float:
        selected = [pair for pair in rows if pair["arm"] == "S"]
        return sum(
            analysis.category_for(pair, "no_cue", "p25", "strict") == "target"
            for pair in selected
        ) / len(selected)

    first = analysis.family_bootstrap(pairs, statistic, samples=100, seed=20260722)
    second = analysis.family_bootstrap(pairs, statistic, samples=100, seed=20260722)
    assert first == second


def test_runtime_thinking_and_phase_token_accounting() -> None:
    pairs = frozen_inputs()["pairs"]
    for pair in pairs:
        for record in pair["records"].values():
            thinking = record["boundary"]["thinking_tokens"]
            assert isinstance(thinking, int)
            assert 0 <= thinking <= record["response_tokens"]
            prefill = analysis.aggregate_reference_exposure(record, "prefill")
            decode = analysis.aggregate_reference_exposure(record, "decode")
            assert prefill["count"] == record["prompt_tokens"] * len(analysis.LAYERS)
            assert decode["count"] == (record["response_tokens"] - 1) * len(
                analysis.LAYERS
            )
            if record["condition"] == "p25":
                actual = analysis.aggregate_actual_cap(record)
                expected = (
                    record["prompt_tokens"] + record["response_tokens"] - 1
                ) * len(analysis.LAYERS)
                assert actual["count"] == expected

    truncated = [
        pair
        for pair in pairs
        if pair["arm"] == "S"
        and pair["records"]["no_cue_uncapped"]["truncated"]
    ]
    assert len(truncated) == 6
    assert all(
        analysis.length_value(
            pair["records"]["no_cue_uncapped"], "thinking_tokens"
        )
        == 2048
        for pair in truncated
    )
    roundtrip_failures = [
        record
        for pair in pairs
        for record in pair["records"].values()
        if not record["boundary"]["tokenization_roundtrip"]
        and not record["truncated"]
    ]
    assert len(roundtrip_failures) == 5
    assert all(
        analysis.length_value(record, "thinking_tokens") > 0
        for record in roundtrip_failures
    )
    assert all(analysis.final_token_count(record) > 0 for record in roundtrip_failures)


def test_repeated_small_analysis_has_identical_output_hashes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    analysis.analyze(
        output_root=first,
        bootstrap_samples=20,
        require_committed_script=False,
    )
    analysis.analyze(
        output_root=second,
        bootstrap_samples=20,
        require_committed_script=False,
    )
    assert directory_hashes(first) == directory_hashes(second)
