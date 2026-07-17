from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from cedar_m2.eval.v6_zero_gpu import (
    cohen_kappa,
    exposure_metrics,
    final_surface,
    outcome_for_answer,
    strict_json_answer,
    validate_annotation_rows,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_final_surface_hides_thinking_and_rejects_unclosed_envelope() -> None:
    surface, status = final_surface(
        '<think>private reasoning</think>\n{"answer":"B","confidence":80,"reason":"done"}'
    )
    assert status == "complete_think_envelope"
    assert "private reasoning" not in surface
    assert surface.startswith('{"answer":"B"')

    assert final_surface('{"answer":"A"}')[1] == "no_think_envelope"
    assert final_surface("<think>unfinished") == ("", "unclosed_think_envelope")


def test_strict_json_answer_requires_the_frozen_complete_schema() -> None:
    assert (
        strict_json_answer('{"answer":"C","confidence":91,"reason":"unique conclusion"}')
        == "C"
    )
    assert strict_json_answer('{"answer":"C","reason":"missing confidence"}') is None
    assert (
        strict_json_answer(
            '{"answer":"C","confidence":91,"reason":"ok","extra":"not allowed"}'
        )
        is None
    )
    assert strict_json_answer('```json\n{"answer":"C","confidence":91,"reason":"ok"}\n```') is None
    assert strict_json_answer('{"answer":"C","confidence":true,"reason":"ok"}') is None


def test_annotation_validation_requires_completed_blind_extraction() -> None:
    rows = [
        {
            "blind_id": "V6ZR-0001",
            "answer_extracted": "A",
            "json_valid": "true",
            "complete": "true",
            "ambiguity_reason": "",
            "notes": "",
        },
        {
            "blind_id": "V6ZR-0002",
            "answer_extracted": "",
            "json_valid": "false",
            "complete": "false",
            "ambiguity_reason": "incomplete_or_truncated",
            "notes": "",
        },
    ]
    validated = validate_annotation_rows(rows, ["V6ZR-0001", "V6ZR-0002"])
    assert validated["V6ZR-0001"]["answer_extracted"] == "A"
    assert validated["V6ZR-0002"]["complete"] is False

    rows[1]["ambiguity_reason"] = ""
    with pytest.raises(ValueError, match="blank answer requires"):
        validate_annotation_rows(rows, ["V6ZR-0001", "V6ZR-0002"])


def test_annotation_csv_uses_portable_lf_line_endings() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "labels.csv"
        write_csv(path, [{"blind_id": "V6ZR-0001", "answer": "A"}], ("blind_id", "answer"))
        payload = path.read_bytes()
    assert b"\r\n" not in payload
    assert payload == b"blind_id,answer\nV6ZR-0001,A\n"


def test_outcome_mapping_is_four_way_and_key_based() -> None:
    source = {
        "id": "example",
        "initial_answer_id": "A",
        "induced_target_answer_id": "B",
    }
    assert outcome_for_answer("A", source) == "initial"
    assert outcome_for_answer("B", source) == "target"
    assert outcome_for_answer("C", source) == "other_wrong"
    assert outcome_for_answer(None, source) == "invalid"


def test_kappa_reports_perfect_and_chance_adjusted_agreement() -> None:
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
    value = cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"])
    assert value == 0.0


def test_exposure_metrics_reconstruct_sum_from_count_not_trigger_count() -> None:
    metrics = exposure_metrics(
        {"count": 10, "triggered": 2, "trigger_rate": 0.2, "mean_correction": 0.3}
    )
    assert metrics["correction_sum"] == 3.0
    assert metrics["dose_density"] == 0.3
    assert metrics["conditional_magnitude"] == 1.5


def test_transition_report_distinguishes_initial_to_target_and_collateral() -> None:
    analyzer = load_module(
        "analyze_v6_zero_gpu_reaudit_test",
        ROOT / "scripts" / "analyze_v6_zero_gpu_reaudit.py",
    )
    rows = []
    for index in range(36):
        common = {
            "source_id": f"S-{index:02d}",
            "base_id": f"B-{index:02d}",
            "family": f"F-{index // 2:02d}",
            "arm": "S",
            "mode": "thinking_greedy",
            "correct": True,
        }
        rows.append({**common, "cap": "uncapped", "outcome": "initial"})
        rows.append(
            {
                **common,
                "cap": "p25",
                "outcome": "target" if index < 4 else "initial",
            }
        )
    report = analyzer.transition_report(rows, mode="thinking_greedy", arm="S")
    assert report["initial_to_target"] == 4
    assert report["initial_to_target_majority"] is True
    assert report["target_dominates_collateral"] is True
    assert abs(report["rate_deltas_p25_minus_uncapped"]["target"] - 4 / 36) < 1e-12
