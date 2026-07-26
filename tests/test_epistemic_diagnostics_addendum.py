from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_epistemic_diagnostics_addendum.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("epistemic_diagnostics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis_module() -> Any:
    module = load_module()
    module.BOOTSTRAP_DRAWS = 200
    return module


@pytest.fixture(scope="module")
def rows(analysis_module: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        analysis_module.read_jsonl(analysis_module.E2A_PATH),
        analysis_module.read_jsonl(analysis_module.E2C_PATH),
    )


def test_frozen_input_audit_passes(analysis_module: Any, rows: Any) -> None:
    e2a, e2c = rows
    audit = analysis_module.validate_inputs(e2a, e2c)
    assert audit["status"] == "passed"
    assert audit["validity_answer_pairs"] == 192
    assert audit["missing_answer_logits"]["E2A"] == []
    assert audit["missing_answer_logits"]["E2C"] == ["CEDAR-E2C-022-F"]


def test_validity_choice_pairs_are_complete(analysis_module: Any, rows: Any) -> None:
    e2a, _ = rows
    result, pairs = analysis_module.analyze_validity_choice(e2a)
    assert result["pairs"] == 192
    assert len(pairs) == 192
    assert {row["arm"] for row in pairs} == {"V", "F"}
    assert {row["condition"] for row in pairs} == {
        "uncapped",
        "full_p25",
        "prefill_only",
        "decode_only",
    }


def test_sdt_mapping_and_direction(analysis_module: Any, rows: Any) -> None:
    e2a, e2c = rows
    datasets = {
        "E2A": [row for row in e2a if row["assay"] == "answer"],
        "E2C": e2c,
    }
    result, _ = analysis_module.analyze_sdt(datasets)
    e2a_full = result["datasets"]["E2A"]["full_p25"][
        "change_cap_minus_uncapped"
    ]
    e2c_full = result["datasets"]["E2C"]["full_p25"][
        "change_cap_minus_uncapped"
    ]
    assert e2a_full["criterion"]["estimate"] < 0
    assert e2c_full["d_prime"]["estimate"] < 0


def test_probability_roles_conserve_option_mass(analysis_module: Any, rows: Any) -> None:
    e2a, e2c = rows
    datasets = {
        "E2A": [row for row in e2a if row["assay"] == "answer"],
        "E2C": e2c,
    }
    _, base_rows, _ = analysis_module.analyze_probability_flow(datasets)
    assert base_rows
    for row in base_rows:
        arm_total = row["arm_p_initial"] + row["arm_p_target"] + row["arm_p_background"]
        r_total = row["r_p_initial"] + row["r_p_target"] + row["r_p_background"]
        assert arm_total == pytest.approx(1.0, abs=1e-10)
        assert r_total == pytest.approx(1.0, abs=1e-10)


def test_calibration_keeps_format_and_decision_uncertainty_separate(
    analysis_module: Any, rows: Any
) -> None:
    e2a, e2c = rows
    datasets = {
        "E2A": [row for row in e2a if row["assay"] == "answer"],
        "E2C": e2c,
    }
    result, _, _ = analysis_module.analyze_calibration(datasets)
    assert "conditional_entropy" in result["definitions"]
    assert "option_mass" in result["definitions"]
    assert "ece" in result["definitions"]
    missing = result["datasets"]["E2C"]["contrasts"]["full_p25"]["F"]["brier"][
        "missing_pair_ids"
    ]
    assert missing == ["CEDAR-E2C-022-F"]


def test_stable_offset_is_process_independent(analysis_module: Any) -> None:
    first = analysis_module.stable_offset("full_p25", "F", "validity_correct")
    second = analysis_module.stable_offset("full_p25", "F", "validity_correct")
    different = analysis_module.stable_offset("full_p25", "V", "validity_correct")
    assert first == second
    assert first != different
