from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from scripts import audit_c3_sycobench_bidirectional as audit
from scripts import build_c3_sycobench_bidirectional as builder
from scripts import run_c3_sycobench_bidirectional as runner


class FakeTokenizer:
    def encode(self, text, **_kwargs):
        if text == runner.ASSISTANT_PREFIX:
            return [90, 91, 92]
        return [ord(character) for character in text]

    def decode(self, ids, **_kwargs):
        return f"token-{ids[0]}"


def test_phase_resolution_and_missing_preflight_block(monkeypatch, tmp_path) -> None:
    assert runner.resolve_spec("smoke", "nohook").phase_name == "smoke-nohook"
    assert runner.resolve_spec("smoke", "uncapped").phase_name == "smoke-monitor"
    assert runner.resolve_spec("formal", "uncapped").phase_name == "formal-uncapped"
    assert runner.resolve_spec("formal", "p25").phase_name == "formal-p25"
    with pytest.raises(ValueError, match="only permits"):
        runner.resolve_spec("formal", "nohook")
    monkeypatch.setattr(runner, "PREFLIGHT_PATH", tmp_path / "missing-preflight.json")
    with pytest.raises(ValueError, match="execution_preflight_manifest"):
        runner.validate_execution_preflight()


def test_row_selection_preserves_frozen_grid() -> None:
    rows = builder.build(allow_network=False, write=False)["arms"]
    assert len(runner.select_rows(rows, runner.resolve_spec("formal", "uncapped"))) == 2400
    smoke = runner.select_rows(rows, runner.resolve_spec("smoke", "nohook"))
    assert len(smoke) == 96
    assert len({row["base_id"] for row in smoke}) == 24
    assert {(row["direction"], row["arm"]) for row in smoke} == {
        ("harmful", "AO"),
        ("harmful", "S"),
        ("helpful", "AO"),
        ("helpful", "S"),
    }


def test_runtime_revalidates_frozen_review_without_requiring_artifact_absence(
    monkeypatch, tmp_path
) -> None:
    gate = {
        "status": "passed",
        "gpu_authorized": True,
        "review_mode": "agent_self_review_user_waiver",
        "protocol_deviation": True,
        "human_review_performed": False,
        "data_lead_review_performed": False,
        "user_review_waived": True,
        "dual_independent_review": False,
        "cohen_kappa": None,
    }
    gate_path = tmp_path / "REVIEW_GATE.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    class FakeValidator:
        @staticmethod
        def load_builder():
            return object()

        @staticmethod
        def validate_review_gate(_builder, *, enforce_pre_artifact):
            assert enforce_pre_artifact is False
            return gate

    monkeypatch.setattr(runner, "REVIEW_GATE_PATH", gate_path)
    monkeypatch.setattr(runner, "load_validator", lambda: FakeValidator())
    assert runner.validate_review_gate() == gate


def test_run_order_blocks_skipping_preregistered_phases(tmp_path) -> None:
    with pytest.raises(ValueError, match="smoke-nohook"):
        runner.validate_run_order(runner.resolve_spec("smoke", "uncapped"), tmp_path)
    with pytest.raises(ValueError, match="SMOKE_AUDIT"):
        runner.validate_run_order(runner.resolve_spec("formal", "uncapped"), tmp_path)
    with pytest.raises(ValueError, match="formal uncapped"):
        runner.validate_run_order(runner.resolve_spec("formal", "p25"), tmp_path)


def test_four_option_diagnostics_include_full_vocab_integrity() -> None:
    logits = torch.full((100,), -10.0)
    logits[32] = 0.5
    logits[33] = 1.5
    logits[34] = 2.5
    logits[35] = 3.5
    row = {"initial_answer": "D", "target_answer": "A"}
    result = runner.choice_diagnostics(logits, row, FakeTokenizer())
    assert result["option_argmax"] == "D"
    assert result["target_minus_initial"] == -3.0
    assert set(result["logits"]) == set(runner.LABELS)
    assert 0.0 <= result["full_vocab_top_probability"] <= 1.0
    assert result["top_token_is_option"] is True


def synthetic_telemetry():
    phases = {}
    for phase, count in (
        ("conversation_prefill", 3),
        ("assistant_prefix_context", 2),
        ("answer_step", 1),
    ):
        phases[phase] = {
            "reference_cap_exposure": {
                "p25": {
                    str(layer): {
                        "count": count,
                        "triggered": count,
                        "trigger_rate": 1.0,
                        "correction_sum": float(count),
                        "mean_correction_all_positions": 1.0,
                    }
                    for layer in runner.LAYERS
                }
            }
        }
    return {"phases": phases}


def test_telemetry_summary_separates_prefill_and_answer_step() -> None:
    telemetry = synthetic_telemetry()
    uncapped = runner.telemetry_summary(telemetry, None, "uncapped")
    assert uncapped is not None
    assert uncapped["mode"] == "uncapped_counterfactual_p25"
    assert uncapped["prefill_trigger"]["total_positions"] == 5 * len(runner.LAYERS)
    assert uncapped["answer_step_trigger"]["total_positions"] == len(runner.LAYERS)
    actual = {
        phase: {
            str(layer): {
                "total_positions": payload["reference_cap_exposure"]["p25"][str(layer)]["count"],
                "triggered_positions": payload["reference_cap_exposure"]["p25"][str(layer)][
                    "triggered"
                ],
                "projection_correction_sum": payload["reference_cap_exposure"]["p25"][str(layer)][
                    "correction_sum"
                ],
            }
            for layer in runner.LAYERS
        }
        for phase, payload in telemetry["phases"].items()
    }
    capped = runner.telemetry_summary(telemetry, actual, "p25")
    assert capped is not None and capped["mode"] == "actual_p25"
    assert capped["trigger_rate_by_layer"]["53"] == 1.0


def valid_audit_record() -> dict:
    probabilities = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
    logits = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
    return {
        "initial_answer": "D",
        "target_answer": "A",
        "candidate_token_ids": runner.LABEL_TOKEN_IDS,
        "logit_A": 0.0,
        "logit_B": 1.0,
        "logit_C": 2.0,
        "logit_D": 3.0,
        "initial_logit": 3.0,
        "target_logit": 0.0,
        "target_initial_margin": -3.0,
        "option_argmax": "D",
        "full_vocab_top_token": "D",
        "full_vocab_top_probability": 0.4,
        "top_token_is_option": True,
        "answer_logits": {
            "logits": logits,
            "full_vocab_probabilities": probabilities,
            "conditional_probabilities": probabilities,
            "initial_logit": 3.0,
            "target_logit": 0.0,
            "target_minus_initial": -3.0,
            "option_argmax": "D",
            "option_mass": 1.0,
            "full_vocab_entropy": 1.0,
            "full_vocab_top_token_id": 35,
            "full_vocab_top_token_text": "D",
            "full_vocab_top_probability": 0.4,
            "top_token_is_option": True,
        },
    }


def test_auditor_validates_margin_aliases_and_probability_roundoff() -> None:
    record = valid_audit_record()
    assert audit.diagnostics_valid(record)
    changed = deepcopy(record)
    changed["target_initial_margin"] = 3.0
    assert not audit.diagnostics_valid(changed)
    changed = deepcopy(record)
    changed["answer_logits"]["option_mass"] = 1.0001
    assert not audit.diagnostics_valid(changed)


def test_mapping_validation_catches_reversed_direction() -> None:
    harmful = {
        "direction": "harmful",
        "gold_answer": "D",
        "foil_answer": "A",
        "initial_answer": "D",
        "target_answer": "A",
        "initial_position": "D",
        "target_position": "A",
    }
    assert audit.mapping_valid(harmful)
    harmful["target_answer"] = "D"
    assert not audit.mapping_valid(harmful)


def test_comparable_configuration_ignores_only_paired_condition_fields() -> None:
    left = {
        "kind": "C3",
        "phase": "formal-uncapped",
        "condition": "uncapped",
        "instrumentation": "uncapped",
        "dataset": {"arms_semantic_sha256": "x"},
        "model": {"revision": "r"},
    }
    right = deepcopy(left)
    right.update(phase="formal-p25", condition="p25", instrumentation="p25")
    assert audit.comparable_configuration(left) == audit.comparable_configuration(right)
    right["dataset"]["arms_semantic_sha256"] = "y"
    assert audit.comparable_configuration(left) != audit.comparable_configuration(right)
