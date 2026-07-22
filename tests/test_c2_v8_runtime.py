from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from cedar_m2.interventions.capping import ActivationCapper, LayerCap
from scripts import audit_c2_v8_natural_confirmation as audit
from scripts import build_c2_v8_natural_confirmation as builder
from scripts import run_c2_v8_natural_confirmation as runner


class FakeTokenizer:
    def encode(self, text, **_kwargs):
        if text == runner.ASSISTANT_PREFIX:
            return [90, 91, 92]
        return [ord(character) for character in text]

    def decode(self, ids, **_kwargs):
        return f"token-{ids[0]}"


def test_frozen_phase_resolution_and_row_selection() -> None:
    assert runner.validate_execution_preflight()["gpu_or_qwen_execution_performed"] is False
    rows, gate = runner.validate_inputs(require_review=False)
    assert gate == {"status": "not_checked"}
    assert len(rows) == 1200
    direct = runner.resolve_spec("direct", "formal", "uncapped")
    free = runner.resolve_spec("free", "formal", "p25")
    smoke = runner.resolve_spec("direct", "smoke", "nohook")
    assert direct.phase_name == "direct-uncapped"
    assert free.phase_name == "free-p25"
    assert smoke.phase_name == "smoke-direct-nohook"
    assert len(runner.select_rows(rows, direct)) == 1200
    assert len(runner.select_rows(rows, free)) == 480
    assert len(runner.select_rows(rows, smoke)) == 24
    with pytest.raises(ValueError, match="only permit"):
        runner.resolve_spec("direct", "formal", "nohook")


def test_generation_gate_discloses_single_review_protocol_deviation() -> None:
    gate = runner.validate_review_gate()
    assert gate["status"] == "passed"
    assert gate["gpu_authorized"] is True
    assert gate["protocol_deviation"] is True
    assert gate["raw_agreement"] is None
    assert gate["cohen_kappa"] is None


def test_run_order_blocks_skipping_smoke_and_uncapped(tmp_path) -> None:
    with pytest.raises(ValueError, match="prerequisite smoke-direct-nohook"):
        runner.validate_run_order(
            runner.resolve_spec("direct", "smoke", "uncapped"), tmp_path
        )
    with pytest.raises(ValueError, match="combined smoke audit"):
        runner.validate_run_order(
            runner.resolve_spec("direct", "formal", "uncapped"), tmp_path
        )
    with pytest.raises(ValueError, match="prerequisite direct-uncapped"):
        runner.validate_run_order(runner.resolve_spec("direct", "formal", "p25"), tmp_path)


def test_four_option_diagnostics_and_classification_include_d() -> None:
    logits = torch.full((100,), -10.0)
    logits[32] = 0.5
    logits[33] = 1.5
    logits[34] = 2.5
    logits[35] = 3.5
    row = {"initial_answer": "D", "target_answer": "A"}
    result = runner.choice_diagnostics(logits, row, FakeTokenizer())
    assert result["option_argmax"] == "D"
    assert result["target_minus_initial"] == -3.0
    assert set(result["logits"]) == {"A", "B", "C", "D"}
    assert runner.classify(row, "D", False) == "initial"
    assert runner.classify(row, "A", False) == "target"
    assert runner.classify(row, "C", False) == "other_wrong"
    assert runner.classify(row, "A", True) == "invalid_or_truncated"


def test_direct_and_free_chat_rendering_use_exact_frozen_qwen_template() -> None:
    tokenizer = builder.load_tokenizer()
    runner.validate_label_tokens(tokenizer)
    rows, _ = runner.validate_inputs(require_review=False)
    for row in rows:
        direct, direct_ids, conversation_tokens = runner.render_direct(tokenizer, row)
        assert direct.endswith(runner.ASSISTANT_PREFIX)
        assert "</think>" in direct
        assert conversation_tokens < len(direct_ids)
    free, free_ids = runner.render_free(tokenizer, rows[0])
    assert free.endswith("<|im_start|>assistant\n")
    assert not free.endswith(runner.ASSISTANT_PREFIX)
    assert free_ids == tokenizer.encode(free, add_special_tokens=False)


def test_free_parsers_are_strict_and_lenient_without_answer_repair() -> None:
    response = '<think>Reasoning.</think>\n{"answer":"D"}'
    assert runner.strict_answer(response, truncated=False) == ("D", None)
    assert runner.lenient_answer(response) == ("D", None)
    strict, error = runner.strict_answer(
        '<think>x</think>{"answer":"D","reason":"x"}', truncated=False
    )
    assert strict is None and error == "strict_json_has_extra_fields"
    assert runner.strict_answer(response, truncated=True) == (None, "truncated")


def test_free_logit_capture_keeps_only_exact_answer_prefix_steps() -> None:
    tokenizer = FakeTokenizer()
    row = {"initial_answer": "D", "target_answer": "A"}
    capture = runner.AnswerPrefixLogitCapture(tokenizer, row, prompt_tokens=5)
    scores = torch.full((1, 100), -10.0)
    scores[0, 32:36] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    untouched = capture(torch.tensor([[1, 2, 3, 4, 5, 90, 91, 92]]), scores)
    assert untouched is scores
    assert list(capture.by_response_index) == [3]
    assert capture.by_response_index[3]["target_minus_initial"] == 3.0
    capture(torch.tensor([[1, 2, 3, 4, 5, 90, 91, 0]]), scores)
    assert list(capture.by_response_index) == [3]


def test_actual_phase_stats_must_reconstruct_pooled_cap_stats() -> None:
    phases = {}
    for phase, count in (("prefill", 3), ("decode", 2)):
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
    telemetry = {"phases": phases}
    cap = {
        "by_layer": {
            str(layer): {
                "total_positions": 5,
                "triggered_positions": 5,
                "projection_correction_sum": 5.0,
            }
            for layer in runner.LAYERS
        }
    }
    split = runner.actual_cap_by_phase(telemetry, cap)
    assert split is not None
    assert split["decode"]["53"]["total_positions"] == 2
    broken = deepcopy(cap)
    broken["by_layer"]["46"]["projection_correction_sum"] = 4.0
    with pytest.raises(RuntimeError, match="layer 46"):
        runner.actual_cap_by_phase(telemetry, broken)


def test_segmented_direct_monitor_observes_pre_cap_values_and_splits_positions() -> None:
    class LayerStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(64)])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = LayerStack()

    model = FakeModel()
    axis = torch.zeros((64, 4))
    axis[:, 0] = 1.0
    caps = {
        layer: LayerCap(layer=layer, direction=torch.tensor([1.0, 0.0, 0.0, 0.0]), threshold=0.5)
        for layer in runner.LAYERS
    }
    masks = {
        "conversation_prefill": torch.tensor([[True, True, True, False, False]]),
        "answer_prefix": torch.tensor([[False, False, False, True, True]]),
    }
    monitor = runner.SegmentedAxisMonitor(model, axis, caps, masks)
    capper = ActivationCapper(model, caps, positions="all", track_stats=True)
    hidden = torch.ones((1, 5, 4))
    with monitor:
        with capper:
            for layer in runner.LAYERS:
                hidden = model.model.layers[layer](hidden)
    telemetry = monitor.materialize()
    stats = runner.cap_stats_to_dict(capper.stats)
    split = runner.actual_cap_by_phase(telemetry, stats)
    assert split is not None
    conversation = telemetry["phases"]["conversation_prefill"]
    prefix = telemetry["phases"]["answer_prefix"]
    assert conversation["reference_cap_exposure"]["p25"]["46"]["count"] == 3
    assert prefix["reference_cap_exposure"]["p25"]["46"]["count"] == 2
    assert stats["by_layer"]["46"]["total_positions"] == 5
    assert torch.allclose(hidden[..., 0], torch.full((1, 5), 0.5))


def test_auditor_pairs_four_option_logits_by_id() -> None:
    def record(item_id: str, offset: float):
        return {
            "id": item_id,
            "answer_logits": {
                "logits": {
                    "A": offset,
                    "B": offset + 1,
                    "C": offset + 2,
                    "D": offset + 3,
                }
            },
        }

    left = [record("b", 1.0), record("a", 2.0)]
    right = [record("a", 2.00001), record("b", 1.00002)]
    assert audit.max_logit_difference(left, right, ("answer_logits",)) < 1e-4


def test_comparable_configuration_ignores_only_paired_run_fields() -> None:
    left = {
        "kind": "C2",
        "scope": "formal",
        "track": "direct",
        "phase": "direct-uncapped",
        "condition": "uncapped",
        "instrumentation": "uncapped",
        "monitor_enabled": True,
        "selection": "all_arms",
        "seed": 0,
        "dataset": {
            "arms_content_sha256": "x",
            "human_review_gate_canonical_text_sha256": "a",
        },
        "model": {"revision": "r"},
        "measurement": {"content_retry": False},
        "axis_telemetry": {"layers": list(runner.LAYERS)},
        "capping": {"experiment": "p25"},
        "human_review": {"status": "passed"},
        "git": {"commit": "c"},
    }
    right = deepcopy(left)
    right.update(phase="direct-p25", condition="p25", instrumentation="p25")
    assert audit.comparable_configuration(left) == audit.comparable_configuration(right)
    right["dataset"]["human_review_gate_canonical_text_sha256"] = "b"
    assert audit.comparable_configuration(left) != audit.comparable_configuration(right)
