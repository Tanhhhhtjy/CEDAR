from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from cedar_m2.interventions.axis_monitor import AssistantAxisMonitor
from cedar_m2.interventions.capping import ActivationCapper, LayerCap
from scripts import analyze_e_minus1b_philpapers as analyzer
from scripts import run_e_minus1b_philpapers as runner

ROOT = Path(__file__).resolve().parents[1]


class _PassLayer(nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_PassLayer()])


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0](value)[0]


def test_protocol_and_runner_freeze_official_p25_contract() -> None:
    protocol = (ROOT / "docs" / "E_MINUS1B_PHILPAPERS_CAP_AUDIT.md").read_text(
        encoding="utf-8"
    )
    assert "layers_46:54-p0.25" in protocol
    assert "trigger_rate * mean_correction" in protocol
    assert runner.CAP_EXPERIMENT == "layers_46:54-p0.25"
    assert runner.LAYERS == tuple(range(46, 54))
    assert runner.FORMAL_CONDITIONS == ("uncapped_monitor", "p25_cap")
    assert runner.EXPECTED_FORMAL_FORWARDS == 1800
    assert runner.EXPECTED_SENTINEL_FORWARDS == 24


def test_monitor_exposes_unambiguous_correction_fields() -> None:
    model = _DummyModel()
    axis = torch.tensor([[1.0, 0.0]])
    caps = {0: LayerCap(0, torch.tensor([1.0, 0.0]), 1.0)}
    monitor = AssistantAxisMonitor(
        model,
        axis,
        layers=(0,),
        prefill_attention_mask=torch.tensor([[1, 1]]),
        reference_caps={"p25": caps},
    )
    with monitor:
        model(torch.tensor([[[2.0, 0.0], [0.5, 0.0]]]))
    exposure = monitor.materialize()["phases"]["prefill"]["reference_cap_exposure"][
        "p25"
    ]["0"]
    assert exposure == {
        "count": 2,
        "triggered": 1,
        "trigger_rate": 0.5,
        "correction_sum": 1.0,
        "mean_correction_all_positions": 0.5,
        "mean_correction_triggered_positions": 1.0,
        "mean_correction": 0.5,
    }


def test_monitor_before_capper_matches_actual_same_layer_stats(monkeypatch) -> None:
    monkeypatch.setattr(runner, "LAYERS", (0,))
    model = _DummyModel()
    axis = torch.tensor([[1.0, 0.0]])
    caps = {0: LayerCap(0, torch.tensor([1.0, 0.0]), 1.0)}
    value = torch.tensor([[[2.0, 3.0], [0.5, 4.0]]])
    monitor = AssistantAxisMonitor(
        model,
        axis,
        layers=(0,),
        prefill_attention_mask=torch.tensor([[1, 1]]),
        reference_caps={"p25": caps},
    )
    capper = ActivationCapper(model, caps, positions="all", track_stats=True)
    with monitor:
        with capper:
            output = model(value)
    torch.testing.assert_close(output, torch.tensor([[[1.0, 3.0], [0.5, 4.0]]]))
    telemetry = monitor.materialize()
    actual = runner.cap_stats_to_dict(capper.stats)
    audit = runner.compare_monitor_and_cap(telemetry, actual)
    assert audit["status"] == "passed"
    assert audit["registration_order"] == ["AssistantAxisMonitor", "ActivationCapper"]
    assert actual["pooled"]["correction_sum"] == 1.0


def _source_row(question_id: int, variant: str) -> dict:
    return {
        "dataset_version": "e_minus1a_philpapers_v1",
        "sample_seed": 42,
        "pair_id": "pair0",
        "base_id": 0,
        "orig_id": 99,
        "question_id": question_id,
        "variant": variant,
        "split": "test",
        "sycophantic_answer": "A" if variant == "original" else "B",
        "non_sycophantic_answer": "B" if variant == "original" else "A",
    }


def _result(source: dict, condition: str, margin: float, exposure: float) -> dict:
    sycophantic = source["sycophantic_answer"]
    predicted = sycophantic if margin >= 0 else source["non_sycophantic_answer"]
    pooled = {
        "total_positions": 16,
        "triggered_positions": int(exposure > 0),
        "trigger_rate": 1 / 16 if exposure > 0 else 0.0,
        "correction_sum": exposure,
        "mean_correction_all_positions": exposure / 16,
        "mean_correction_triggered_positions": exposure if exposure > 0 else 0.0,
        "row_any_trigger": exposure > 0,
        "triggered_layers": [46] if exposure > 0 else [],
    }
    return {
        **source,
        "condition": condition,
        "predicted_forced_choice": predicted,
        "sycophancy_margin": margin,
        "p_syc_within_AB": 0.5,
        "AB_mean_logit": 1.0 if condition == "uncapped_monitor" else 0.75,
        "AB_probability_mass": 0.1 if condition == "uncapped_monitor" else 0.08,
        "full_vocab_entropy": 2.0 if condition == "uncapped_monitor" else 2.2,
        "full_vocab_top1_probability": 0.3,
        "full_vocab_top1_token_id": 100 if condition == "uncapped_monitor" else 101,
        "monitor_p25_exposure": {"pooled": pooled},
        "actual_cap_stats": {"pooled": pooled} if condition == "p25_cap" else None,
    }


def test_pair_builder_averages_original_swapped_after_row_contrast(monkeypatch) -> None:
    monkeypatch.setattr(analyzer, "EXPECTED_TEST_PAIRS", 1)
    sources = [_source_row(1, "original"), _source_row(2, "swapped")]
    rows = []
    for source in sources:
        rows.append(_result(source, "uncapped_monitor", 1.0, 4.0))
        rows.append(_result(source, "p25_cap", 0.25, 3.0))
    pair = analyzer.build_pair_results(sources, rows)[0]
    assert pair["delta_sycophancy_margin"] == -0.75
    assert pair["counterfactual_correction_sum"] == 4.0
    assert pair["actual_correction_sum"] == 3.0
    assert pair["sycophantic_to_non_sycophantic_rate"] == 0.0
    assert pair["top_token_changed_rate"] == 1.0


def test_frozen_dataset_remains_900_test_rows_and_450_pairs() -> None:
    rows = runner.validate_formal_rows(runner.DATA_PATH)
    assert len(rows) == 900
    assert len({(row["sample_seed"], row["pair_id"]) for row in rows}) == 450
    assert {
        row["sample_seed"] for row in rows
    } == {42, 7, 123}
    assert runner.EXPECTED_DATA_SHA256 == analyzer.EXPECTED_DATA_SHA256


def test_prepare_configuration_does_not_request_generation() -> None:
    source = (ROOT / "scripts" / "run_e_minus1b_philpapers.py").read_text(encoding="utf-8")
    assert ".generate(" not in source
    assert '"generated_text": False' in source
    assert '"thinking_enabled": False' in source
    json.dumps({"conditions": runner.FORMAL_CONDITIONS})
