from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn

from cedar_m2.interventions.axis_monitor import AssistantAxisMonitor
from cedar_m2.interventions.capping import LayerCap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "v6_factorial_diagnostic"


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_v6_formal_data_are_balanced_and_response_independent() -> None:
    rows = read_jsonl(DATA / "m2_2b_v6_factorial144.jsonl")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 144
    assert Counter(row["arm"] for row in rows) == {"N": 36, "V": 36, "F": 36, "S": 36}
    assert len({row["source_candidate_base_id"] for row in rows}) == 36
    assert len({row["template_family"] for row in rows}) == 18
    assert manifest["source"]["v5_model_outputs_read_by_builder"] is False
    assert manifest["selection"]["response_independent"] is True
    assert manifest["factorial"]["stage_1_new_formal_generations"] == 1728
    assert manifest["factorial"]["historical_anchor_is_primary_factorial_data"] is False

    by_family: dict[str, list[dict]] = defaultdict(list)
    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["template_family"]].append(row)
        by_base[row["source_candidate_base_id"]].append(row)
        assert row["selection_status"] == "mechanical_response_independent"
        assert row["recommended_batch_size"] == 1
    for family_rows in by_family.values():
        assert len({row["gradient_level"] for row in family_rows if row["arm"] in {"V", "F"}}) == 1
    for base_rows in by_base.values():
        assert {row["arm"] for row in base_rows} == {"N", "V", "F", "S"}
        assert len({row["gradient_level"] for row in base_rows if row["arm"] in {"V", "F"}}) == 1


def test_v6_axis_controls_are_paired_and_safe() -> None:
    rows = read_jsonl(DATA / "m2_2b_v6_axis_controls24.jsonl")
    assert len(rows) == 24
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
        assert row["safety_content"] == "safe identity elicitation only"
        assert row["recommended_batch_size"] == 1
    assert len(pairs) == 12
    for pair in pairs.values():
        assert {row["condition"] for row in pair} == {"default_assistant", "persona"}
        assert len({row["question"] for row in pair}) == 1


class _PassLayer(nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_PassLayer(), _PassLayer()])


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            value = layer(value)[0]
        return value


def test_axis_monitor_separates_prefill_decode_and_masks_padding() -> None:
    model = _DummyModel()
    axis = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    caps = {
        layer: LayerCap(layer=layer, direction=torch.tensor([1.0, 0.0]), threshold=0.5)
        for layer in (0, 1)
    }
    monitor = AssistantAxisMonitor(
        model,
        axis,
        layers=(0, 1),
        prefill_attention_mask=torch.tensor([[0, 1]]),
        reference_caps={"p25": caps},
    )
    with monitor:
        model(torch.tensor([[[9.0, 0.0], [1.0, 0.0]]]))
        model(torch.tensor([[[2.0, 0.0]]]))
    report = monitor.materialize()["phases"]
    for layer in ("0", "1"):
        assert report["prefill"]["assistant_axis_projection"][layer]["count"] == 1
        assert report["prefill"]["assistant_axis_projection"][layer]["mean"] == 1.0
        assert report["decode"]["assistant_axis_projection"][layer]["count"] == 1
        assert report["decode"]["assistant_axis_projection"][layer]["mean"] == 2.0
        exposure = report["prefill"]["reference_cap_exposure"]["p25"][layer]
        assert exposure["trigger_rate"] == 1.0
        assert exposure["correction_sum"] == 0.5
        assert exposure["mean_correction_all_positions"] == 0.5
        assert exposure["mean_correction_triggered_positions"] == 0.5
        assert exposure["mean_correction"] == 0.5


def test_v6_runner_freezes_modes_and_per_record_sampling() -> None:
    runner = load_module(
        "run_m2_2b_v6_factorial_diagnostic_test",
        ROOT / "scripts" / "run_m2_2b_v6_factorial_diagnostic.py",
    )
    rows = runner.load_rows(DATA / "m2_2b_v6_factorial144.jsonl", "formal")
    assert len(rows) == 144
    assert runner.MODES["thinking_sampling"].enable_thinking is True
    assert runner.MODES["nonthinking_sampling"].enable_thinking is False
    for name in ("thinking_sampling", "nonthinking_sampling"):
        mode = runner.MODES[name]
        assert (mode.temperature, mode.top_p, mode.top_k) == (0.6, 0.95, 20)
    assert runner.MAX_NEW_TOKENS == {"formal": 2048, "axis": 512}


def test_analyzer_parses_only_the_final_answer_json() -> None:
    analyzer = load_module(
        "analyze_m2_2b_v6_factorial_diagnostic_test",
        ROOT / "scripts" / "analyze_m2_2b_v6_factorial_diagnostic.py",
    )
    response = '<think>{"answer":"A"}</think>\n{"answer":"B","reason":"final"}'
    assert analyzer.parse_final_answer(response) == "B"
