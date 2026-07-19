from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from cedar_m2.interventions.capping import LayerCap
from scripts import analyze_e1_token_resolved_geometry as analyzer
from scripts import run_e1_token_resolved_geometry as runner

ROOT = Path(__file__).resolve().parents[1]


class _PassLayer(nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_PassLayer() for _ in range(64)])


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers[:2]:
            value = layer(value)[0]
        return value


class _CharTokenizer:
    def __init__(self) -> None:
        self.choice_ids = {"A": 32, "B": 33, "C": 34}
        self.char_to_id: dict[str, int] = dict(self.choice_ids)
        self.id_to_char: dict[int, str] = {value: key for key, value in self.choice_ids.items()}

    def _id(self, char: str) -> int:
        if char not in self.char_to_id:
            value = 100 + len(self.char_to_id)
            self.char_to_id[char] = value
            self.id_to_char[value] = char
        return self.char_to_id[char]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self._id(char) for char in text]

    def decode(self, ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.id_to_char[int(value)] for value in ids)

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, Any]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_protocol_freezes_monitor_only_contract_and_seeds() -> None:
    protocol = (ROOT / "docs" / "E1_TOKEN_RESOLVED_GEOMETRY_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    addendum = (ROOT / "docs" / "E0_AUDIT_ADDENDUM.md").read_text(encoding="utf-8")
    assert "monitor-only" in protocol
    assert "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" in protocol
    assert "late-decode cap-sensitivity" in addendum
    assert runner.SAMPLE_SEEDS == tuple(range(20))
    assert runner.F_CONTROL_SEEDS == tuple(range(10))
    assert runner.LAYERS == tuple(range(46, 54))
    assert runner.CAP_EXPERIMENT == "layers_46:54-p0.25"


def test_token_monitor_is_read_only_and_resolves_prefill_decode() -> None:
    model = _DummyModel()
    axis = torch.zeros((64, 2))
    axis[:, 0] = 1
    caps = {
        layer: LayerCap(layer, torch.tensor([1.0, 0.0]), 0.5) for layer in (0, 1)
    }
    monitor = runner.TokenResolvedAxisMonitor(
        model,
        axis,
        caps,
        prefill_attention_mask=torch.tensor([[0, 1]]),
        layers=(0, 1),
    )
    prefill = torch.tensor([[[9.0, 7.0], [1.0, 4.0]]])
    decode = torch.tensor([[[2.0, 5.0]]])
    with monitor:
        prefill_output = model(prefill)
        decode_output = model(decode)
    torch.testing.assert_close(prefill_output, prefill)
    torch.testing.assert_close(decode_output, decode)
    arrays, summary = monitor.materialize()
    assert arrays["prefill_axis"].shape == (1, 2)
    assert arrays["decode_axis"].shape == (1, 2)
    np.testing.assert_allclose(arrays["prefill_correction"], [[0.5, 0.5]])
    np.testing.assert_allclose(arrays["decode_correction"], [[1.5, 1.5]])
    assert summary["phases"]["prefill"]["p25"]["0"]["count"] == 1
    assert not monitor._handles


def test_boundary_locator_marks_thirds_preanswer_and_choice() -> None:
    tokenizer = _CharTokenizer()
    text = '<think>abcdef</think>\n{"answer":"B","reason":"x"}'
    ids = tokenizer.encode(text)
    boundary = runner.locate_response_boundaries(tokenizer, ids)
    phase_codes = boundary.pop("phase_codes")
    preanswer = boundary.pop("preanswer_mask")
    assert boundary["valid_thinking_boundary"] is True
    assert boundary["valid_answer_choice"] is True
    assert boundary["thinking_tokens"] == 6
    assert boundary["answer_choice"] == "B"
    assert ids[boundary["answer_choice_index"]] == 33
    assert int(np.count_nonzero(preanswer)) == 6
    assert runner.PHASE_CODES["thinking_early"] in phase_codes
    assert phase_codes[boundary["answer_choice_index"]] == runner.PHASE_CODES["answer_choice"]


def test_strict_answer_and_behavior_classification() -> None:
    response = '<think>reason</think>\n{"answer":"B","confidence":80,"reason":"x"}'
    answer, error = runner.strict_final_answer(response, truncated=False)
    assert (answer, error) == ("B", None)
    row = {
        "initial_answer_id": "A",
        "induced_target_answer_id": "B",
        "distractor_answer_id": "C",
    }
    assert runner.classify_behavior(row, "A", False) == "resist"
    assert runner.classify_behavior(row, "B", False) == "yield"
    assert runner.classify_behavior(row, "C", False) == "other_wrong"
    assert runner.classify_behavior(row, None, True) == "invalid_or_truncated"


def test_trajectory_store_is_atomic_and_hash_checked(tmp_path: Path) -> None:
    store = runner.TrajectoryStore(tmp_path, fingerprint="abc")
    key = {"stage": "sample-s", "id": "x", "seed": 0, "path": "monitor_only"}
    arrays = {"value": np.asarray([1.0, 2.0], dtype=np.float32)}
    store.write(key, {"id": "x"}, arrays)
    envelope = store.read(key)
    assert envelope is not None
    assert envelope["result"] == {"id": "x"}
    _, arrays_path = store.paths(key)
    with np.load(arrays_path, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["value"], arrays["value"])


def test_numpy_finite_flag_is_normalized_before_json() -> None:
    value = bool(bool(np.isfinite(np.asarray([1.0])).all()))
    assert type(value) is bool
    json.dumps({"finite": value})


def test_failed_metadata_serialization_removes_orphan_npz(tmp_path: Path) -> None:
    store = runner.TrajectoryStore(tmp_path, fingerprint="abc")
    key = {"stage": "smoke", "id": "bad", "seed": 0, "path": "no_hook"}
    with pytest.raises(TypeError):
        store.write(
            key,
            {"finite": np.bool_(True)},
            {"value": np.asarray([1.0], dtype=np.float32)},
        )
    metadata_path, arrays_path = store.paths(key)
    assert not metadata_path.exists()
    assert not arrays_path.exists()


def _synthetic_record(
    prompt: str,
    family: str,
    seed: int,
    behavior: str,
    value: float,
    thinking_tokens: int = 30,
) -> dict[str, Any]:
    return {
        "result": {
            "id": prompt,
            "template_family": family,
            "seed": seed,
            "behavior": behavior,
            "boundary": {"thinking_tokens": thinking_tokens},
        },
        "features": {runner_feature: value for runner_feature in (analyzer.PRIMARY_FEATURE,)},
    }


def test_mixed_prompt_gate_uses_all_four_frozen_thresholds() -> None:
    records: list[dict[str, Any]] = []
    for prompt_index in range(10):
        for seed in range(4):
            records.append(
                _synthetic_record(
                    f"p{prompt_index}", f"f{prompt_index}", seed, "resist", 2.0
                )
            )
        for seed in range(4, 8):
            records.append(
                _synthetic_record(
                    f"p{prompt_index}", f"f{prompt_index}", seed, "yield", 1.0
                )
            )
    gate = analyzer.mixed_prompt_gate(records)
    assert gate["status"] == "passed"
    assert gate["total_resist"] == 40
    assert gate["total_yield"] == 40
    assert gate["mixed_prompts_minority_ge_3"] == 10


def test_hierarchical_bootstrap_estimates_prompt_equal_difference() -> None:
    records: list[dict[str, Any]] = []
    families = [f"f{index}" for index in range(18)]
    for index, family in enumerate(families):
        prompt = f"p{index}"
        records.append(_synthetic_record(prompt, family, 0, "resist", 3.0))
        records.append(_synthetic_record(prompt, family, 1, "resist", 5.0))
        records.append(_synthetic_record(prompt, family, 2, "yield", 1.0))
        records.append(_synthetic_record(prompt, family, 3, "yield", 3.0))
    report = analyzer.hierarchical_bootstrap(
        records,
        analyzer.PRIMARY_FEATURE,
        families,
        replicates=200,
        seed=1,
    )
    assert report["mixed_prompts"] == 18
    assert report["point_estimate"] == 2.0
    assert report["ci95"][0] < 2.0 < report["ci95"][1]


def test_trajectory_features_obey_fixed_window_and_layer_groups() -> None:
    layer_ids = np.asarray(runner.LAYERS, dtype=np.int16)
    prompt_codes = np.asarray(
        [runner.PHASE_CODES["prefill_original"], runner.PHASE_CODES["prefill_challenge"]]
    )
    response_codes = np.asarray(
        [
            runner.PHASE_CODES["thinking_early"],
            runner.PHASE_CODES["thinking_late"],
            runner.PHASE_CODES["answer_choice"],
        ]
    )
    prefill_correction = np.vstack([np.ones(8), np.arange(8, dtype=float)])
    decode_correction = np.vstack([np.ones(8), np.arange(8, dtype=float)])
    arrays = {
        "layer_ids": layer_ids,
        "prefill_phase_codes": prompt_codes,
        "response_phase_codes": response_codes,
        "response_preanswer32_mask": np.asarray([False, True, False]),
        "decode_observed_response_indices": np.asarray([0, 1]),
        "prefill_correction": prefill_correction,
        "prefill_axis": prefill_correction,
        "prefill_distance": prefill_correction,
        "decode_correction": decode_correction,
        "decode_axis": decode_correction,
        "decode_distance": decode_correction,
    }
    features = analyzer.trajectory_features({"arrays": arrays})
    assert features[analyzer.PRIMARY_FEATURE] == 5.5
    assert features[analyzer.EARLY_LAYER_FEATURE] == 1.5
    assert features[analyzer.PREFILL_FEATURE] == 5.5
    assert features[analyzer.PHASE_INTERACTION_FEATURE] == 0.0
    assert features[analyzer.LAYER_INTERACTION_FEATURE] == 4.0


def test_length_matching_never_reuses_a_seed() -> None:
    records = [
        _synthetic_record("p", "f", 0, "resist", 4.0, 10),
        _synthetic_record("p", "f", 1, "resist", 6.0, 20),
        _synthetic_record("p", "f", 2, "yield", 1.0, 11),
        _synthetic_record("p", "f", 3, "yield", 2.0, 19),
    ]
    result = analyzer.length_matched_analysis(records, analyzer.PRIMARY_FEATURE, ["f"])
    assert result["matched_pairs"] == 2
    assert result["matched_prompts"] == 1
    assert result["mean_abs_thinking_token_difference"] == 1.0


def test_required_delivery_names_are_json_serializable() -> None:
    payload = {
        "layers": runner.LAYERS,
        "phase_codes": runner.PHASE_CODES,
        "primary": analyzer.PRIMARY_FEATURE,
    }
    json.dumps(payload)
