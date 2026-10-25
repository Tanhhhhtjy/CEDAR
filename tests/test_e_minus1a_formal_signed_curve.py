from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cedar_m2.models.qwen import QwenModelSpec
from scripts.analyze_e_minus1a_formal_signed_curve import (
    build_pair_results,
    build_seed_summary,
    write_curve_plot,
)
from scripts.run_e_minus1a_formal_signed_curve import (
    FORMAL_CONDITIONS,
    SENTINEL_CONDITIONS,
    _validate_addition_stats,
    audit_sentinel,
    forced_choice_forward,
    load_sentinel_rows,
    validate_formal_rows,
)
from scripts.run_e_minus1a_philpapers import EXPECTED_AXIS_SHA256


class _PassLayer(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_PassLayer() for _index in range(64)])


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()
        self.embedding = nn.Embedding(64, 2)
        self.config = SimpleNamespace(max_position_embeddings=128)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert use_cache is False
        assert attention_mask.shape == input_ids.shape
        hidden = self.embedding(input_ids)
        hidden = self.model.layers[32](hidden)
        logits = torch.zeros(*hidden.shape[:2], 64, device=hidden.device)
        logits[..., 32] = hidden[..., 0]
        logits[..., 33] = hidden[..., 1]
        return SimpleNamespace(logits=logits)


class _FakeTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        assert [message["role"] for message in chat] == ["user"]
        assert kwargs["enable_thinking"] is False
        return "rendered prompt"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "A":
            return [32]
        if text == "B":
            return [33]
        return [1, 2, 3]

    def __call__(self, texts, **kwargs):
        assert texts == ["rendered prompt"]
        assert kwargs["truncation"] is False
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        return f"token-{token_ids[0]}"


def _projection_stats(alpha: float | None) -> dict[str, object]:
    shift = 0.0 if alpha is None else alpha
    return {
        "calls": 1,
        "total_positions": 3,
        "valid_positions": 3,
        "modified_positions": 0 if alpha is None else 3,
        "finite": True,
        "within_tolerance_fraction": 1.0,
        "projection_shift": shift,
        "projection_shift_min": shift,
        "projection_shift_max": shift,
    }


def _synthetic_formal_results() -> list[dict[str, object]]:
    margins = {
        "alpha_neg200": 2.5,
        "alpha_neg100": 3.2,
        "alpha_neg50": 3.8,
        "alpha_0": 4.0,
        "alpha_pos50": 3.5,
        "alpha_pos100": 3.0,
        "alpha_pos200": 2.0,
    }
    rows: list[dict[str, object]] = []
    for seed in (42, 7, 123):
        for pair_index in range(150):
            for variant in ("original", "swapped"):
                for condition, alpha in FORMAL_CONDITIONS:
                    rows.append(
                        {
                            "sample_seed": seed,
                            "pair_id": f"seed{seed}-pair{pair_index}",
                            "base_id": pair_index,
                            "orig_id": pair_index,
                            "split": "test",
                            "condition": condition,
                            "coefficient": alpha,
                            "variant": variant,
                            "question_id": pair_index * 2 + (variant == "swapped"),
                            "sycophancy_margin": margins[condition],
                            "p_syc_within_AB": 0.8,
                            "AB_probability_mass": 0.5,
                            "AB_mean_logit": 2.0,
                            "full_vocab_entropy": 1.0,
                            "full_vocab_top1_probability": 0.5,
                            "projection_shift": alpha,
                            "full_vocab_top1_is_AB": True,
                            "predicted_answer": "A",
                            "sycophantic_answer": "A",
                        }
                    )
    return rows


def test_formal_data_and_frozen_sentinel_are_complete() -> None:
    rows = validate_formal_rows(
        Path("datasets/e_minus1a_philpapers/philpapers1800_all_seeds.jsonl")
    )
    assert len(rows) == 900
    assert len({row["pair_id"] for row in rows}) == 450
    assert {row["split"] for row in rows} == {"test"}
    sentinel, prompt_hashes = load_sentinel_rows()
    assert len(sentinel) == 8
    assert sentinel[0]["question_id"] == 548
    assert sentinel[-1]["question_id"] == 167
    assert len(prompt_hashes) == 8


def test_committed_smoke_outputs_pass_new_runner_sentinel_contract() -> None:
    sentinel, prompt_hashes = load_sentinel_rows()
    prior = [
        json.loads(line)
        for line in Path(
            "artifacts/m2/e_minus1a_philpapers/smoke64/row_results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for source in sentinel
        for condition, _alpha in SENTINEL_CONDITIONS
        for row in prior
        if row["sample_seed"] == source["sample_seed"]
        and row["question_id"] == source["question_id"]
        and row["condition"] == condition
    ]
    audit = audit_sentinel(sentinel, selected, prompt_hashes)
    assert audit["status"] == "passed"
    assert audit["behavior_direction_used_as_gate"] is False


def test_addition_stats_fail_closed_on_sign_and_tolerance() -> None:
    _validate_addition_stats(_projection_stats(50.0), alpha=50.0, prompt_tokens=3)
    _validate_addition_stats(_projection_stats(-50.0), alpha=-50.0, prompt_tokens=3)
    _validate_addition_stats(_projection_stats(0.0), alpha=0.0, prompt_tokens=3)
    _validate_addition_stats(_projection_stats(None), alpha=None, prompt_tokens=3)
    wrong_sign = _projection_stats(50.0)
    wrong_sign["projection_shift_min"] = -1.0
    with pytest.raises(RuntimeError, match="Positive addition"):
        _validate_addition_stats(wrong_sign, alpha=50.0, prompt_tokens=3)
    outside_tolerance = _projection_stats(-50.0)
    outside_tolerance["within_tolerance_fraction"] = 0.98
    with pytest.raises(RuntimeError, match="99%"):
        _validate_addition_stats(outside_tolerance, alpha=-50.0, prompt_tokens=3)


def test_forced_choice_forward_records_full_vocab_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fake-gpu")
    row = {
        "dataset_version": "e_minus1a_philpapers_v1",
        "sample_seed": 42,
        "pair_id": "pair",
        "base_id": 0,
        "orig_id": 0,
        "question_id": 0,
        "variant": "original",
        "split": "test",
        "question_text": "Question\nAnswer:",
        "sycophantic_answer": "A",
        "non_sycophantic_answer": "B",
    }
    axis = {
        "repo": "lu-christina/assistant-axis-vectors",
        "revision": "revision",
        "file": "qwen-3-32b/assistant_axis.pt",
        "sha256": EXPECTED_AXIS_SHA256,
        "raw_norm": 1.0,
        "unit_norm": 1.0,
    }
    result = forced_choice_forward(
        _FakeModel(),
        _FakeTokenizer(),
        row,
        condition="alpha_0",
        alpha=0.0,
        direction=torch.tensor([1.0, 0.0]),
        model_spec=QwenModelSpec(),
        axis_metadata=axis,
    )
    assert result["generated_text"] is False
    assert result["thinking_enabled"] is False
    assert result["prompt_truncated"] is False
    assert 0 <= result["AB_probability_mass"] <= 1
    assert 0 <= result["p_syc_within_AB"] <= 1
    assert result["full_vocab_size"] == 64


def test_pair_and_seed_analysis_use_original_swapped_as_one_unit() -> None:
    pair_rows = build_pair_results(_synthetic_formal_results())
    assert len(pair_rows) == 3150
    summary = build_seed_summary(pair_rows)
    positive_200 = [
        row for row in summary["entries"] if row["condition"] == "alpha_pos200"
    ]
    assert len(positive_200) == 3
    assert all(row["pairs"] == 150 for row in positive_200)
    assert all(row["deltas_vs_alpha0"]["sycophancy_margin"] == -2.0 for row in positive_200)


def test_curve_plot_is_a_real_png(tmp_path: Path) -> None:
    curve = []
    for _condition, alpha in FORMAL_CONDITIONS:
        curve.append(
            {
                "coefficient": alpha,
                "seed_mean_margins": {str(seed): 4 - abs(alpha) / 100 for seed in (42, 7, 123)},
                "seed_deltas_vs_alpha0": {
                    str(seed): -abs(alpha) / 100 for seed in (42, 7, 123)
                },
                "actual_projection_shift": alpha,
                "diagnostic_means": {
                    "AB_probability_mass": 0.5,
                    "top1_is_AB_rate": 0.75,
                },
            }
        )
    output = tmp_path / "curve.png"
    write_curve_plot(output, curve)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
