from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cedar_m2.checkpoint import file_sha256
from scripts.analyze_e_minus1a_philpapers import build_reports
from scripts.run_e_minus1a_philpapers import (
    CONDITIONS,
    DATA_PATH,
    EXPECTED_AXIS_RAW_NORM,
    EXPECTED_AXIS_SHA256,
    render_prompt,
    validate_rows,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


def _input_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(64):
        variant = "original" if index % 2 == 0 else "swapped"
        rows.append(
            {
                "dataset_version": "e_minus1a_philpapers_v1",
                "sample_seed": 42,
                "pair_id": f"pair-{index // 2:02d}",
                "base_id": index // 2,
                "orig_id": 1000 + index // 2,
                "question_id": index,
                "variant": variant,
                "split": "tune",
                "question_text": f"Question {index}\nAnswer:",
                "sycophantic_answer": "A" if variant == "original" else "B",
                "non_sycophantic_answer": "B" if variant == "original" else "A",
            }
        )
    return rows


def _result_rows(inputs: list[dict[str, object]]) -> list[dict[str, object]]:
    margins = {
        "no_hook": 1.0,
        "alpha_0": 1.0,
        "alpha_pos200": 0.0,
        "alpha_neg200": 2.0,
    }
    shifts = {
        "no_hook": 0.0,
        "alpha_0": 0.0,
        "alpha_pos200": 200.0,
        "alpha_neg200": -200.0,
    }
    results: list[dict[str, object]] = []
    for source in inputs:
        prompt_hash = hashlib.sha256(str(source["question_id"]).encode()).hexdigest()
        for condition, alpha in CONDITIONS:
            margin = margins[condition]
            sycophantic = str(source["sycophantic_answer"])
            a_logit = margin if sycophantic == "A" else 0.0
            b_logit = margin if sycophantic == "B" else 0.0
            shift = shifts[condition]
            results.append(
                {
                    **{
                        field: source[field]
                        for field in (
                            "dataset_version",
                            "sample_seed",
                            "pair_id",
                            "base_id",
                            "orig_id",
                            "question_id",
                            "variant",
                            "split",
                            "sycophantic_answer",
                            "non_sycophantic_answer",
                        )
                    },
                    "condition": condition,
                    "alpha": 0.0 if alpha is None else alpha,
                    "intervention_hook": condition != "no_hook",
                    "telemetry_hook": True,
                    "prompt_sha256": prompt_hash,
                    "prompt_token_count": 10,
                    "A_token_id": 32,
                    "B_token_id": 33,
                    "A_logit": a_logit,
                    "B_logit": b_logit,
                    "predicted_answer": "A" if a_logit >= b_logit else "B",
                    "sycophancy_margin": margin,
                    "model_repo": "Qwen/Qwen3-32B",
                    "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
                    "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137",
                    "axis_repo": "lu-christina/assistant-axis-vectors",
                    "axis_revision": "3b3b788432ad33e3a28d9ff08e88a530c0740814",
                    "axis_file": "qwen-3-32b/assistant_axis.pt",
                    "axis_sha256": EXPECTED_AXIS_SHA256,
                    "axis_raw_norm": EXPECTED_AXIS_RAW_NORM,
                    "axis_unit_norm": 1.0,
                    "target_layer": 32,
                    "decoder_block_count": 64,
                    "positions": "all",
                    "intervention_formula": "h_new = h + alpha * unit_assistant_axis",
                    "dtype": "bfloat16",
                    "batch_size": 1,
                    "use_cache": False,
                    "thinking_enabled": False,
                    "generated_text": False,
                    "projection_before": 5.0,
                    "projection_after": 5.0 + shift,
                    "projection_shift": shift,
                    "projection_stats": {
                        "calls": 1,
                        "total_positions": 10,
                        "valid_positions": 10,
                        "modified_positions": 0 if condition == "no_hook" else 10,
                        "finite_positions": 10,
                        "finite": True,
                        "expected_shift": shift,
                        "tolerance": max(2.0, 0.02 * abs(shift)),
                        "within_tolerance_positions": 10,
                        "within_tolerance_fraction": 1.0,
                        "projection_before": 5.0,
                        "projection_after": 5.0 + shift,
                        "projection_shift": shift,
                        "projection_before_min": 5.0,
                        "projection_before_max": 5.0,
                        "projection_after_min": 5.0 + shift,
                        "projection_after_max": 5.0 + shift,
                        "projection_shift_min": shift,
                        "projection_shift_max": shift,
                        "max_abs_shift_error": 0.0,
                    },
                    "hook_count_before": 0,
                    "hook_count_after": 0,
                    "hook_residual": False,
                    "finite": True,
                    "error": None,
                }
            )
    return results


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_frozen_smoke_and_author_prompt_contract() -> None:
    rows = validate_rows(DATA_PATH)
    assert len(rows) == 64
    tokenizer = _FakeTokenizer()
    assert render_prompt(tokenizer, rows[0]["question_text"]) == "rendered"
    assert tokenizer.messages == [{"role": "user", "content": rows[0]["question_text"]}]
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_smoke_analyzer_passes_complete_directional_fixture(tmp_path: Path) -> None:
    inputs = _input_rows()
    results = _result_rows(inputs)
    results_path = tmp_path / "row_results.jsonl"
    _write_jsonl(results_path, results)
    runtime = {
        "status": "complete",
        "completed_forwards": 256,
        "expected_forwards": 256,
        "resumed_records": 0,
        "content_retries": 0,
        "errors": [],
        "row_results_sha256": file_sha256(results_path),
        "run_fingerprint": "fixture",
        "configuration": {"git": {"commit": "fixture", "branch": "fixture"}},
    }
    prepare = {"status": "passed", "run_fingerprint": "fixture"}

    completeness, projection, summary = build_reports(
        inputs,
        results,
        runtime,
        prepare,
        results_path=results_path,
    )

    assert completeness["status"] == "passed"
    assert projection["status"] == "passed"
    assert summary["gate_i"]["status"] == "passed"
    assert summary["gate_ii"]["status"] == "passed"
    assert summary["gate_ii"]["analysis"]["delta_pos200_vs_zero"] == -1.0
    assert summary["gate_ii"]["analysis"]["delta_neg200_vs_zero"] == 1.0


def test_smoke_analyzer_fails_direction_without_changing_instrument_gate(
    tmp_path: Path,
) -> None:
    inputs = _input_rows()
    results = _result_rows(inputs)
    for row in results:
        if row["condition"] != "alpha_pos200":
            continue
        sycophantic = str(row["sycophantic_answer"])
        row["A_logit"] = 2.0 if sycophantic == "A" else 0.0
        row["B_logit"] = 2.0 if sycophantic == "B" else 0.0
        by_answer = {"A": float(row["A_logit"]), "B": float(row["B_logit"])}
        row["sycophancy_margin"] = by_answer[sycophantic] - by_answer[
            str(row["non_sycophantic_answer"])
        ]
        row["predicted_answer"] = (
            "A" if float(row["A_logit"]) >= float(row["B_logit"]) else "B"
        )
    results_path = tmp_path / "row_results.jsonl"
    _write_jsonl(results_path, results)
    runtime = {
        "status": "complete",
        "completed_forwards": 256,
        "expected_forwards": 256,
        "resumed_records": 0,
        "content_retries": 0,
        "errors": [],
        "row_results_sha256": file_sha256(results_path),
        "run_fingerprint": "fixture",
        "configuration": {"git": {"commit": "fixture", "branch": "fixture"}},
    }
    prepare = {"status": "passed", "run_fingerprint": "fixture"}

    _completeness, _projection, summary = build_reports(
        inputs,
        results,
        runtime,
        prepare,
        results_path=results_path,
    )

    assert summary["gate_i"]["status"] == "passed"
    assert summary["gate_ii"]["status"] == "failed"
