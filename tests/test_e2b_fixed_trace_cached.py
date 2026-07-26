"""CPU tests for the E2B cache-faithful replay amendment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cached"
V1_DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
for path in (DATA, V1_DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e2b_cached_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_SMOKE_CELLS,
    REPLAY_TOLERANCE,
    SOURCE_MANIFEST,
    SOURCE_MATRIX,
    SOURCE_TRACES,
    ordered_cells,
    read_json,
    read_jsonl,
    v1_smoke_replay_diagnostic,
    validate_v1_failure_evidence,
)
from run_e2b_fixed_trace_cached import (  # noqa: E402
    FrozenTokenForcer,
    ProcessorOrderState,
    RawAnswerCapture,
)


def test_scientific_matrix_and_analysis_constants_are_unchanged() -> None:
    traces = read_jsonl(SOURCE_TRACES)
    matrix = read_jsonl(SOURCE_MATRIX)
    manifest = read_json(SOURCE_MANIFEST)
    assert len(traces) == 192
    assert len(matrix) == EXPECTED_FORMAL_CELLS == 384
    assert len(
        ordered_cells(traces, scope="smoke", smoke_ids=manifest["smoke_prompt_ids"])
    ) == EXPECTED_SMOKE_CELLS == 32
    assert BOOTSTRAP_DRAWS == 20_000
    assert BOOTSTRAP_SEED == 20260726
    assert REPLAY_TOLERANCE == 1e-4


def test_v1_failure_is_retained_and_diagnostic_is_not_common_shift_only() -> None:
    evidence = validate_v1_failure_evidence()
    assert evidence["rows"] == 32
    assert evidence["diagonal_checked"] == 16
    assert evidence["diagonal_failed"] == 14
    diagnostic = v1_smoke_replay_diagnostic()
    assert diagnostic["conclusion"] == "not_common_shift_only"
    assert diagnostic["summary"]["all"]["raw_max_abs"]["max_abs"] == 0.625
    assert diagnostic["summary"]["all"]["centered_max_abs"]["max_abs"] == 0.28125
    assert diagnostic["summary"]["all"]["m_general"]["max_abs"] == 0.375


def test_capture_precedes_force_and_raw_scores_are_preserved() -> None:
    state = ProcessorOrderState(prompt_tokens=2, prefix_ids=[7, 8])
    captured: list[torch.Tensor] = []
    capture = RawAnswerCapture(
        state,
        lambda scores: captured.append(scores.detach().clone()) or {"captured": True},
    )
    force = FrozenTokenForcer(state, terminal_probe_token_id=9)
    cases = (
        (torch.tensor([[1, 2]]), 7),
        (torch.tensor([[1, 2, 7]]), 8),
        (torch.tensor([[1, 2, 7, 8]]), 9),
    )
    final_raw = None
    for input_ids, expected in cases:
        raw = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        final_raw = raw.clone()
        unchanged = capture(input_ids, raw)
        assert torch.equal(unchanged, raw)
        forced = force(input_ids, unchanged)
        assert int(torch.argmax(forced, dim=-1).item()) == expected
        assert torch.isneginf(forced[0, :expected]).all()
    assert len(captured) == 1
    assert torch.equal(captured[0], final_raw[0])
    assert state.capture_calls == state.force_calls == 3
    assert state.capture_call_index == 2
    assert state.order_violations == []


def test_force_before_capture_is_rejected() -> None:
    state = ProcessorOrderState(prompt_tokens=2, prefix_ids=[7])
    force = FrozenTokenForcer(state, terminal_probe_token_id=9)
    with pytest.raises(RuntimeError, match="force:step=0"):
        force(torch.tensor([[1, 2]]), torch.zeros((1, 12)))
    assert state.order_violations
