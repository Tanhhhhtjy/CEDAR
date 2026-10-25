"""Targeted CPU tests for the frozen E2B implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from e2b_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_SMOKE_CELLS,
    family_bootstrap,
    geometry,
    object_sha256,
    ordered_cells,
    read_json,
    read_jsonl,
    reconstruct_response_ids,
    smoke_prompt_ids,
    validate_frozen_protocol_hashes,
)


def test_frozen_protocol_hashes_match_manifest() -> None:
    observed = validate_frozen_protocol_hashes()
    assert len(observed) == 4


def test_trace_and_matrix_shapes() -> None:
    traces = read_jsonl(DATA / "frozen_traces192.jsonl")
    report = read_json(DATA / "build_report.json")
    assert len(traces) == 192
    assert len({row["trace_id"] for row in traces}) == 192
    formal = ordered_cells(traces, scope="formal", smoke_ids=report["smoke_prompt_ids"])
    smoke = ordered_cells(traces, scope="smoke", smoke_ids=report["smoke_prompt_ids"])
    assert len(formal) == EXPECTED_FORMAL_CELLS
    assert len(smoke) == EXPECTED_SMOKE_CELLS


def test_smoke_selection_is_hash_deterministic() -> None:
    traces = read_jsonl(DATA / "frozen_traces192.jsonl")
    prompts = []
    for source_id in sorted({row["source_id"] for row in traces}):
        prompts.append(next(row for row in traces if row["source_id"] == source_id))
    assert smoke_prompt_ids(prompts) == read_json(DATA / "build_report.json")["smoke_prompt_ids"]


def test_geometry_contract() -> None:
    row = {
        "initial_answer": "A",
        "target_answer": "B",
        "logit_diagnostics": {"logits": {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}},
    }
    observed = geometry(row)
    assert observed == {
        "m_target": -1.0,
        "m_background": -2.5,
        "m_general": -2.0,
        "m_specific": 1.5,
    }


def test_family_bootstrap_is_frozen_and_deterministic() -> None:
    scores = {f"family-{index:02d}": float(index) for index in range(12)}
    first = family_bootstrap(scores)
    second = family_bootstrap(scores)
    assert first == second
    assert first["draws"] == BOOTSTRAP_DRAWS
    assert first["seed"] == BOOTSTRAP_SEED
    assert first["estimate"] == 5.5


def test_pinned_tokenizer_reconstructs_all_frozen_responses() -> None:
    os.environ.setdefault("HF_HOME", "/data1/WuMingrui/HF_HOME")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-32B",
        revision="9216db5781bf21249d130ec9da846c4624c16137",
        local_files_only=True,
    )
    anchor = read_jsonl(
        ROOT / "artifacts/m2/e2_phase_validity_choice/formal-all-seed0/results.jsonl"
    )
    rows = [
        row
        for row in anchor
        if row.get("assay") == "answer"
        and row.get("condition") in {"uncapped", "full_p25"}
    ]
    methods = set()
    for row in rows:
        token_ids, method = reconstruct_response_ids(tokenizer, row)
        assert len(token_ids) == row["response_tokens"]
        assert object_sha256(token_ids) == row["response_token_ids_sha256"]
        methods.add(method)
    assert methods == {
        "canonical_reencode",
        "bytelevel_single_split",
        "bytelevel_repeated_split",
    }
