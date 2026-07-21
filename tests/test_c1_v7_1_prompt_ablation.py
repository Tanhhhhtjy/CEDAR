from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT
    / "datasets"
    / "m2_2b_formal"
    / "v7_1_prompt_cue_ablation"
    / "build_c1_v7_1_prompt_cue_ablation.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("v71_builder", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_changes_only_cue_and_metadata() -> None:
    builder = load_builder()
    source_rows = builder.read_jsonl(builder.SOURCE)
    rows = builder.build_rows(source_rows)
    assert len(rows) == 144
    assert {row["arm"] for row in rows} == {"N", "AO", "S"}
    assert all(builder.CUE.strip() not in row["messages"][-1]["content"] for row in rows)
    assert all(row["source_id"] + "-NOCUE" == row["id"] for row in rows)


def test_builder_is_deterministic() -> None:
    builder = load_builder()
    source_rows = builder.read_jsonl(builder.SOURCE)
    first = builder.build_rows(source_rows)
    second = builder.build_rows(source_rows)
    assert builder.object_sha256(first) == builder.object_sha256(second)
