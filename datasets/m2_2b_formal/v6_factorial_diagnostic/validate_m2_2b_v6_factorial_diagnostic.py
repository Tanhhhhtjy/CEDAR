#!/usr/bin/env python3
"""Validate the frozen M2.2b v6 diagnostic data and its provenance contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FORMAL = HERE / "m2_2b_v6_factorial144.jsonl"
AXIS = HERE / "m2_2b_v6_axis_controls24.jsonl"
MANIFEST = HERE / "manifest.json"
BUILDER = HERE / "build_m2_2b_v6_factorial_diagnostic.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("m2_2b_v6_builder", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import v6 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_messages(row: dict[str, Any], failures: list[str]) -> None:
    messages = row.get("messages")
    require(isinstance(messages, list) and bool(messages), f"{row.get('id')}: messages", failures)
    if not isinstance(messages, list):
        return
    for message in messages:
        require(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            and bool(message["content"].strip()),
            f"{row.get('id')}: malformed message",
            failures,
        )


def main() -> None:
    failures: list[str] = []
    for path in (FORMAL, AXIS, MANIFEST, BUILDER):
        require(path.exists(), f"missing {path.relative_to(ROOT)}", failures)
    if failures:
        raise SystemExit("\n".join(failures))

    formal = read_jsonl(FORMAL)
    axis = read_jsonl(AXIS)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(len(formal) == 144, f"formal rows={len(formal)}", failures)
    require(len(axis) == 24, f"axis rows={len(axis)}", failures)
    require(len({row.get("id") for row in formal + axis}) == 168, "duplicate IDs", failures)
    require(
        Counter(row.get("arm") for row in formal) == {"N": 36, "V": 36, "F": 36, "S": 36},
        "formal arm balance",
        failures,
    )
    require(
        len({row.get("source_candidate_base_id") for row in formal}) == 36,
        "base count",
        failures,
    )
    require(len({row.get("template_family") for row in formal}) == 18, "family count", failures)

    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in formal:
        validate_messages(row, failures)
        by_base[row["source_candidate_base_id"]].append(row)
        require(
            row.get("selection_status") == "mechanical_response_independent",
            f"{row['id']}: selection",
            failures,
        )
        require(row.get("recommended_batch_size") == 1, f"{row['id']}: batch size", failures)
        require(
            row.get("split") == "development_factorial_diagnostic",
            f"{row['id']}: split",
            failures,
        )
    for base_id, rows in by_base.items():
        require({row["arm"] for row in rows} == {"N", "V", "F", "S"}, f"{base_id}: arms", failures)
        vf_levels = {row["gradient_level"] for row in rows if row["arm"] in {"V", "F"}}
        require(len(vf_levels) == 1, f"{base_id}: V/F level mismatch", failures)

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in axis:
        validate_messages(row, failures)
        by_pair[row["pair_id"]].append(row)
        require(row.get("recommended_batch_size") == 1, f"{row['id']}: batch size", failures)
        require(
            row.get("safety_content") == "safe identity elicitation only",
            f"{row['id']}: safety",
            failures,
        )
    require(len(by_pair) == 12, "axis pair count", failures)
    for pair_id, rows in by_pair.items():
        require(
            {row["condition"] for row in rows} == {"default_assistant", "persona"},
            f"{pair_id}: conditions",
            failures,
        )
        require(
            len({row["question"] for row in rows}) == 1,
            f"{pair_id}: question mismatch",
            failures,
        )

    require(
        manifest.get("source", {}).get("v5_model_outputs_read_by_builder") is False,
        "response-independence flag",
        failures,
    )
    require(
        manifest.get("selection", {}).get("response_independent") is True,
        "selection flag",
        failures,
    )
    require(
        manifest.get("factorial", {}).get("stage_1_new_formal_generations") == 1728,
        "factorial size",
        failures,
    )
    for path in (FORMAL, AXIS):
        expected = manifest.get("files", {}).get(path.name, {}).get("sha256")
        require(expected == sha256(path), f"{path.name}: hash mismatch", failures)

    builder = load_builder()
    rebuilt_formal, rebuilt_levels = builder.build_formal()
    rebuilt_axis = builder.build_axis_controls()
    require(
        rebuilt_formal == formal,
        "formal output is not deterministic from frozen sources",
        failures,
    )
    require(rebuilt_axis == axis, "axis output is not deterministic from frozen sources", failures)
    require(
        rebuilt_levels == manifest.get("selection", {}).get("family_levels"),
        "family levels",
        failures,
    )

    report = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "counts": {
            "formal": len(formal),
            "axis": len(axis),
            "bases": len(by_base),
            "families": len({row["template_family"] for row in formal}),
        },
        "hashes": {FORMAL.name: sha256(FORMAL), AXIS.name: sha256(AXIS)},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
