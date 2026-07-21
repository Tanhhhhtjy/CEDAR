#!/usr/bin/env python3
"""Build the frozen V7.1 no-cue prompt variants from the approved V7 arms."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "v7_heldout_confirmation" / "m2_2b_v7_heldout192_arms.jsonl"
OUTPUT = HERE / "c1_v7_1_no_cue144_arms.jsonl"
MANIFEST = HERE / "manifest.json"
SOURCE_VERSION = "m2.2b-v7-heldout-confirmation"
VERSION = "m2.2b-v7.1-prompt-cue-ablation"
ARMS = ("N", "AO", "S")
CUE = " No new task information is supplied."


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in source_rows if row.get("arm") in ARMS]
    if len(selected) != 144:
        raise ValueError(f"Expected 144 selected source rows, found {len(selected)}")
    output: list[dict[str, Any]] = []
    for source in selected:
        if source.get("dataset_version") != SOURCE_VERSION:
            raise ValueError(f"Unexpected source version for {source.get('id')}")
        messages = source.get("messages")
        if not isinstance(messages, list) or len(messages) != 4:
            raise ValueError(f"Malformed messages for {source.get('id')}")
        final = messages[-1].get("content")
        if not isinstance(final, str) or final.count(CUE) != 1:
            raise ValueError(f"Expected exactly one frozen cue in {source.get('id')}")
        row = copy.deepcopy(source)
        row["source_id"] = source["id"]
        row["source_row_sha256"] = object_sha256(source)
        row["id"] = f"{source['id']}-NOCUE"
        row["dataset_version"] = VERSION
        row["prompt_variant"] = "natural_no_epistemic_status_cue"
        row["removed_text"] = CUE.strip()
        row["messages"][-1]["content"] = final.replace(CUE, "", 1)
        output.append(row)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(f"{canonical(row)}\n" for row in rows), encoding="utf-8")


def main() -> None:
    source_rows = read_jsonl(SOURCE)
    rows = build_rows(source_rows)
    write_jsonl(OUTPUT, rows)
    manifest = {
        "status": "frozen_zero_gpu_prompt_ablation",
        "dataset_version": VERSION,
        "source_path": str(SOURCE.relative_to(HERE.parents[2])).replace("\\", "/"),
        "source_sha256": file_sha256(SOURCE),
        "output_path": str(OUTPUT.relative_to(HERE.parents[2])).replace("\\", "/"),
        "output_sha256": file_sha256(OUTPUT),
        "rows": len(rows),
        "arms": {arm: sum(row["arm"] == arm for row in rows) for arm in ARMS},
        "bases": len({row["base_id"] for row in rows}),
        "families": len({row["family_id"] for row in rows}),
        "transformation": "delete the exact frozen epistemic-status sentence once",
        "removed_text": CUE.strip(),
        "other_prompt_edits": 0,
        "qwen_gpu_or_capping_run": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
