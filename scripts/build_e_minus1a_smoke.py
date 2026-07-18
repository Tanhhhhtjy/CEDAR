"""Build the deterministic, engineering-only E-1a 64-row smoke manifest."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets" / "e_minus1a_philpapers"
INPUT_PATH = DATA_DIR / "philpapers1800_all_seeds.jsonl"
OUTPUT_PATH = DATA_DIR / "smoke64.jsonl"
MANIFEST_PATH = DATA_DIR / "smoke64_manifest.json"
EXPECTED_INPUT_SHA256 = "4d345d0254ba46b829c3055d854eed371bbc625d2e754aa5cdda448460f1b888"
SMOKE_SEED = 42
SMOKE_SPLIT = "tune"
N_PAIRS = 32


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    input_payload = INPUT_PATH.read_bytes()
    input_sha256 = sha256_bytes(input_payload)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            f"Frozen input hash mismatch: expected {EXPECTED_INPUT_SHA256}, got {input_sha256}"
        )

    rows = [json.loads(line) for line in input_payload.decode("utf-8").splitlines() if line]
    eligible = [
        row
        for row in rows
        if row["sample_seed"] == SMOKE_SEED and row["split"] == SMOKE_SPLIT
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[row["pair_id"]].append(row)

    if len(grouped) != 150 or len(eligible) != 300:
        raise ValueError("Expected 150 complete tune pairs (300 rows) for seed 42")
    for pair_id, pair_rows in grouped.items():
        variants = {row["variant"] for row in pair_rows}
        if len(pair_rows) != 2 or variants != {"original", "swapped"}:
            raise ValueError(f"Incomplete or malformed pair: {pair_id}")

    ranked_pair_ids = sorted(
        grouped,
        key=lambda pair_id: (hashlib.sha256(pair_id.encode()).hexdigest(), pair_id),
    )
    selected_pair_ids = ranked_pair_ids[:N_PAIRS]
    variant_order = {"original": 0, "swapped": 1}
    selected_rows = [
        row
        for pair_id in selected_pair_ids
        for row in sorted(grouped[pair_id], key=lambda item: variant_order[item["variant"]])
    ]

    if len(selected_rows) != 64:
        raise ValueError(f"Expected 64 smoke rows, got {len(selected_rows)}")
    if len({row["pair_id"] for row in selected_rows}) != 32:
        raise ValueError("Smoke selection must contain 32 complete pairs")
    if sum(row["sycophantic_answer"] == "A" for row in selected_rows) != 32:
        raise ValueError("Counterbalanced smoke set must contain 32 sycophantic A labels")

    write_jsonl(OUTPUT_PATH, selected_rows)
    output_payload = OUTPUT_PATH.read_bytes()
    manifest = {
        "dataset_version": "e_minus1a_philpapers_smoke64_v1",
        "purpose": "engineering_smoke_only_not_for_scientific_estimation_or_coefficient_selection",
        "source": {
            "path": "datasets/e_minus1a_philpapers/philpapers1800_all_seeds.jsonl",
            "sha256": input_sha256,
        },
        "selection": {
            "sample_seed": SMOKE_SEED,
            "split": SMOKE_SPLIT,
            "n_pairs": N_PAIRS,
            "n_rows": len(selected_rows),
            "rule": "sort complete pair_id values by SHA256(pair_id), then take the first 32",
            "selected_pair_ids_in_rank_order": selected_pair_ids,
        },
        "invariants": {
            "complete_original_swapped_pairs": True,
            "sycophantic_answer_A_rows": 32,
            "sycophantic_answer_B_rows": 32,
            "test_rows_used": 0,
        },
        "output": {
            "path": "datasets/e_minus1a_philpapers/smoke64.jsonl",
            "sha256": sha256_bytes(output_payload),
            "bytes": len(output_payload),
        },
        "builder": {
            "path": "scripts/build_e_minus1a_smoke.py",
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"Built {len(selected_rows)} rows ({N_PAIRS} pairs) at {OUTPUT_PATH}")
    print("Smoke SHA-256:", manifest["output"]["sha256"])


if __name__ == "__main__":
    main()
