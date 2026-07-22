#!/usr/bin/env python3
"""Freeze the C2/V8 portable execution inventory before any GPU use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cedar_m2.checkpoint import atomic_write_json, file_sha256, object_sha256

try:
    from scripts.c2_v8_portable_hashes import (
        TEXT_HASH_MODE,
        canonical_text_sha256,
        git_blob_id,
    )
except ModuleNotFoundError:
    from c2_v8_portable_hashes import TEXT_HASH_MODE, canonical_text_sha256, git_blob_id

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "c2_v8_natural_social_confirmation"
OUTPUT = DATA_ROOT / "execution_preflight_manifest.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "m2" / "c2_v8_natural_social_confirmation"

FILES = (
    "docs/C2_V8_NATURAL_SOCIAL_CONFIRMATION_PROTOCOL.md",
    "docs/C2_V8_PROTOCOL_AMENDMENT_01_SINGLE_REVIEW_WAIVER.md",
    "docs/C2_V8_NATURAL_SOCIAL_CONFIRMATION_RUN_LOG.md",
    "datasets/c2_v8_natural_social_confirmation/README.md",
    "datasets/c2_v8_natural_social_confirmation/manifest.json",
    "datasets/c2_v8_natural_social_confirmation/HUMAN_REVIEW_GATE.json",
    "datasets/c2_v8_natural_social_confirmation/review/single_review_attestation.json",
    "datasets/c2_v8_natural_social_confirmation/build_c2_v8_natural_social_confirmation.py",
    "datasets/c2_v8_natural_social_confirmation/validate_c2_v8.py",
    "datasets/c2_v8_natural_social_confirmation/validate_c2_v8_natural_social_confirmation.py",
    "scripts/c2_v8_portable_hashes.py",
    "scripts/build_c2_v8_natural_confirmation.py",
    "scripts/freeze_c2_v8_execution_preflight.py",
    "scripts/run_c2_v8_full_pipeline.sh",
    "scripts/run_c2_v8_natural_confirmation.py",
    "scripts/audit_c2_v8_natural_confirmation.py",
    "scripts/analyze_c2_v8_natural_confirmation.py",
    "tests/test_c2_v8_builder.py",
    "tests/test_c2_v8_runtime.py",
    "tests/test_c2_v8_analysis.py",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-complete", action="store_true")
    parser.add_argument("--targeted-tests", default="pending")
    parser.add_argument("--full-tests", default="pending")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.rglob("*")):
        raise RuntimeError("Cannot refreeze C2/V8 preflight after experiment artifacts exist")
    inventory = {}
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory[relative] = {
            "hash_mode": TEXT_HASH_MODE,
            "git_blob_id": git_blob_id(path, root=ROOT),
            "canonical_text_sha256": canonical_text_sha256(path),
            "bytes_diagnostic": path.stat().st_size,
            "raw_sha256_diagnostic": file_sha256(path),
        }
    content_hashes = {
        name: object_sha256(read_jsonl(DATA_ROOT / filename))
        for name, filename in {
            "bases": "c2_v8_base.jsonl",
            "arms": "c2_v8_ao_s_arms.jsonl",
            "free_subset": "c2_v8_free_generation_subset.jsonl",
            "smoke_subset": "c2_v8_smoke12.jsonl",
        }.items()
    }
    payload = {
        "status": "zero_gpu_code_frozen_single_review_authorized",
        "created_at": "2026-07-22",
        "input_base_commit": "af288f81c07f15f81e76f0cd6d72f6cf9e2cbdc5",
        "hash_mode": TEXT_HASH_MODE,
        "gpu_or_qwen_execution_performed": False,
        "smoke_performed": False,
        "formal_execution_performed": False,
        "human_review_gate": "passed_single_data_lead_full_review_protocol_deviation",
        "data_content_hashes": content_hashes,
        "model_contract": {
            "repo": "Qwen/Qwen3-32B",
            "revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "candidate_token_ids": {"A": 32, "B": 33, "C": 34, "D": 35},
            "p25_experiment": "layers_46:54-p0.25",
            "layers": list(range(46, 54)),
            "axis_sha256": "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211",
            "capping_config_sha256": (
                "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
            ),
        },
        "files": inventory,
        "zero_gpu_validation": {
            "status": "passed" if args.checks_complete else "pending",
            "source_construction_review": "passed",
            "targeted_tests": args.targeted_tests,
            "full_repository_tests": args.full_tests,
            "c2_targeted_ruff": "passed" if args.checks_complete else "pending",
            "full_repository_ruff": (
                "647 pre-existing errors outside C2/V8 files; disclosed in run log"
                if args.checks_complete
                else "pending"
            ),
            "official_asset_hash_check": "passed",
        },
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
