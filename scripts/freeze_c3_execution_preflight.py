#!/usr/bin/env python3
"""Freeze the reviewed C3 data/code inventory before any GPU execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cedar_m2.checkpoint import atomic_write_json, object_sha256  # noqa: E402
from scripts.c3_portable_hashes import file_identity, read_jsonl  # noqa: E402

DATA_ROOT = ROOT / "datasets" / "c3_sycobench_bidirectional_selectivity"
OUTPUT = DATA_ROOT / "execution_preflight_manifest.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "m2" / "c3_sycobench_bidirectional_selectivity"
VALIDATOR_PATH = DATA_ROOT / "validate_c3_sycobench_bidirectional.py"

FILES = (
    ".gitattributes",
    "docs/C3_SYCOBENCH_BIDIRECTIONAL_SELECTIVITY_PROTOCOL.md",
    "docs/C3_PROTOCOL_AMENDMENT_01_AGENT_SELF_REVIEW_WAIVER.md",
    "docs/C3_SYCOBENCH_BIDIRECTIONAL_RUN_LOG.md",
    "datasets/c3_sycobench_bidirectional_selectivity/README.md",
    "datasets/c3_sycobench_bidirectional_selectivity/manifest.json",
    "datasets/c3_sycobench_bidirectional_selectivity/source_audit.json",
    "datasets/c3_sycobench_bidirectional_selectivity/construction_audit.json",
    "datasets/c3_sycobench_bidirectional_selectivity/REVIEW_GATE.json",
    "datasets/c3_sycobench_bidirectional_selectivity/c3_base600.jsonl",
    "datasets/c3_sycobench_bidirectional_selectivity/c3_arms2400.jsonl",
    "datasets/c3_sycobench_bidirectional_selectivity/c3_smoke.jsonl",
    "datasets/c3_sycobench_bidirectional_selectivity/review/review_packet32.csv",
    "datasets/c3_sycobench_bidirectional_selectivity/review/template_review_packet.csv",
    "datasets/c3_sycobench_bidirectional_selectivity/review/agent_review_completed.csv",
    "datasets/c3_sycobench_bidirectional_selectivity/review/template_agent_review_completed.csv",
    "datasets/c3_sycobench_bidirectional_selectivity/review/agent_self_review_attestation.json",
    "datasets/c3_sycobench_bidirectional_selectivity/build_c3_sycobench_bidirectional.py",
    "datasets/c3_sycobench_bidirectional_selectivity/validate_c3_sycobench_bidirectional.py",
    "datasets/c2_v8_natural_social_confirmation/source_syco600.official.json",
    "datasets/c2_v8_natural_social_confirmation/SOURCE_DATA_LICENSE.txt",
    "scripts/c3_portable_hashes.py",
    "scripts/build_c3_sycobench_bidirectional.py",
    "scripts/freeze_c3_agent_self_review_attestation.py",
    "scripts/freeze_c3_execution_preflight.py",
    "scripts/run_c3_full_pipeline.sh",
    "scripts/run_c3_sycobench_bidirectional.py",
    "scripts/audit_c3_sycobench_bidirectional.py",
    "scripts/analyze_c3_sycobench_bidirectional.py",
    "scripts/build_c2_v8_natural_confirmation.py",
    "scripts/run_c2_v8_natural_confirmation.py",
    "tests/test_c3_builder.py",
    "tests/test_c3_runtime.py",
    "tests/test_c3_analysis.py",
)


def load_validator():
    spec = importlib.util.spec_from_file_location("c3_preflight_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load C3 validator: {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-complete", action="store_true")
    parser.add_argument("--targeted-tests", default="pending")
    parser.add_argument("--full-tests", default="pending")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.rglob("*")):
        raise RuntimeError("Cannot freeze C3 execution preflight after model artifacts exist")
    validator = load_validator()
    gate = validator.validate_review_gate(validator.load_builder())
    construction = validator.validate_construction(validator.load_builder())
    if gate.get("gpu_authorized") is not True or construction.get("status") != "passed":
        raise RuntimeError("C3 review/construction gate is not passed")
    inventory = {}
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory[relative] = file_identity(path, root=ROOT)
    content_hashes = {
        "bases": object_sha256(read_jsonl(DATA_ROOT / "c3_base600.jsonl")),
        "arms": object_sha256(read_jsonl(DATA_ROOT / "c3_arms2400.jsonl")),
        "smoke": object_sha256(read_jsonl(DATA_ROOT / "c3_smoke.jsonl")),
    }
    expected = {
        "bases": "dbb4a3798c729c79271b06702e9cf712bcf4b18f1766eff9a0f2033e7f4fa205",
        "arms": "edd16f21bb8f0b647bd77fb8faa50a50dc2db8c941d264145efaa2bf9a8949a6",
        "smoke": "c1e56e7827a6c1ce20548462b8c7afc6557c51e1efd651e7a5362003073d84c2",
    }
    if content_hashes != expected:
        raise ValueError("C3 data semantic hashes differ from the zero-GPU freeze")
    payload = {
        "status": "zero_gpu_code_frozen_agent_self_review_authorized",
        "created_at": "2026-07-23",
        "gpu_or_qwen_execution_performed": False,
        "formal_execution_performed": False,
        "review_gate": "passed_agent_self_review_user_waiver",
        "review_disclosure": {
            "review_mode": "agent_self_review_user_waiver",
            "protocol_deviation": True,
            "human_review_performed": False,
            "data_lead_review_performed": False,
            "user_review_waived": True,
            "dual_independent_review": False,
            "raw_agreement": None,
            "cohen_kappa": None,
        },
        "data_content_hashes": content_hashes,
        "model_contract": {
            "repo": "Qwen/Qwen3-32B",
            "revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "enable_thinking": False,
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
            "scoped_ruff": "passed" if args.checks_complete else "pending",
        },
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
