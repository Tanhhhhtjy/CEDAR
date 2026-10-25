#!/usr/bin/env python3
"""Run the v1.1 12-family uncapped template calibration.

This runner deliberately supports only the pre-capping template-calibration
phase.  Development validation and full-bank execution remain blocked until a
single S1/S2 template is frozen by the calibration analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_development_v1_1"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from cedar_m2.checkpoint import RecordStore, atomic_write_json, file_sha256, initialize_run, object_sha256
from development_v1_1_common import (
    CALIBRATION_FAMILIES,
    EVIDENCE_ARMS,
    PARENT_RESULT_COMMIT,
    SOCIAL_CANDIDATE_ARMS,
    TOKENIZER_BINDING,
    canonical_sha256,
    file_sha256 as data_file_sha256,
    read_jsonl,
    validate_parent_files,
)

CANDIDATE_PATH = DATA_ROOT / "development_candidates336.jsonl"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
GATE_PATH = DATA_ROOT / "ZERO_GPU_GATE.json"
FINGERPRINTS_PATH = DATA_ROOT / "runner_input_fingerprints.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_v1_1_uncapped"

MODEL_REPO = TOKENIZER_BINDING["repo"]
MODEL_REVISION = TOKENIZER_BINDING["revision"]
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
MAX_NEW_TOKENS = 4096
SEED = 0
RUNNER_AMENDMENT = "development-v1.1-template-calibration-uncapped"
TRACKS = {
    "evidence": {"mode": "thinking", "arms": EVIDENCE_ARMS},
    "social-candidates": {"mode": "nonthinking", "arms": SOCIAL_CANDIDATE_ARMS},
}
SMOKE_BASE_IDS = (
    "CEDAR-DEV-001",
    "CEDAR-DEV-009",
    "CEDAR-DEV-018",
    "CEDAR-DEV-026",
)
EXPECTED_ROWS = {
    ("smoke", "evidence"): 16,
    ("smoke", "social-candidates"): 16,
    ("formal", "evidence"): 96,
    ("formal", "social-candidates"): 96,
}


def _base_module() -> Any:
    import run_prover_generator_step5_uncapped as base

    base.MAX_NEW_TOKENS = MAX_NEW_TOKENS
    base.SEED = SEED
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--track", choices=tuple(TRACKS), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "<local-path>")))
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_provenance() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.rstrip()
    if status:
        raise RuntimeError("v1.1 calibration refuses to run from a dirty worktree")
    upstream = subprocess.run(["git", "rev-parse", "@{u}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if commit != upstream:
        raise RuntimeError("v1.1 calibration requires local HEAD to equal upstream")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_RESULT_COMMIT, commit], cwd=ROOT)
    if ancestor.returncode != 0:
        raise RuntimeError("Runner HEAD must descend from the frozen v1 result commit")
    return {
        "commit": commit,
        "branch": branch,
        "upstream_commit": upstream,
        "head_matches_upstream": True,
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "parent_result_is_ancestor": True,
        "dirty": False,
    }


def validate_inputs() -> list[dict[str, Any]]:
    validate_parent_files()
    gate = read_json(GATE_PATH)
    if gate.get("gate") != "A1-template-calibration" or gate.get("may_start_template_calibration") is not True or gate.get("p25_authorized") is not False:
        raise ValueError("v1.1 Gate A1 does not authorize uncapped template calibration")
    manifest = read_json(MANIFEST_PATH)
    if manifest.get("fixed_tokenizer_gate") != "passed" or manifest.get("human_naturalization_gate") != "passed" or manifest.get("p25_authorized") is not False:
        raise ValueError("v1.1 release manifest is not ready for template calibration")
    file_entry = next((item for item in manifest.get("files", []) if item.get("path") == CANDIDATE_PATH.name), None)
    if file_entry is None or file_entry.get("raw_file_sha256") != data_file_sha256(CANDIDATE_PATH):
        raise ValueError("v1.1 candidate-bank hash is not bound by the release manifest")
    fingerprints = read_json(FINGERPRINTS_PATH)
    if fingerprints.get("status") != "passed" or fingerprints.get("candidate_raw_sha256") != data_file_sha256(CANDIDATE_PATH):
        raise ValueError("v1.1 runner input fingerprints are absent or stale")
    rows = read_jsonl(CANDIDATE_PATH)
    if len(rows) != 336 or len({row["id"] for row in rows}) != 336:
        raise ValueError("v1.1 requires 336 unique candidate arms")
    if Counter(row["arm"] for row in rows) != Counter({arm: 48 for arm in ("N", "R", "V", "F", "AO", "S1", "S2")}):
        raise ValueError("v1.1 candidate-arm balance changed")
    return rows


def select_rows(rows: Sequence[dict[str, Any]], scope: str, track: str) -> list[dict[str, Any]]:
    arms = set(TRACKS[track]["arms"])
    selected = [
        row
        for row in rows
        if row["family_id"] in set(CALIBRATION_FAMILIES)
        and row["development_split"] == "template_calibration"
        and row["arm"] in arms
        and (scope == "formal" or row["base_id"] in set(SMOKE_BASE_IDS))
    ]
    expected = EXPECTED_ROWS[(scope, track)]
    if len(selected) != expected or len({row["id"] for row in selected}) != expected:
        raise ValueError(f"{scope}/{track} expected {expected} unique rows")
    per_arm = expected // len(arms)
    if Counter(row["arm"] for row in selected) != Counter({arm: per_arm for arm in TRACKS[track]["arms"]}):
        raise ValueError(f"{scope}/{track} lost complete arm balance")
    return selected


def run_name(scope: str, track: str) -> str:
    return f"v1_1_template_calibration_{scope}_{track.replace('-', '_')}_uncapped_seed0"


def completed_phase(output_root: Path, scope: str, track: str) -> bool:
    run_dir = output_root / run_name(scope, track)
    runtime_path = run_dir / "runtime.json"
    results_path = run_dir / "results.jsonl"
    if not runtime_path.is_file() or not results_path.is_file():
        return False
    runtime = read_json(runtime_path)
    return runtime.get("complete_results") == EXPECTED_ROWS[(scope, track)] and runtime.get("results_sha256") == file_sha256(results_path)


def validate_run_order(scope: str, track: str, output_root: Path) -> None:
    if scope == "smoke" and track == "social-candidates" and not completed_phase(output_root, "smoke", "evidence"):
        raise ValueError("Social-candidate smoke is blocked until evidence smoke completes")
    if scope == "formal":
        audit = output_root / "TEMPLATE_CALIBRATION_SMOKE_AUDIT.json"
        if not audit.is_file() or read_json(audit).get("status") != "passed":
            raise ValueError("Formal calibration is blocked until the combined smoke audit passes")
        if track == "social-candidates" and not completed_phase(output_root, "formal", "evidence"):
            raise ValueError("Formal social-candidate calibration is blocked until evidence completes")


def expected_fingerprints(scope: str, track: str) -> dict[str, str]:
    payload = read_json(FINGERPRINTS_PATH)
    key = f"template-calibration/{scope}/{track}"
    cell = payload.get("cells", {}).get(key)
    if not isinstance(cell, dict):
        raise ValueError(f"Missing frozen input fingerprint cell: {key}")
    return {"rendered_prompts_sha256": cell["rendered_prompts_sha256"], "tokenized_input_ids_sha256": cell["tokenized_input_ids_sha256"]}


def record_key(index: int, row: Mapping[str, Any], scope: str, track: str) -> dict[str, Any]:
    return {"index": index, "id": row["id"], "phase": "template-calibration", "scope": scope, "track": track, "condition": "uncapped", "seed": SEED}


def materialize(store: RecordStore, rows: Sequence[dict[str, Any]], *, scope: str, track: str, results_path: Path, runtime_path: Path) -> list[dict[str, Any]]:
    results = []
    payloads = []
    for index, row in enumerate(rows):
        payload = store.read(record_key(index, row, scope, track))
        if payload is None:
            continue
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("id") != row["id"] or result.get("track") != track:
            raise ValueError(f"Checkpoint mismatch at row {index}")
        results.append(result)
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    atomic_write_jsonl(runtime_path, payloads)
    return results


def update_runtime(run_dir: Path, values: Mapping[str, Any]) -> None:
    path = run_dir / "runtime.json"
    runtime = read_json(path) if path.is_file() else {}
    runtime.update(values)
    atomic_write_json(path, runtime)


def configuration(rows: Sequence[dict[str, Any]], scope: str, track: str, git: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "CEDAR prover-generator development v1.1 uncapped template calibration",
        "runner_amendment": RUNNER_AMENDMENT,
        "phase": "template-calibration",
        "scope": scope,
        "track": track,
        "mode": TRACKS[track]["mode"],
        "condition": "uncapped",
        "seed": SEED,
        "dataset": {
            "path": CANDIDATE_PATH.relative_to(ROOT).as_posix(),
            "raw_sha256": data_file_sha256(CANDIDATE_PATH),
            "selected_rows": len(rows),
            "selected_ids_sha256": canonical_sha256([row["id"] for row in rows]),
            "family_split": "template_calibration",
            "families": list(CALIBRATION_FAMILIES),
            "smoke_base_ids": list(SMOKE_BASE_IDS) if scope == "smoke" else None,
        },
        "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION, "dtype": MODEL_DTYPE, "attention_implementation": ATTENTION_IMPLEMENTATION},
        "generation": {"batch_size": 1, "do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "content_retry": False, "answer_repair": False},
        "prohibited_conditions": ["p25", "p1", "held_out_confirmation", "LoFa", "J-Lens"],
        "git": dict(git),
    }


def main() -> None:
    args = parse_args()
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    base = _base_module()
    all_rows = validate_inputs()
    rows = select_rows(all_rows, args.scope, args.track)
    validate_run_order(args.scope, args.track, args.output_root)
    git_state = git_provenance()
    run_dir = args.output_root / run_name(args.scope, args.track)
    manifest = initialize_run(run_dir, configuration(rows, args.scope, args.track, git_state))
    store = RecordStore(run_dir / "records", run_fingerprint=str(manifest["fingerprint"]))
    results_path = run_dir / "results.jsonl"
    runtime_rows_path = run_dir / "record_runtime.jsonl"
    results = materialize(store, rows, scope=args.scope, track=args.track, results_path=results_path, runtime_path=runtime_rows_path)
    if args.stage in {"prepare", "all"}:
        print(json.dumps({"stage": "prepare", "scope": args.scope, "track": args.track, "mode": TRACKS[args.track]["mode"], "rows": len(rows), "existing": len(results), "run_dir": str(run_dir), "runner_commit": git_state["commit"], "gpu_used": False}, sort_keys=True), flush=True)
        if args.stage == "prepare":
            return
    missing = [index for index, row in enumerate(rows) if not store.contains(record_key(index, row, args.scope, args.track))]
    if missing:
        import torch

        if Path(sys.prefix).name != "cedar":
            raise RuntimeError(f"v1.1 generation requires cedar, found {sys.prefix}")
        if torch.cuda.device_count() < 4:
            raise RuntimeError("Qwen3-32B v1.1 calibration requires at least four visible GPUs")
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
        import transformers
        from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model

        mode_name = TRACKS[args.track]["mode"]
        max_memory = {index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())}
        spec = QwenModelSpec(repo_id=MODEL_REPO, revision=MODEL_REVISION, dtype=MODEL_DTYPE, attention_implementation=ATTENTION_IMPLEMENTATION)
        model, tokenizer = load_qwen_model(spec, max_memory=max_memory, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base.validate_label_tokens(tokenizer)
        rendered = [base.render_prompt(tokenizer, row, base.MODES[mode_name]) for row in rows]
        tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
        rendered_hash = object_sha256([{"id": row["id"], "prompt": prompt} for row, prompt in zip(rows, rendered, strict=True)])
        tokenized_hash = object_sha256([{"id": row["id"], "input_ids": ids} for row, ids in zip(rows, tokenized, strict=True)])
        expected = expected_fingerprints(args.scope, args.track)
        if rendered_hash != expected["rendered_prompts_sha256"] or tokenized_hash != expected["tokenized_input_ids_sha256"]:
            raise RuntimeError("Frozen v1.1 rendered/tokenized input hash changed")
        update_runtime(run_dir, {"generation_started_at": datetime.now().astimezone().isoformat(), "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "visible_gpu_count": torch.cuda.device_count(), "hf_home": str(args.hf_home.resolve())}, "rendered_prompts_sha256": rendered_hash, "tokenized_input_ids_sha256": tokenized_hash})
        for index in missing:
            row = rows[index]
            result = base.generate_one(model, tokenizer, transformers, row, scope=f"template-calibration-{args.scope}", mode_name=mode_name, record_seed=index)
            result.update(
                {
                    "phase": "template-calibration",
                    "track": args.track,
                    "development_split": row["development_split"],
                    "position_transition": row["position_transition"],
                    "social_template": row.get("social_template"),
                }
            )
            store.write(record_key(index, row, args.scope, args.track), {"index": index, "result": result})
            results = materialize(store, rows, scope=args.scope, track=args.track, results_path=results_path, runtime_path=runtime_rows_path)
            print(json.dumps({"stage": "generate", "scope": args.scope, "track": args.track, "complete": len(results), "expected": len(rows)}), flush=True)
        del model, tokenizer
        base.release_cuda_cache()
    results = materialize(store, rows, scope=args.scope, track=args.track, results_path=results_path, runtime_path=runtime_rows_path)
    if len(results) == len(rows):
        update_runtime(run_dir, {"generation_completed_at": datetime.now().astimezone().isoformat(), "complete_results": len(results), "results_sha256": file_sha256(results_path), "record_runtime_sha256": file_sha256(runtime_rows_path), "content_retries": 0, "answer_repairs": 0, "parse_failures": [row["id"] for row in results if row["final_answer_parse_status"] != "passed"], "missing_answer_logits": [row["id"] for row in results if row["answer_logits"] is None]})
    print(json.dumps({"scope": args.scope, "track": args.track, "complete": len(results), "expected": len(rows), "condition": "uncapped"}), flush=True)


if __name__ == "__main__":
    main()
