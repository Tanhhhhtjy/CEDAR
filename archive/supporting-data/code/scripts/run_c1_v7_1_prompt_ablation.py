#!/usr/bin/env python3
"""Run the frozen C1/V7.1 no-cue prompt ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers

try:
    from scripts.run_c1_v7_heldout import (
        ABC_TOKEN_IDS,
        CHOICES,
        LAYERS,
        MAX_NEW_TOKENS,
        Phase,
        aggregate_cap_stats,
        generate_one,
        git_provenance,
        load_cap_assets,
        materialize,
        read_jsonl,
        record_key,
        release_cuda_cache,
        update_runtime,
    )
except ModuleNotFoundError:
    from run_c1_v7_heldout import (
        ABC_TOKEN_IDS,
        CHOICES,
        LAYERS,
        MAX_NEW_TOKENS,
        Phase,
        aggregate_cap_stats,
        generate_one,
        git_provenance,
        load_cap_assets,
        materialize,
        read_jsonl,
        record_key,
        release_cuda_cache,
        update_runtime,
    )

from cedar_m2.checkpoint import RecordStore, file_sha256, initialize_run, object_sha256
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "m2_2b_formal" / "v7_1_prompt_cue_ablation"
DATA_PATH = DATA_DIR / "c1_v7_1_no_cue144_arms.jsonl"
PREFLIGHT_PATH = DATA_DIR / "execution_preflight_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c1_v7_1_prompt_cue_ablation"
EXPECTED_DATA_CONTENT_SHA256 = (
    "f416d576407ff4e237554009b20458977ff4604116852a54986afbe593c9aa91"
)
EXPECTED_SOURCE_CONTENT_SHA256 = (
    "a25577a3e6dcd117254efebe44fcd274307872410438461650d2a0f4137e1a15"
)
SMOKE_BASE = "C1V7-001"
PHASES = {
    "smoke-nohook": Phase("uncapped", False, "smoke_no_cue"),
    "smoke-monitor": Phase("uncapped", True, "smoke_no_cue"),
    "smoke-p25": Phase("p25", True, "smoke_p25_s"),
    "formal-uncapped": Phase("uncapped", True, "full_no_cue"),
    "formal-p25": Phase("p25", True, "full_no_cue"),
}


def normalized_text_sha256(path: Path) -> str:
    """Hash UTF-8 text after normalizing platform line endings to LF."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path("<local-path>"))
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def validate_inputs(data_path: Path, preflight_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(data_path)
    data_content_sha256 = object_sha256(rows)
    if data_content_sha256 != EXPECTED_DATA_CONTENT_SHA256:
        raise ValueError("C1/V7.1 canonical dataset content hash changed")
    if len(rows) != 144:
        raise ValueError(f"C1/V7.1 expected 144 rows, found {len(rows)}")
    if {row.get("arm") for row in rows} != {"N", "AO", "S"}:
        raise ValueError("C1/V7.1 arm set changed")
    if any(row.get("prompt_variant") != "natural_no_epistemic_status_cue" for row in rows):
        raise ValueError("C1/V7.1 prompt variant changed")
    if any(
        "No new task information is supplied." in row["messages"][-1]["content"]
        for row in rows
    ):
        raise ValueError("C1/V7.1 still contains the removed epistemic-status cue")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status")
        != "data_and_execution_code_frozen_awaiting_smoke_after_hash_amendment"
    ):
        raise ValueError("C1/V7.1 preflight status changed")
    if preflight.get("dataset_content_sha256") != data_content_sha256:
        raise ValueError("C1/V7.1 preflight dataset content hash changed")
    if preflight.get("source_v7_content_sha256") != EXPECTED_SOURCE_CONTENT_SHA256:
        raise ValueError("C1/V7.1 source V7 hash changed")
    if preflight.get("runner_normalized_text_sha256") != normalized_text_sha256(Path(__file__)):
        raise ValueError("C1/V7.1 runner hash does not match preflight")
    return rows


def select_rows(rows: list[dict[str, Any]], phase: Phase) -> list[dict[str, Any]]:
    if phase.selection == "full_no_cue":
        selected = rows
        expected = 144
    elif phase.selection == "smoke_no_cue":
        selected = [row for row in rows if row["base_id"] == SMOKE_BASE]
        expected = 3
    elif phase.selection == "smoke_p25_s":
        selected = [row for row in rows if row["base_id"] == SMOKE_BASE and row["arm"] == "S"]
        expected = 1
    else:
        raise ValueError(f"Unknown selection: {phase.selection}")
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} selected rows, found {len(selected)}")
    return selected


def main() -> None:
    args = parse_args()
    if args.seed != 0:
        raise ValueError("C1/V7.1 frozen greedy run uses seed 0 only")
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"Run inside the cedar environment, found {sys.prefix}")
    phase = PHASES[args.phase]
    rows = select_rows(validate_inputs(args.data, args.preflight), phase)
    axis, cap_sets, cap_metadata = load_cap_assets(args.cache_dir)
    run_dir = args.output_root / args.phase
    model_spec = QwenModelSpec()
    configuration = {
        "kind": "C1/V7.1 epistemic-status cue ablation",
        "scope": "paired no-cue sensitivity analysis; no item selection",
        "phase": args.phase,
        "condition": phase.condition,
        "monitor_enabled": phase.monitor_enabled,
        "selection": phase.selection,
        "seed": args.seed,
        "dataset": {
            "path": str(args.data),
            "content_sha256": object_sha256(rows),
            "file_sha256_diagnostic": file_sha256(args.data),
            "rows": len(rows),
            "selected_ids_sha256": object_sha256([row["id"] for row in rows]),
            "preflight_path": str(args.preflight),
            "preflight_sha256": file_sha256(args.preflight),
            "source_v7_content_sha256": EXPECTED_SOURCE_CONTENT_SHA256,
            "hash_policy": (
                "canonical parsed JSON content is frozen; raw file bytes are diagnostic"
            ),
        },
        "model": asdict(model_spec),
        "generation": {
            "enable_thinking": True,
            "do_sample": False,
            "num_beams": 1,
            "batch_size": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "content_retry": False,
            "answer_repair": False,
        },
        "axis_telemetry": {
            "layers": list(LAYERS),
            "hook_order": "monitor before same-layer cap",
            "phases": ["prefill", "decode"],
            "reference_caps": ["p25"] if phase.monitor_enabled else [],
        },
        "capping": cap_metadata,
        "prompt_ablation": {
            "removed_text": "No new task information is supplied.",
            "other_prompt_edits": 0,
        },
        "review_deviation": (
            "inherits V7 single data-lead global review; no item-level labels, "
            "second reviewer, agreement, or kappa"
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": torch.cuda.device_count(),
        },
        "git": git_provenance(allow_dirty=args.allow_dirty),
    }
    manifest = initialize_run(run_dir, configuration)
    store = RecordStore(run_dir / "records", run_fingerprint=manifest["fingerprint"])
    results_path = run_dir / "results.jsonl"
    runtime_path = run_dir / "record_runtime.jsonl"
    results, payloads = materialize(
        store,
        rows,
        phase_name=args.phase,
        seed=args.seed,
        results_path=results_path,
        runtime_path=runtime_path,
    )
    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "stage": "prepare",
                    "rows": len(rows),
                    "existing": len(results),
                }
            )
        )
        return
    missing = [
        index
        for index, row in enumerate(rows)
        if not store.contains(record_key(index, row, args.phase, args.seed))
    ]
    if missing:
        if torch.cuda.device_count() < 4:
            raise RuntimeError("C1/V7.1 Qwen3-32B run requires four visible GPUs")
        max_memory = {
            index: f"{args.gpu_memory_gib}GiB"
            for index in range(torch.cuda.device_count())
        }
        update_runtime(run_dir, {"generation_started_at": datetime.now().astimezone().isoformat()})
        model, tokenizer = load_qwen_model(max_memory=max_memory, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        observed_abc_ids = tuple(
            tokenizer.encode(choice, add_special_tokens=False)[0] for choice in CHOICES
        )
        if observed_abc_ids != ABC_TOKEN_IDS:
            raise RuntimeError("Qwen A/B/C token IDs changed")
        for index in missing:
            row = rows[index]
            result = generate_one(
                model,
                tokenizer,
                row,
                phase_name=args.phase,
                phase=phase,
                seed=args.seed,
                record_seed=args.seed * 1_000_003 + index,
                axis=axis,
                cap_sets=cap_sets,
            )
            store.write(
                record_key(index, row, args.phase, args.seed),
                {"index": index, "result": result},
            )
            results, payloads = materialize(
                store,
                rows,
                phase_name=args.phase,
                seed=args.seed,
                results_path=results_path,
                runtime_path=runtime_path,
            )
            if index == 0 and not result["thinking_integrity"]["passed"]:
                del model, tokenizer
                release_cuda_cache()
                raise RuntimeError("C1/V7.1 thinking integrity smoke failed")
            print(
                json.dumps(
                    {
                        "phase": args.phase,
                        "complete": len(results),
                        "expected": len(rows),
                    }
                ),
                flush=True,
            )
        del model, tokenizer
        release_cuda_cache()
    results, payloads = materialize(
        store,
        rows,
        phase_name=args.phase,
        seed=args.seed,
        results_path=results_path,
        runtime_path=runtime_path,
    )
    if len(results) == len(rows):
        update_runtime(
            run_dir,
            {
                "generation_completed_at": datetime.now().astimezone().isoformat(),
                "complete_results": len(results),
                "results_sha256": file_sha256(results_path),
                "record_runtime_sha256": file_sha256(runtime_path),
                "cap_trigger_stats": aggregate_cap_stats(
                    [{"cap_stats": payload["result"].get("cap_stats")} for payload in payloads]
                ),
                "thinking_integrity_failures": [
                    row["id"] for row in results if not row["thinking_integrity"]["passed"]
                ],
            },
        )
    print(json.dumps({"phase": args.phase, "complete": len(results), "expected": len(rows)}))


if __name__ == "__main__":
    main()
