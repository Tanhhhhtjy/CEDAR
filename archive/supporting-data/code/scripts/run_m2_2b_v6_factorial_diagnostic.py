#!/usr/bin/env python3
"""Run one cell of the M2.2b v6 thinking x decoding factorial diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from run_m2_2b_v4_neutral_pair import (
    aggregate_cap_stats,
    atomic_write_jsonl,
    direction_sha256,
    git_provenance,
    release_cuda_cache,
    stats_to_dict,
    update_runtime,
)

from cedar_m2.assets.capping_config import (
    ASSISTANT_AXIS_REPO,
    ASSISTANT_AXIS_REVISION,
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.checkpoint import RecordStore, file_sha256, initialize_run, object_sha256
from cedar_m2.interventions.axis_monitor import AssistantAxisMonitor
from cedar_m2.interventions.capping import ActivationCapper, LayerCap
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device
from cedar_m2.runners.belief import CONDITION_EXPERIMENTS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "m2_2b_formal" / "v6_factorial_diagnostic"
DATASETS = {
    "formal": DATA_DIR / "m2_2b_v6_factorial144.jsonl",
    "axis": DATA_DIR / "m2_2b_v6_axis_controls24.jsonl",
}
EXPECTED_ROWS = {"formal": 144, "axis": 24}
MAX_NEW_TOKENS = {"formal": 2048, "axis": 512}
CAPS = ("uncapped", "p25", "p1")
LAYERS = tuple(range(46, 54))


@dataclass(frozen=True)
class Mode:
    enable_thinking: bool
    do_sample: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None


MODES = {
    "thinking_greedy": Mode(True, False, None, None, None),
    "thinking_sampling": Mode(True, True, 0.6, 0.95, 20),
    "nonthinking_greedy": Mode(False, False, None, None, None),
    # Deliberately shares the thinking-mode sampler so the thinking factor is isolated.
    "nonthinking_sampling": Mode(False, True, 0.6, 0.95, 20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="formal")
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--condition", choices=CAPS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "manifest.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "m2" / "m2_2b_v6_factorial_diagnostic",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--cache-dir", type=Path, default=Path("<local-path>"))
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, dataset: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_ROWS[dataset]:
        raise ValueError(f"Expected {EXPECTED_ROWS[dataset]} {dataset} rows, found {len(rows)}")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("Every diagnostic row must have a unique ID")
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{row.get('id')}: missing messages")
        if any(
            message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            for message in messages
        ):
            raise ValueError(f"{row.get('id')}: malformed messages")
        if row.get("recommended_batch_size") != 1:
            raise ValueError(f"{row['id']}: v6 telemetry requires batch size 1")
    if dataset == "formal":
        counts = {arm: sum(row.get("arm") == arm for row in rows) for arm in ("N", "V", "F", "S")}
        if counts != {arm: 36 for arm in ("N", "V", "F", "S")}:
            raise ValueError(f"Formal arm counts changed: {counts}")
    return rows


def validate_manifest(path: Path, data: Path, dataset: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready_for_factorial_diagnostic":
        raise ValueError("v6 manifest is not ready")
    expected = manifest.get("files", {}).get(data.name, {}).get("sha256")
    if expected != file_sha256(data):
        raise ValueError("Dataset hash does not match v6 manifest")
    if manifest.get("source", {}).get("v5_model_outputs_read_by_builder") is not False:
        raise ValueError("Response-independence flag changed")
    count_key = "formal_rows" if dataset == "formal" else "axis_control_rows"
    expected_rows = manifest["counts"][count_key]
    if expected_rows != EXPECTED_ROWS[dataset]:
        raise ValueError("Manifest row count changed")
    return manifest


def load_cap_assets(
    cache_dir: Path,
) -> tuple[torch.Tensor, dict[str, dict[int, LayerCap]], dict[str, Any]]:
    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=cache_dir,
        local_files_only=True,
    )
    axis = load_axis(axis_path)
    cap_sets: dict[str, dict[int, LayerCap]] = {}
    metadata: dict[str, Any] = {
        "repo_id": ASSISTANT_AXIS_REPO,
        "revision": ASSISTANT_AXIS_REVISION,
        "axis_sha256": file_sha256(axis_path),
        "capping_config_sha256": file_sha256(config_path),
        "positions": "all decoder-block output positions",
        "experiments": {},
    }
    for label in ("p25", "p1"):
        experiment = CONDITION_EXPERIMENTS[label]
        if experiment is None:
            raise ValueError(f"Missing official experiment for {label}")
        caps = load_capping_experiment(config_path, experiment)
        orientation = verify_axis_orientation(caps, axis)
        if tuple(sorted(caps)) != LAYERS:
            raise ValueError(f"{label}: official layer range changed")
        cap_sets[label] = caps
        metadata["experiments"][label] = {
            "id": experiment,
            "layers": {
                str(layer): {
                    "threshold": cap.threshold,
                    "direction_sha256": direction_sha256(cap),
                    "axis_cosine": orientation[layer],
                }
                for layer, cap in sorted(caps.items())
            },
        }
    return axis, cap_sets, metadata


def render_prompt(tokenizer: Any, row: dict[str, Any], mode: Mode) -> str:
    return tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=mode.enable_thinking,
    )


def thinking_integrity(response: str, enabled: bool) -> dict[str, Any]:
    stripped = response.lstrip()
    opening = stripped.find("<think>")
    closing = stripped.find("</think>", opening + 7)
    reasoning = stripped[opening + 7 : closing].strip() if closing > opening >= 0 else ""
    final = stripped[closing + 8 :].strip() if closing >= 0 else stripped.strip()
    if enabled:
        passed = opening == 0 and closing > opening and bool(reasoning) and bool(final)
    else:
        passed = bool(final) and not bool(reasoning)
    return {
        "enabled": enabled,
        "open_tag": opening >= 0,
        "ordered_close_tag": closing > opening >= 0,
        "reasoning_nonempty": bool(reasoning),
        "final_nonempty": bool(final),
        "passed": passed,
    }


def record_key(index: int, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "index": index,
        "id": row["id"],
        "dataset": args.dataset,
        "mode": args.mode,
        "condition": args.condition,
        "seed": args.seed,
    }


def generation_kwargs(mode: Mode, max_new_tokens: int, tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_sample": mode.do_sample,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if mode.do_sample:
        kwargs.update(
            temperature=mode.temperature,
            top_p=mode.top_p,
            top_k=mode.top_k,
        )
    return kwargs


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    mode_name: str,
    condition: str,
    seed: int,
    record_seed: int,
    max_new_tokens: int,
    axis: torch.Tensor,
    cap_sets: dict[str, dict[int, LayerCap]],
) -> dict[str, Any]:
    mode = MODES[mode_name]
    prompt = render_prompt(tokenizer, row, mode)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=False)
    if tuple(encoded["input_ids"].shape[:1]) != (1,):
        raise ValueError("v6 diagnostic requires batch size 1")
    prompt_width = int(encoded["input_ids"].shape[1])
    device = model_input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}

    monitor = AssistantAxisMonitor(
        model,
        axis,
        layers=LAYERS,
        prefill_attention_mask=encoded["attention_mask"],
        reference_caps=cap_sets,
    )
    capper: ActivationCapper | None = None
    cap_context: Any = nullcontext()
    if condition != "uncapped":
        capper = ActivationCapper(
            model,
            cap_sets[condition],
            positions="all",
            track_stats=True,
            defer_stats_sync=True,
        )
        cap_context = capper

    # Per-record seeds make a resumed run bitwise equivalent to an uninterrupted run.
    transformers.set_seed(record_seed)
    with monitor:
        with cap_context:
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    **generation_kwargs(mode, max_new_tokens, tokenizer),
                )
    response_ids = output_ids[0, prompt_width:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    cap_stats = stats_to_dict(capper.stats) if capper is not None else None
    if cap_stats is not None:
        cap_stats["definition"] = (
            "all batch-size-1 prompt and generated-token positions; no padding positions"
        )
    return {
        "id": row["id"],
        "source_candidate_id": row.get("source_candidate_id"),
        "dataset": row["diagnostic_kind"],
        "mode": mode_name,
        "cap": condition,
        "seed": seed,
        "record_seed": record_seed,
        "thinking_enabled": mode.enable_thinking,
        "response": response,
        "prompt_tokens": prompt_width,
        "response_tokens": int(response_ids.numel()),
        "thinking_integrity": thinking_integrity(response, mode.enable_thinking),
        "cap_stats": cap_stats,
        "axis_telemetry": monitor.materialize(),
    }


def materialize(
    store: RecordStore,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    results_path: Path,
    runtime_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = store.read(record_key(index, row, args))
        if payload is None:
            continue
        if payload.get("index") != index or payload.get("result", {}).get("id") != row["id"]:
            raise ValueError(f"Checkpoint mismatch at row {index}")
        result = payload["result"]
        if result.get("mode") != args.mode or result.get("cap") != args.condition:
            raise ValueError(f"Condition mismatch at row {index}")
        results.append(result)
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    atomic_write_jsonl(runtime_path, payloads)
    return results, payloads


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"Run inside the cedar environment, found {sys.prefix}")
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")

    data = args.data or DATASETS[args.dataset]
    rows = load_rows(data, args.dataset)
    dataset_manifest = validate_manifest(args.manifest, data, args.dataset)
    axis, cap_sets, cap_metadata = load_cap_assets(args.cache_dir)
    mode = MODES[args.mode]
    max_new_tokens = MAX_NEW_TOKENS[args.dataset]
    run_name = args.run_name or (
        f"qwen3_32b_v6_{args.dataset}_{args.mode}_{args.condition}_seed{args.seed}_development"
    )
    run_dir = args.output_root / run_name
    model_spec = QwenModelSpec()
    configuration = {
        "kind": "M2.2b v6 factorial causal diagnostic",
        "scope": "development diagnosis; not confirmatory or method evaluation",
        "dataset": {
            "label": args.dataset,
            "path": str(data),
            "sha256": file_sha256(data),
            "manifest": str(args.manifest),
            "manifest_sha256": file_sha256(args.manifest),
            "rows": len(rows),
            "row_ids_sha256": object_sha256([row["id"] for row in rows]),
            "response_independent_selection": dataset_manifest["selection"]["response_independent"],
        },
        "mode": {"name": args.mode, **asdict(mode)},
        "condition": args.condition,
        "seed": args.seed,
        "model": asdict(model_spec),
        "generation": {
            "batch_size": 1,
            "max_new_tokens": max_new_tokens,
            "padding_side": "left",
            "raw_response": True,
            "content_retry": False,
            "answer_repair": False,
        },
        "axis_telemetry": {
            "layers": list(LAYERS),
            "hook_order": "monitor before same-layer cap",
            "phases": ["prefill", "decode"],
            "reference_caps": ["p25", "p1"],
        },
        "capping": cap_metadata,
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
    runtime_rows_path = run_dir / "record_runtime.jsonl"
    results, payloads = materialize(store, rows, args, results_path, runtime_rows_path)

    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "run_dir": str(run_dir),
                    "rows": len(rows),
                    "existing": len(results),
                    "fingerprint": manifest["fingerprint"],
                }
            ),
            flush=True,
        )

    if args.stage in {"generate", "all"}:
        missing = [
            index
            for index, row in enumerate(rows)
            if not store.contains(record_key(index, row, args))
        ]
        if missing:
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen3-32B diagnostic requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
            }
            update_runtime(
                run_dir,
                {"generation_started_at": datetime.now().astimezone().isoformat()},
            )
            model, tokenizer = load_qwen_model(max_memory=max_memory, local_files_only=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            rendered = [render_prompt(tokenizer, row, mode) for row in rows]
            tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
            update_runtime(
                run_dir,
                {
                    "rendered_prompts_sha256": object_sha256(
                        [
                            {"id": row["id"], "prompt": prompt}
                            for row, prompt in zip(rows, rendered, strict=True)
                        ]
                    ),
                    "tokenized_input_ids_sha256": object_sha256(
                        [
                            {"id": row["id"], "ids": ids}
                            for row, ids in zip(rows, tokenized, strict=True)
                        ]
                    ),
                    "device_map": getattr(model, "hf_device_map", None),
                    "tokenizer_padding_side": tokenizer.padding_side,
                    "tokenizer_pad_token_id": tokenizer.pad_token_id,
                },
            )
            for index in missing:
                row = rows[index]
                result = generate_one(
                    model,
                    tokenizer,
                    row,
                    mode_name=args.mode,
                    condition=args.condition,
                    seed=args.seed,
                    record_seed=args.seed * 1_000_003 + index,
                    max_new_tokens=max_new_tokens,
                    axis=axis,
                    cap_sets=cap_sets,
                )
                store.write(record_key(index, row, args), {"index": index, "result": result})
                results, payloads = materialize(
                    store, rows, args, results_path, runtime_rows_path
                )
                if index == 0 and not result["thinking_integrity"]["passed"]:
                    del model, tokenizer
                    release_cuda_cache()
                    raise RuntimeError("Mode integrity smoke check failed")
                print(
                    json.dumps(
                        {
                            "stage": "generate",
                            "dataset": args.dataset,
                            "mode": args.mode,
                            "condition": args.condition,
                            "row": index + 1,
                            "complete": len(results),
                            "expected": len(rows),
                        }
                    ),
                    flush=True,
                )
            del model, tokenizer
            release_cuda_cache()

        results, payloads = materialize(store, rows, args, results_path, runtime_rows_path)
        if len(results) == len(rows):
            cap_payloads = [
                {"cap_stats": payload["result"].get("cap_stats")} for payload in payloads
            ]
            update_runtime(
                run_dir,
                {
                    "generation_completed_at": datetime.now().astimezone().isoformat(),
                    "complete_results": len(results),
                    "results_sha256": file_sha256(results_path),
                    "record_runtime_sha256": file_sha256(runtime_rows_path),
                    "cap_trigger_stats": aggregate_cap_stats(cap_payloads),
                    "mode_integrity_failures": [
                        row["id"] for row in results if not row["thinking_integrity"]["passed"]
                    ],
                },
            )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "complete": len(results),
                    "expected": len(rows),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
