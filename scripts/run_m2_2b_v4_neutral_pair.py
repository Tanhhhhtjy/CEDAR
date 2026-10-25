#!/usr/bin/env python3
"""Run one condition of the paired neutral-prompt M2.2b v4 calibration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from cedar_m2.assets.capping_config import (
    ASSISTANT_AXIS_REPO,
    ASSISTANT_AXIS_REVISION,
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.checkpoint import (
    RecordStore,
    atomic_write_json,
    file_sha256,
    initialize_run,
    object_sha256,
)
from cedar_m2.data.m2_2b_v4_neutral import (
    ARMS,
    PROMPT_CONTRACT_SHA256,
    SYSTEM,
    neutral_followup,
)
from cedar_m2.interventions.capping import ActivationCapper, CapStats, LayerCap
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device

EXPECTED_ROWS = 180
EXPECTED_BASES = 36
EXPECTED_BATCH_SIZE = 2
EXPECTED_MAX_NEW_TOKENS = 2048
P75_EXPERIMENT = "layers_46:54-p0.75"
CONDITIONS = ("uncapped", "p75")
RUN_NAMES = {
    "uncapped": "qwen3_32b_neutral_thinking_uncapped_greedy_seed0_development",
    "p75": "qwen3_32b_neutral_thinking_p75_greedy_seed0_development",
}
MESSAGE_FIELDS = ("system", "initial_user", "initial_assistant", "followup_user")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "datasets/m2_2b_formal/v4_public_poc/"
            "m2_2b_v4_public_poc180_neutral_prompt_arms.jsonl"
        ),
    )
    parser.add_argument(
        "--diff-audit",
        type=Path,
        default=Path(
            "datasets/m2_2b_formal/v4_public_poc/neutral_prompt_diff_audit.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/m2/m2_2b_v4_public_poc"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/data1/WuMingrui/HF_HOME/hub"))
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=EXPECTED_MAX_NEW_TOKENS)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("Every row needs a non-empty string ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Row IDs must be unique")
    if Counter(row.get("arm") for row in rows) != Counter({arm: 36 for arm in ARMS}):
        raise ValueError("Expected 36 records for each N/R/V/F/S arm")

    base_arms: dict[str, list[str]] = {}
    for row in rows:
        item_id = row["id"]
        base_id = row.get("base_id")
        if not isinstance(base_id, str) or not base_id:
            raise ValueError(f"Row {item_id} has a malformed base_id")
        base_arms.setdefault(base_id, []).append(str(row.get("arm")))
        for field in MESSAGE_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Row {item_id} has malformed {field}")
        try:
            initial = json.loads(row["initial_assistant"])
        except json.JSONDecodeError as error:
            raise ValueError(f"Row {item_id} has invalid initial_assistant JSON") from error
        if initial.get("answer") != row.get("initial_answer_id"):
            raise ValueError(f"Row {item_id} fixed initial assistant does not match metadata")
        if row["system"] != SYSTEM:
            raise ValueError(f"Row {item_id} does not use the frozen neutral system")
        if row["followup_user"] != neutral_followup(row):
            raise ValueError(f"Row {item_id} does not use the frozen {row['arm']} contract")
        if row.get("recommended_enable_thinking") is not True:
            raise ValueError(f"Row {item_id} does not require thinking mode")
        if row.get("recommended_do_sample") is not False:
            raise ValueError(f"Row {item_id} does not require greedy decoding")
        if int(row.get("recommended_max_new_tokens", 0)) < EXPECTED_MAX_NEW_TOKENS:
            raise ValueError(f"Row {item_id} has an invalid max-token recommendation")
    if len(base_arms) != EXPECTED_BASES or any(
        observed != list(ARMS) for observed in base_arms.values()
    ):
        raise ValueError("Expected 36 ordered N/R/V/F/S base units")
    return rows


def validate_diff_audit(path: Path, data: Path) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("Neutral prompt diff audit did not pass")
    if audit.get("neutral_sha256") != file_sha256(data):
        raise ValueError("Neutral prompt data hash does not match its diff audit")
    if audit.get("prompt_contract_sha256") != PROMPT_CONTRACT_SHA256:
        raise ValueError("Neutral prompt contract hash changed")
    if audit.get("violations"):
        raise ValueError("Neutral prompt diff audit contains violations")
    return audit


def messages_for_row(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["initial_user"]},
        {"role": "assistant", "content": row["initial_assistant"]},
        {"role": "user", "content": row["followup_user"]},
    ]


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        messages_for_row(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def thinking_smoke_check(response: str) -> dict[str, Any]:
    stripped = response.lstrip()
    open_index = stripped.find("<think>")
    close_index = stripped.find("</think>", open_index + len("<think>"))
    reasoning = (
        stripped[open_index + len("<think>") : close_index].strip()
        if open_index >= 0 and close_index > open_index
        else ""
    )
    final = stripped[close_index + len("</think>") :].strip() if close_index >= 0 else ""
    checks = {
        "starts_with_think": stripped.startswith("<think>"),
        "has_ordered_close": close_index > open_index >= 0,
        "reasoning_nonempty": bool(reasoning),
        "final_nonempty": bool(final),
    }
    return {**checks, "passed": all(checks.values())}


def git_provenance(*, allow_dirty: bool) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.rstrip()
    if status and not allow_dirty:
        raise RuntimeError("Refusing a paired calibration run from a dirty worktree")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], check=True, capture_output=True
    ).stdout
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def direction_sha256(cap: LayerCap) -> str:
    tensor = cap.direction.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_p75_contract(cache_dir: Path) -> tuple[dict[int, LayerCap], dict[str, Any]]:
    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=cache_dir,
        local_files_only=True,
    )
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, P75_EXPERIMENT)
    orientation = verify_axis_orientation(caps, axis)
    if sorted(caps) != list(range(46, 54)):
        raise ValueError(f"Official p75 layers changed: {sorted(caps)}")
    metadata = {
        "repo_id": ASSISTANT_AXIS_REPO,
        "revision": ASSISTANT_AXIS_REVISION,
        "axis_file": "qwen-3-32b/assistant_axis.pt",
        "axis_sha256": file_sha256(axis_path),
        "capping_config_file": "qwen-3-32b/capping_config.pt",
        "capping_config_sha256": file_sha256(config_path),
        "experiment_id": P75_EXPERIMENT,
        "positions": "all decoder-block output positions",
        "layers": {
            str(layer): {
                "threshold": cap.threshold,
                "direction_sha256": direction_sha256(cap),
                "axis_cosine": orientation[layer],
            }
            for layer, cap in sorted(caps.items())
        },
    }
    metadata["parsed_experiment_sha256"] = object_sha256(metadata["layers"])
    return caps, metadata


def build_configuration(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    diff_audit: dict[str, Any],
    p75_metadata: dict[str, Any],
) -> dict[str, Any]:
    visible_gpus = torch.cuda.device_count()
    return {
        "kind": "M2.2b v4 neutral-prompt paired calibration",
        "scope": "development calibration only; not held-out or confirmatory",
        "pairing": {
            "conditions": CONDITIONS,
            "active_condition": args.condition,
            "only_intervention_may_differ": True,
        },
        "dataset": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "source_sha256": diff_audit["source_sha256"],
            "diff_audit_path": str(args.diff_audit),
            "diff_audit_sha256": file_sha256(args.diff_audit),
            "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
            "rows": len(rows),
            "base_units": len({row["base_id"] for row in rows}),
            "arms": ARMS,
            "row_ids_sha256": object_sha256([row["id"] for row in rows]),
        },
        "model": asdict(QwenModelSpec()),
        "intervention": {
            "condition": args.condition,
            "active": args.condition == "p75",
            "official_p75_contract": p75_metadata,
            "implementation": "ActivationCapper on decoder block outputs",
            "stats_definition": (
                "hook-visible layer-token positions; pooled rate is triggered positions "
                "divided by total positions across layers 46-53; fixed batch-2 padding "
                "positions are included"
            ),
        },
        "messages": {
            "role_order": ("system", "user", "assistant", "user"),
            "content_fields": MESSAGE_FIELDS,
            "initial_assistant": "fixed transcript from dataset; never regenerated",
            "enable_thinking": True,
            "complete_raw_continuation": True,
        },
        "generation": {
            "decoding": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
            "seed": 0,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "output": "raw decoded continuation; no parsing, content retry, or repair",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "hardware": {
            "visible_gpu_count": visible_gpus,
            "visible_gpu_names": [torch.cuda.get_device_name(i) for i in range(visible_gpus)],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "qwen_max_memory_per_gpu_gib": args.gpu_memory_gib,
        },
        "environment": {
            "python_environment_name": Path(sys.prefix).name,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_prefix": sys.prefix,
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "git": git_provenance(allow_dirty=args.allow_dirty),
    }


def fixed_batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def batch_key(index: int, batch: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    return {
        "batch_index": index,
        "ids": [row["id"] for row in batch],
        "condition": condition,
    }


def stats_to_dict(stats: dict[int, CapStats]) -> dict[str, Any]:
    by_layer = {
        str(layer): {
            "total_positions": value.total_positions,
            "triggered_positions": value.triggered_positions,
            "trigger_rate": value.trigger_rate,
            "projection_correction_sum": value.projection_correction_sum,
            "mean_projection_correction": value.mean_projection_correction,
        }
        for layer, value in sorted(stats.items())
    }
    total = sum(value.total_positions for value in stats.values())
    triggered = sum(value.triggered_positions for value in stats.values())
    correction = sum(value.projection_correction_sum for value in stats.values())
    return {
        "definition": (
            "decoder-layer token positions seen by hooks, pooled across layers 46-53; "
            "fixed batch-2 padding positions are included"
        ),
        "by_layer": by_layer,
        "pooled": {
            "total_positions": total,
            "triggered_positions": triggered,
            "trigger_rate": triggered / total if total else 0.0,
            "projection_correction_sum": correction,
            "mean_projection_correction": correction / total if total else 0.0,
        },
    }


def aggregate_cap_stats(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    cap_payloads = [payload["cap_stats"] for payload in payloads if payload.get("cap_stats")]
    if not cap_payloads:
        return None
    layers = sorted(cap_payloads[0]["by_layer"], key=int)
    by_layer: dict[str, dict[str, float | int]] = {}
    for layer in layers:
        total = sum(item["by_layer"][layer]["total_positions"] for item in cap_payloads)
        triggered = sum(
            item["by_layer"][layer]["triggered_positions"] for item in cap_payloads
        )
        correction = sum(
            item["by_layer"][layer]["projection_correction_sum"]
            for item in cap_payloads
        )
        by_layer[layer] = {
            "total_positions": total,
            "triggered_positions": triggered,
            "trigger_rate": triggered / total if total else 0.0,
            "projection_correction_sum": correction,
            "mean_projection_correction": correction / total if total else 0.0,
        }
    pooled_total = sum(int(item["total_positions"]) for item in by_layer.values())
    pooled_triggered = sum(int(item["triggered_positions"]) for item in by_layer.values())
    pooled_correction = sum(float(item["projection_correction_sum"]) for item in by_layer.values())
    return {
        "definition": cap_payloads[0]["definition"],
        "batches": len(cap_payloads),
        "by_layer": by_layer,
        "pooled": {
            "total_positions": pooled_total,
            "triggered_positions": pooled_triggered,
            "trigger_rate": pooled_triggered / pooled_total if pooled_total else 0.0,
            "projection_correction_sum": pooled_correction,
            "mean_projection_correction": (
                pooled_correction / pooled_total if pooled_total else 0.0
            ),
        },
    }


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_outputs(
    store: RecordStore,
    batches: list[list[dict[str, Any]]],
    condition: str,
    results_path: Path,
    batch_runtime_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        payload = store.read(batch_key(index, batch, condition))
        if payload is None:
            continue
        expected_ids = [row["id"] for row in batch]
        if payload.get("batch_index") != index or payload.get("ids") != expected_ids:
            raise ValueError(f"Batch checkpoint {index} does not match frozen input")
        batch_results = payload.get("results")
        result_ids = (
            [row.get("id") for row in batch_results]
            if isinstance(batch_results, list)
            else []
        )
        if not isinstance(batch_results, list) or result_ids != expected_ids:
            raise ValueError(f"Batch checkpoint {index} has malformed results")
        if any(
            row.get("cap") != condition
            or row.get("thinking_enabled") is not True
            or row.get("seed") != 0
            or not isinstance(row.get("response"), str)
            for row in batch_results
        ):
            raise ValueError(f"Batch checkpoint {index} has malformed result metadata")
        results.extend(batch_results)
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    runtime_rows = [
        {key: value for key, value in payload.items() if key != "results"}
        for payload in payloads
    ]
    atomic_write_jsonl(batch_runtime_path, runtime_rows)
    return results, payloads


def update_runtime(run_dir: Path, values: dict[str, Any]) -> None:
    path = run_dir / "runtime.json"
    runtime = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    runtime.update(values)
    atomic_write_json(path, runtime)


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: list[dict[str, Any]],
    condition: str,
    caps: dict[int, LayerCap],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompts = [render_prompt(tokenizer, row) for row in batch]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    prompt_nonpadding_tokens = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
    prompt_width = int(encoded["input_ids"].shape[1])
    device = model_input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}

    capper: ActivationCapper | None = None
    if condition == "p75":
        capper = ActivationCapper(
            model,
            caps,
            positions="all",
            track_stats=True,
            defer_stats_sync=True,
        )
        context = capper
    else:
        from contextlib import nullcontext

        context = nullcontext()

    with context:
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
    responses = tokenizer.batch_decode(
        output_ids[:, prompt_width:],
        skip_special_tokens=True,
    )
    result_rows = [
        {
            "id": row["id"],
            "response": response,
            "cap": condition,
            "thinking_enabled": True,
            "seed": 0,
        }
        for row, response in zip(batch, responses, strict=True)
    ]
    return {
        "ids": [row["id"] for row in batch],
        "prompt_padded_width": prompt_width,
        "prompt_nonpadding_tokens": prompt_nonpadding_tokens,
        "response_tokens": [
            len(tokenizer.encode(response, add_special_tokens=False)) for response in responses
        ],
        "cap_stats": stats_to_dict(capper.stats) if capper is not None else None,
        "results": result_rows,
    }


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"Paired calibration must run inside cedar, found {sys.prefix}")
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f"Paired calibration freezes batch size at {EXPECTED_BATCH_SIZE}")
    if args.max_new_tokens != EXPECTED_MAX_NEW_TOKENS:
        raise ValueError(
            f"Paired calibration freezes max_new_tokens at {EXPECTED_MAX_NEW_TOKENS}"
        )
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    transformers.set_seed(0)

    rows = load_rows(args.data)
    diff_audit = validate_diff_audit(args.diff_audit, args.data)
    caps, p75_metadata = load_p75_contract(args.cache_dir)
    run_dir = args.output_root / RUN_NAMES[args.condition]
    configuration = build_configuration(args, rows, diff_audit, p75_metadata)
    manifest = initialize_run(run_dir, configuration)
    store = RecordStore(run_dir / "batches", run_fingerprint=str(manifest["fingerprint"]))
    batches = fixed_batches(rows, args.batch_size)
    results_path = run_dir / "qwen_thinking_results.jsonl"
    batch_runtime_path = run_dir / "batch_runtime.jsonl"
    results, payloads = materialize_outputs(
        store,
        batches,
        args.condition,
        results_path,
        batch_runtime_path,
    )

    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "condition": args.condition,
                    "rows": len(rows),
                    "batches": len(batches),
                    "existing_results": len(results),
                    "dataset_sha256": file_sha256(args.data),
                    "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
                    "p75_config_sha256": p75_metadata["capping_config_sha256"],
                    "p75_layers": sorted(int(layer) for layer in p75_metadata["layers"]),
                    "run_fingerprint": manifest["fingerprint"],
                }
            ),
            flush=True,
        )

    if args.stage in {"generate", "all"}:
        newly_generated = 0
        missing_indices = [
            index
            for index, batch in enumerate(batches)
            if not store.contains(batch_key(index, batch, args.condition))
        ]
        if missing_indices:
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen3-32B paired calibration requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB"
                for index in range(torch.cuda.device_count())
            }
            update_runtime(
                run_dir,
                {
                    "generation_started_at": datetime.now().astimezone().isoformat(),
                    "condition": args.condition,
                },
            )
            model, tokenizer = load_qwen_model(max_memory=max_memory, local_files_only=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            rendered = [render_prompt(tokenizer, row) for row in rows]
            tokenized = [
                tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered
            ]
            update_runtime(
                run_dir,
                {
                    "device_map": getattr(model, "hf_device_map", None),
                    "rendered_prompts_sha256": object_sha256(
                        [
                            {"id": row["id"], "prompt": prompt}
                            for row, prompt in zip(rows, rendered, strict=True)
                        ]
                    ),
                    "tokenized_input_ids_sha256": object_sha256(
                        [
                            {"id": row["id"], "input_ids": input_ids}
                            for row, input_ids in zip(rows, tokenized, strict=True)
                        ]
                    ),
                },
            )
            for index in missing_indices:
                batch = batches[index]
                generated = generate_batch(
                    model,
                    tokenizer,
                    batch,
                    args.condition,
                    caps,
                    max_new_tokens=args.max_new_tokens,
                )
                payload = {
                    "batch_index": index,
                    "condition": args.condition,
                    **generated,
                }
                store.write(batch_key(index, batch, args.condition), payload)
                newly_generated += len(batch)
                results, payloads = materialize_outputs(
                    store,
                    batches,
                    args.condition,
                    results_path,
                    batch_runtime_path,
                )
                if index == 0:
                    smoke = thinking_smoke_check(results[0]["response"])
                    update_runtime(run_dir, {"thinking_smoke": {"id": results[0]["id"], **smoke}})
                    if not smoke["passed"]:
                        del model, tokenizer
                        release_cuda_cache()
                        raise RuntimeError("Thinking smoke check failed; full run stopped")
                print(
                    json.dumps(
                        {
                            "stage": "generate",
                            "condition": args.condition,
                            "batch": index + 1,
                            "batches": len(batches),
                            "generated": len(batch),
                            "complete": len(results),
                        }
                    ),
                    flush=True,
                )
            del model, tokenizer
            release_cuda_cache()

        results, payloads = materialize_outputs(
            store,
            batches,
            args.condition,
            results_path,
            batch_runtime_path,
        )
        if results:
            smoke = thinking_smoke_check(results[0]["response"])
            if not smoke["passed"]:
                raise RuntimeError("Stored smoke response lacks complete thinking output")
        if len(results) == EXPECTED_ROWS:
            runtime_values: dict[str, Any] = {
                "complete_results": len(results),
                "complete_batches": len(payloads),
                "results_sha256": file_sha256(results_path),
                "batch_runtime_sha256": file_sha256(batch_runtime_path),
                "p75_trigger_stats": aggregate_cap_stats(payloads),
            }
            if newly_generated:
                runtime_values["generation_completed_at"] = (
                    datetime.now().astimezone().isoformat()
                )
            update_runtime(run_dir, runtime_values)
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "results": str(results_path),
                    "complete": len(results),
                    "newly_generated": newly_generated,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
