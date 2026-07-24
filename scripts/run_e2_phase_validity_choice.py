#!/usr/bin/env python3
"""Run the frozen E2A Answer/Validity phase-localization matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "datasets" / "m2_2b_formal" / "e2_phase_validity_choice"
EVIDENCE_DATA = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_evidence_validation_v1_1"
for path in (SRC, DATA, EVIDENCE_DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_prover_generator_step5_uncapped as base  # noqa: E402
from e2_common import (  # noqa: E402
    ANCHOR_RESULT_COMMIT,
    ATTENTION_IMPLEMENTATION,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    FORMAL_CONDITIONS,
    MAX_NEW_TOKENS,
    MODEL_DTYPE,
    MODEL_REPO,
    MODEL_REVISION,
    SEED,
    SMOKE_CONDITIONS,
    VALIDITY_LABELS,
    VALIDITY_PREFIX,
    canonical_text_sha256,
    direction_sha256,
    read_json,
    read_jsonl,
    require,
    resolve_hub_cache,
    validate_anchor,
)
from e2_common import (  # noqa: E402
    file_sha256 as data_file_sha256,
)

from cedar_m2.assets.capping_config import (  # noqa: E402
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.checkpoint import (  # noqa: E402
    RecordStore,
    atomic_write_json,
    file_sha256,
    initialize_run,
    object_sha256,
)
from cedar_m2.interventions.capping import ActivationCapper  # noqa: E402
from cedar_m2.interventions.phase_capping import PhaseActivationCapper  # noqa: E402
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "e2_phase_validity_choice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--assay", choices=("answer", "validity", "all"), default="all")
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME"))
    )
    parser.add_argument("--asset-cache-dir", type=Path)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def git_provenance(*, require_synced: bool) -> dict[str, Any]:
    def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True, text=True)

    commit = command("rev-parse", "HEAD").stdout.strip()
    branch = command("branch", "--show-current").stdout.strip()
    status = command("status", "--porcelain").stdout.rstrip()
    upstream_result = command("rev-parse", "@{u}", check=False)
    upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
    ancestor = command("merge-base", "--is-ancestor", ANCHOR_RESULT_COMMIT, commit, check=False)
    require(ancestor.returncode == 0, "E2 runner HEAD must descend from the frozen anchor")
    if require_synced:
        require(not status, "E2 GPU runner requires a clean worktree")
        require(
            upstream is not None and upstream == commit, "E2 GPU runner requires HEAD == upstream"
        )
    return {
        "commit": commit,
        "branch": branch,
        "upstream_commit": upstream,
        "head_matches_upstream": upstream == commit if upstream else False,
        "dirty": bool(status),
        "anchor_is_ancestor": True,
    }


def validate_release() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_anchor()
    answer = read_jsonl(DATA / "answer_inputs96.jsonl")
    validity = read_jsonl(DATA / "validity_inputs48.jsonl")
    preflight = read_json(DATA / "tokenizer_preflight.json")
    review = read_json(DATA / "human_review.json")
    manifest = read_json(DATA / "manifest.json")
    gate = read_json(DATA / "E2_CPU_GATE.json")
    require(
        preflight.get("status") == "passed" and preflight.get("gpu_used") is False,
        "CPU preflight is not passed",
    )
    require(
        review.get("status") == "passed" and review.get("passed_items") == 48,
        "Validity review is not passed",
    )
    require(manifest.get("status") == "cpu_gate_ready", "E2 manifest is not ready")
    require(
        gate.get("gpu_authorized") is True and gate.get("authorized_scope") == "E2A_only",
        "E2 CPU Gate does not authorize E2A",
    )
    for relative, expected in manifest["release_files"].items():
        path = ROOT / relative
        require(path.is_file(), f"Missing E2 release file: {relative}")
        require(canonical_text_sha256(path) == expected, f"E2 release hash changed: {relative}")
    return answer, validity, preflight


def selected_rows(
    answer: Sequence[dict[str, Any]],
    validity: Sequence[dict[str, Any]],
    *,
    scope: str,
    assay: str,
    manifest: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    answer_rows = list(answer)
    validity_rows = list(validity)
    if scope == "smoke":
        answer_ids = set(manifest["smoke_ids"]["answer"])
        validity_ids = set(manifest["smoke_ids"]["validity"])
        answer_rows = [row for row in answer_rows if row["id"] in answer_ids]
        validity_rows = [row for row in validity_rows if row["id"] in validity_ids]
        require(len(answer_rows) == len(answer_ids), "Smoke Answer IDs changed")
        require(len(validity_rows) == len(validity_ids), "Smoke Validity IDs changed")
    rows: list[tuple[str, dict[str, Any]]] = []
    if assay in {"answer", "all"}:
        rows.extend(("answer", row) for row in answer_rows)
    if assay in {"validity", "all"}:
        rows.extend(("validity", row) for row in validity_rows)
    return rows


def cells(
    rows: Sequence[tuple[str, dict[str, Any]]], scope: str
) -> list[tuple[str, dict[str, Any], str]]:
    conditions = SMOKE_CONDITIONS if scope == "smoke" else FORMAL_CONDITIONS
    return [(assay, row, condition) for assay, row in rows for condition in conditions]


def hook_counts(model: Any) -> dict[str, int]:
    require(hasattr(model, "model") and hasattr(model.model, "layers"), "Qwen layers unavailable")
    return {str(layer): len(model.model.layers[layer]._forward_hooks) for layer in CAP_LAYERS}


def full_stats(stats: Mapping[int, Any]) -> dict[str, Any]:
    return {
        "by_layer": {
            str(layer): {
                "total_positions": value.total_positions,
                "triggered_positions": value.triggered_positions,
                "trigger_rate": value.trigger_rate,
                "projection_correction_sum": value.projection_correction_sum,
                "mean_projection_correction": value.mean_projection_correction,
            }
            for layer, value in sorted(stats.items())
        }
    }


def load_p25_assets(cache_dir: Path) -> tuple[dict[int, Any], dict[str, Any]]:
    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=resolve_hub_cache(cache_dir), local_files_only=True
    )
    axis_hash = data_file_sha256(axis_path)
    config_hash = data_file_sha256(config_path)
    require(axis_hash == EXPECTED_AXIS_SHA256, "Assistant Axis hash changed")
    require(config_hash == EXPECTED_CAP_CONFIG_SHA256, "p25 config hash changed")
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, CAP_EXPERIMENT)
    require(tuple(sorted(caps)) == CAP_LAYERS, "p25 layer range changed")
    orientation = verify_axis_orientation(caps, axis)
    return caps, {
        "axis_sha256": axis_hash,
        "capping_config_sha256": config_hash,
        "experiment": CAP_EXPERIMENT,
        "layers": {
            str(layer): {
                "threshold": caps[layer].threshold,
                "direction_sha256": direction_sha256(caps[layer]),
                "axis_cosine": orientation[layer],
            }
            for layer in CAP_LAYERS
        },
    }


def option_diagnostics(
    logits: Tensor, labels: Sequence[str], token_ids: Mapping[str, int]
) -> dict[str, Any]:
    values = logits.detach().float()
    require(values.ndim == 1 and bool(torch.isfinite(values).all()), "Malformed logits")
    log_probs = values - torch.logsumexp(values, dim=-1)
    probabilities = torch.exp(log_probs)
    option_logits = {label: float(values[token_ids[label]].item()) for label in labels}
    option_probabilities = {
        label: float(probabilities[token_ids[label]].item()) for label in labels
    }
    option_mass = sum(option_probabilities.values())
    return {
        "logits": option_logits,
        "full_vocab_probabilities": option_probabilities,
        "conditional_probabilities": {
            label: probability / option_mass if option_mass else 0.0
            for label, probability in option_probabilities.items()
        },
        "ranks": {
            label: int(torch.count_nonzero(values > values[token_ids[label]]).item()) + 1
            for label in labels
        },
        "option_argmax": max(labels, key=option_logits.__getitem__),
        "option_mass": option_mass,
        "full_vocab_entropy": float((-(probabilities * log_probs).sum()).item()),
        "full_vocab_top_token_id": int(torch.argmax(values).item()),
    }


class PrefixLogitCapture:
    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_tokens: int,
        prefix: str,
        labels: Sequence[str],
        token_ids: Mapping[str, int],
    ) -> None:
        self.prompt_tokens = prompt_tokens
        variants = (
            prefix,
            '{"judgment": "',
            '{ "judgment":"',
            '{ "judgment": "',
        )
        self.prefix_variants = tuple(
            tuple(tokenizer.encode(value, add_special_tokens=False)) for value in variants
        )
        self.labels = tuple(labels)
        self.token_ids = dict(token_ids)
        self.by_response_index: dict[int, dict[str, Any]] = {}

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        sequence = tuple(int(value) for value in input_ids[0].tolist())
        if any(
            len(sequence) >= len(prefix_ids) and sequence[-len(prefix_ids) :] == prefix_ids
            for prefix_ids in self.prefix_variants
        ):
            index = len(sequence) - self.prompt_tokens
            self.by_response_index[index] = option_diagnostics(
                scores[0], self.labels, self.token_ids
            )
        return scores


def strict_validity(
    response: str, *, truncated: bool
) -> tuple[str | None, str, str, dict[str, Any]]:
    reasoning, final, sections = base.response_sections(response)
    if truncated:
        return None, "truncated", reasoning, sections
    try:
        payload = json.loads(final)
    except json.JSONDecodeError:
        return None, "invalid_json", reasoning, sections
    if not isinstance(payload, dict) or set(payload) != {"judgment"}:
        return None, "wrong_json_schema", reasoning, sections
    if payload.get("judgment") not in VALIDITY_LABELS:
        return None, "judgment_not_AB", reasoning, sections
    return str(payload["judgment"]), "passed", reasoning, sections


def generate_validity(
    model: nn.Module,
    tokenizer: Any,
    transformers_module: Any,
    row: dict[str, Any],
    *,
    record_seed: int,
    token_ids: Mapping[str, int],
) -> dict[str, Any]:
    mode = base.MODES["thinking"]
    prompt = base.render_prompt(tokenizer, row, mode)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    require(
        tokenizer.encode(prompt, add_special_tokens=False) == prompt_ids,
        f"{row['id']}: tokenization changed",
    )
    encoded = {name: value.to(model_input_device(model)) for name, value in encoded.items()}
    capture = PrefixLogitCapture(
        tokenizer,
        prompt_tokens=len(prompt_ids),
        prefix=VALIDITY_PREFIX,
        labels=VALIDITY_LABELS,
        token_ids=token_ids,
    )
    transformers_module.set_seed(record_seed)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **base.generation_kwargs(tokenizer),
            logits_processor=transformers_module.LogitsProcessorList([capture]),
        )
    response_ids = [int(value) for value in generated.sequences[0, len(prompt_ids) :].tolist()]
    response = tokenizer.decode(
        response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    reason = base.finish_reason(response_ids, tokenizer)
    judgment, parse_status, reasoning, sections = strict_validity(
        response, truncated=reason == "length"
    )
    candidates = [
        {"response_index": index, "generated_label": label}
        for index in sorted(capture.by_response_index)
        if 0 <= index < len(response_ids)
        for label, token_id in token_ids.items()
        if response_ids[index] == token_id
    ]
    matching = [item for item in candidates if item["generated_label"] == judgment]
    selected = matching[-1] if matching else None
    diagnostics = capture.by_response_index[selected["response_index"]] if selected else None
    if judgment is not None and diagnostics is None:
        raise RuntimeError(f"{row['id']}: strict Validity judgment lacks aligned logits")
    semantic = None
    if judgment is not None:
        semantic = "sufficient" if judgment == row["sufficient_label"] else "insufficient"
    return {
        **{
            key: row.get(key)
            for key in (
                "id",
                "source_answer_id",
                "base_id",
                "family_id",
                "structure_class",
                "variant",
                "source_arm",
                "position_transition",
                "error_class",
                "error_position",
                "sufficient_label",
                "insufficient_label",
                "gold_semantic",
                "gold_label",
            )
        },
        "assay": "validity",
        "scope": "e2_phase_validity_choice",
        "mode": "thinking",
        "seed": SEED,
        "record_seed": record_seed,
        "prompt_sha256": object_sha256(prompt),
        "prompt_token_ids_sha256": object_sha256(prompt_ids),
        "prompt_tokens": len(prompt_ids),
        "response": response,
        "response_token_ids_sha256": object_sha256(response_ids),
        "response_tokens": len(response_ids),
        "finish_reason": reason,
        "truncated": reason == "length",
        "reasoning_token_count": len(tokenizer.encode(reasoning, add_special_tokens=False)),
        "mode_integrity": base.mode_integrity(mode, sections),
        "chosen_label": judgment,
        "chosen_semantic": semantic,
        "gold_correct": judgment == row["gold_label"],
        "final_answer_parse_status": parse_status,
        "assistant_prefix": VALIDITY_PREFIX,
        "candidate_token_ids": dict(token_ids),
        "answer_logits": diagnostics,
        "answer_logit_capture_indices": sorted(capture.by_response_index),
        "selected_answer_logit_index": selected["response_index"] if selected else None,
        "answer_logit_candidates": candidates,
        "content_retry_count": 0,
        "answer_repair_performed": False,
    }


def run_one(
    model: nn.Module,
    tokenizer: Any,
    transformers_module: Any,
    caps: Mapping[int, Any],
    assay: str,
    row: dict[str, Any],
    condition: str,
    *,
    record_seed: int,
    validity_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    before = hook_counts(model)
    phase_capper: PhaseActivationCapper | None = None
    full_capper: ActivationCapper | None = None
    if condition == "full_p25":
        phase_capper = PhaseActivationCapper(
            model,
            caps,
            enabled_phases=("prefill", "decode"),
            apply_changes=False,
            defer_stats_sync=True,
        )
        full_capper = ActivationCapper(
            model, caps, positions="all", track_stats=True, defer_stats_sync=True
        )
    elif condition == "prefill_only":
        phase_capper = PhaseActivationCapper(
            model, caps, enabled_phases=("prefill",), defer_stats_sync=True
        )
    elif condition == "decode_only":
        phase_capper = PhaseActivationCapper(
            model, caps, enabled_phases=("decode",), defer_stats_sync=True
        )
    elif condition == "monitor_only":
        phase_capper = PhaseActivationCapper(
            model,
            caps,
            enabled_phases=("prefill", "decode"),
            apply_changes=False,
            defer_stats_sync=True,
        )
    elif condition not in {"uncapped", "nohook"}:
        raise ValueError(f"Unknown E2 condition: {condition}")

    def generate() -> dict[str, Any]:
        if assay == "answer":
            result = base.generate_one(
                model,
                tokenizer,
                transformers_module,
                row,
                scope="e2-phase-answer",
                mode_name="thinking",
                record_seed=record_seed,
            )
            result.update(
                {
                    "assay": "answer",
                    "position_transition": row["position_transition"],
                    "development_split": row["development_split"],
                }
            )
            return result
        return generate_validity(
            model,
            tokenizer,
            transformers_module,
            row,
            record_seed=record_seed,
            token_ids=validity_token_ids,
        )

    if phase_capper is not None and full_capper is not None:
        with phase_capper:
            with full_capper:
                result = generate()
    elif phase_capper is not None:
        with phase_capper:
            result = generate()
    else:
        result = generate()
    after = hook_counts(model)
    require(before == after, f"Hook leak: before={before}, after={after}")
    result.update(
        {
            "condition": condition,
            "phase_telemetry": phase_capper.telemetry() if phase_capper else None,
            "full_cap_stats": full_stats(full_capper.stats) if full_capper else None,
            "hook_counts_before": before,
            "hook_counts_after": after,
        }
    )
    return result


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_key(index: int, assay: str, row: Mapping[str, Any], condition: str) -> dict[str, Any]:
    return {"index": index, "assay": assay, "id": row["id"], "condition": condition, "seed": SEED}


def materialize(
    store: RecordStore,
    matrix: Sequence[tuple[str, dict[str, Any], str]],
    path: Path,
) -> list[dict[str, Any]]:
    results = []
    for index, (assay, row, condition) in enumerate(matrix):
        payload = store.read(record_key(index, assay, row, condition))
        if payload is not None:
            require(payload.get("index") == index, f"Checkpoint index mismatch: {index}")
            results.append(payload["result"])
    atomic_write_jsonl(path, results)
    return results


def main() -> None:
    args = parse_args()
    require(args.gpu_memory_gib > 0, "GPU memory must be positive")
    answer, validity, preflight = validate_release()
    manifest = read_json(DATA / "manifest.json")
    rows = selected_rows(answer, validity, scope=args.scope, assay=args.assay, manifest=manifest)
    matrix = cells(rows, args.scope)
    git = git_provenance(require_synced=args.stage in {"generate", "all"})
    run_dir = args.output_root / f"{args.scope}-{args.assay}-seed0"
    configuration = {
        "schema": "cedar.e2_phase_validity_choice.run.v1",
        "scope": args.scope,
        "assay": args.assay,
        "rows": len(rows),
        "cells": len(matrix),
        "conditions": list(SMOKE_CONDITIONS if args.scope == "smoke" else FORMAL_CONDITIONS),
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "generation": {
            "thinking": True,
            "greedy": True,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "retry": False,
            "repair": False,
        },
        "git": git,
    }
    initialized = initialize_run(run_dir, configuration)
    store = RecordStore(run_dir / "records", run_fingerprint=str(initialized["fingerprint"]))
    results_path = run_dir / "results.jsonl"
    results = materialize(store, matrix, results_path)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "rows": len(rows),
                "cells": len(matrix),
                "existing": len(results),
                "gpu_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.stage == "prepare":
        return

    import transformers

    require(Path(sys.prefix).name == "cedar", f"Generation requires cedar, found {sys.prefix}")
    require(torch.cuda.device_count() == 4, "E2A requires exactly four visible GPUs")
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    base.MAX_NEW_TOKENS = MAX_NEW_TOKENS
    base.SEED = SEED
    cache = args.asset_cache_dir.resolve() if args.asset_cache_dir else args.hf_home.resolve()
    caps, cap_metadata = load_p25_assets(cache)
    spec = QwenModelSpec(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )
    model, tokenizer = load_qwen_model(
        spec,
        max_memory={index: f"{args.gpu_memory_gib}GiB" for index in range(4)},
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base.validate_label_tokens(tokenizer)
    validity_token_ids = {
        label: int(preflight["validity_labels"]["labels"][label]["contextual_suffix_ids"][0])
        for label in VALIDITY_LABELS
    }
    runtime_path = run_dir / "runtime.json"
    atomic_write_json(
        runtime_path,
        {
            "generation_started_at": datetime.now().astimezone().isoformat(),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "visible_gpu_count": torch.cuda.device_count(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "official_p25": cap_metadata,
        },
    )
    for index, (assay, row, condition) in enumerate(matrix):
        key = record_key(index, assay, row, condition)
        if store.contains(key):
            continue
        result = run_one(
            model,
            tokenizer,
            transformers,
            caps,
            assay,
            row,
            condition,
            record_seed=index
            // len(SMOKE_CONDITIONS if args.scope == "smoke" else FORMAL_CONDITIONS),
            validity_token_ids=validity_token_ids,
        )
        store.write(key, {"index": index, "result": result})
        results = materialize(store, matrix, results_path)
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "complete": len(results),
                    "expected": len(matrix),
                    "assay": assay,
                    "id": row["id"],
                    "condition": condition,
                }
            ),
            flush=True,
        )
    results = materialize(store, matrix, results_path)
    runtime = read_json(runtime_path)
    runtime.update(
        {
            "generation_completed_at": datetime.now().astimezone().isoformat(),
            "complete_results": len(results),
            "results_sha256": file_sha256(results_path),
            "truncations": [
                f"{row['id']}::{row['condition']}" for row in results if row.get("truncated")
            ],
            "parse_failures": [
                f"{row['id']}::{row['condition']}"
                for row in results
                if row.get("final_answer_parse_status") != "passed"
            ],
            "missing_logits": [
                f"{row['id']}::{row['condition']}"
                for row in results
                if row.get("answer_logits") is None
            ],
        }
    )
    atomic_write_json(runtime_path, runtime)
    print(json.dumps({"complete": len(results), "expected": len(matrix)}), flush=True)


if __name__ == "__main__":
    main()
