#!/usr/bin/env python3
"""Run one frozen C3 direct forced-choice phase with resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from cedar_m2.checkpoint import (  # noqa: E402
    RecordStore,
    file_sha256,
    initialize_run,
    object_sha256,
)
from cedar_m2.interventions.capping import ActivationCapper, LayerCap  # noqa: E402
from cedar_m2.models.qwen import (  # noqa: E402
    QwenModelSpec,
    load_qwen_model,
    model_input_device,
)
from scripts.c3_portable_hashes import (  # noqa: E402
    canonical_text_sha256,
    file_identity,
    portable_identity_matches,
    read_jsonl,
)
from scripts.run_c2_v8_natural_confirmation import (  # noqa: E402
    SegmentedAxisMonitor,
    actual_cap_by_phase,
    aggregate_run_cap_stats,
    cap_stats_to_dict,
    hook_counts,
    load_cap_assets,
    render_direct,
)
from scripts.run_m2_2b_v4_neutral_pair import (  # noqa: E402
    atomic_write_jsonl,
    release_cuda_cache,
    update_runtime,
)

DATA_ROOT = ROOT / "datasets" / "c3_sycobench_bidirectional_selectivity"
ARMS_PATH = DATA_ROOT / "c3_arms2400.jsonl"
SMOKE_PATH = DATA_ROOT / "c3_smoke.jsonl"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
CONSTRUCTION_AUDIT_PATH = DATA_ROOT / "construction_audit.json"
REVIEW_GATE_PATH = DATA_ROOT / "REVIEW_GATE.json"
PREFLIGHT_PATH = DATA_ROOT / "execution_preflight_manifest.json"
VALIDATOR_PATH = DATA_ROOT / "validate_c3_sycobench_bidirectional.py"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c3_sycobench_bidirectional_selectivity"

LABELS = ("A", "B", "C", "D")
LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
ASSISTANT_PREFIX = '{"answer":"'
LAYERS = tuple(range(46, 54))
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
CAP_EXPERIMENT = "layers_46:54-p0.25"
EXPECTED_AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
EXPECTED_CAP_CONFIG_SHA256 = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
EXPECTED_CONTENT_HASHES = {
    "bases": "dbb4a3798c729c79271b06702e9cf712bcf4b18f1766eff9a0f2033e7f4fa205",
    "arms": "edd16f21bb8f0b647bd77fb8faa50a50dc2db8c941d264145efaa2bf9a8949a6",
    "smoke": "c1e56e7827a6c1ce20548462b8c7afc6557c51e1efd651e7a5362003073d84c2",
}
REQUIRED_PREFLIGHT_FILES = {
    ".gitattributes",
    "docs/C3_SYCOBENCH_BIDIRECTIONAL_SELECTIVITY_PROTOCOL.md",
    "docs/C3_PROTOCOL_AMENDMENT_01_AGENT_SELF_REVIEW_WAIVER.md",
    "datasets/c3_sycobench_bidirectional_selectivity/manifest.json",
    "datasets/c3_sycobench_bidirectional_selectivity/construction_audit.json",
    "datasets/c3_sycobench_bidirectional_selectivity/REVIEW_GATE.json",
    "datasets/c3_sycobench_bidirectional_selectivity/c3_base600.jsonl",
    "datasets/c3_sycobench_bidirectional_selectivity/c3_arms2400.jsonl",
    "datasets/c3_sycobench_bidirectional_selectivity/c3_smoke.jsonl",
    "datasets/c3_sycobench_bidirectional_selectivity/review/agent_review_completed.csv",
    "datasets/c3_sycobench_bidirectional_selectivity/review/template_agent_review_completed.csv",
    "datasets/c3_sycobench_bidirectional_selectivity/review/agent_self_review_attestation.json",
    "scripts/run_c3_sycobench_bidirectional.py",
    "scripts/audit_c3_sycobench_bidirectional.py",
    "scripts/analyze_c3_sycobench_bidirectional.py",
}


@dataclass(frozen=True)
class RunSpec:
    scope: str
    requested_condition: str
    condition: str
    monitor_enabled: bool
    selection: str
    phase_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--condition", choices=("nohook", "uncapped", "p25"), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path("/data1/WuMingrui/HF_HOME/hub"))
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def resolve_spec(scope: str, requested_condition: str) -> RunSpec:
    if scope == "formal" and requested_condition == "nohook":
        raise ValueError("Formal C3 only permits uncapped monitor-only or p25")
    condition = "p25" if requested_condition == "p25" else "uncapped"
    monitor_enabled = requested_condition != "nohook"
    if scope == "smoke":
        suffix = "monitor" if requested_condition == "uncapped" else requested_condition
        phase = f"smoke-{suffix}"
        selection = "smoke96"
    else:
        phase = f"formal-{requested_condition}"
        selection = "all2400"
    return RunSpec(scope, requested_condition, condition, monitor_enabled, selection, phase)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("c3_runtime_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load C3 validator: {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_execution_preflight() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise ValueError("C3 GPU is blocked: execution_preflight_manifest.json is missing")
    preflight = read_json(PREFLIGHT_PATH)
    if preflight.get("status") != "zero_gpu_code_frozen_agent_self_review_authorized":
        raise ValueError("C3 execution preflight is not an authorized zero-GPU freeze")
    if (
        preflight.get("gpu_or_qwen_execution_performed") is not False
        or preflight.get("formal_execution_performed") is not False
        or preflight.get("review_gate") != "passed_agent_self_review_user_waiver"
        or preflight.get("data_content_hashes") != EXPECTED_CONTENT_HASHES
        or preflight.get("zero_gpu_validation", {}).get("status") != "passed"
    ):
        raise ValueError("C3 execution preflight contract changed")
    if preflight.get("review_disclosure") != {
        "review_mode": "agent_self_review_user_waiver",
        "protocol_deviation": True,
        "human_review_performed": False,
        "data_lead_review_performed": False,
        "user_review_waived": True,
        "dual_independent_review": False,
        "raw_agreement": None,
        "cohen_kappa": None,
    }:
        raise ValueError("C3 preflight amended-review disclosure changed")
    model = preflight.get("model_contract", {})
    expected_model = {
        "repo": "Qwen/Qwen3-32B",
        "revision": MODEL_REVISION,
        "enable_thinking": False,
        "candidate_token_ids": LABEL_TOKEN_IDS,
        "p25_experiment": CAP_EXPERIMENT,
        "layers": list(LAYERS),
        "axis_sha256": EXPECTED_AXIS_SHA256,
        "capping_config_sha256": EXPECTED_CAP_CONFIG_SHA256,
    }
    if model != expected_model:
        raise ValueError("C3 preflight model/capping contract changed")
    files = preflight.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("C3 preflight file inventory is missing")
    if not REQUIRED_PREFLIGHT_FILES.issubset(files):
        raise ValueError("C3 preflight omits required data, review, or runtime files")
    for relative, expected in files.items():
        path = ROOT / relative
        observed = file_identity(path, root=ROOT) if path.is_file() else {}
        if not path.is_file() or not portable_identity_matches(observed, expected):
            raise ValueError(f"C3 preflight file identity mismatch: {relative}")
    return preflight


def validate_review_gate() -> dict[str, Any]:
    if not REVIEW_GATE_PATH.is_file():
        raise ValueError("C3 GPU is blocked: REVIEW_GATE.json is missing")
    validator = load_validator()
    expected = validator.validate_review_gate(validator.load_builder(), enforce_pre_artifact=False)
    observed = read_json(REVIEW_GATE_PATH)
    if observed != expected or observed.get("gpu_authorized") is not True:
        raise ValueError("C3 review gate is stale or does not authorize GPU")
    if (
        observed.get("review_mode") != "agent_self_review_user_waiver"
        or observed.get("protocol_deviation") is not True
        or observed.get("human_review_performed") is not False
        or observed.get("data_lead_review_performed") is not False
        or observed.get("user_review_waived") is not True
        or observed.get("dual_independent_review") is not False
        or observed.get("cohen_kappa") is not None
    ):
        raise ValueError("C3 amended review disclosure changed")
    return observed


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.rstrip()

    status = git("status", "--porcelain")
    if status:
        raise RuntimeError("C3 GPU is blocked: worktree is dirty")
    commit = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{u}")
    branch = git("branch", "--show-current")
    if commit != upstream:
        raise RuntimeError("C3 GPU is blocked: local HEAD differs from upstream")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": False,
        "upstream_commit": upstream,
        "head_matches_upstream": True,
    }


def validate_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    preflight = validate_execution_preflight()
    gate = validate_review_gate()
    manifest = read_json(MANIFEST_PATH)
    if (
        manifest.get("status") != "passed"
        or manifest.get("gpu_authorized") is not False
        or manifest.get("content_hashes", {}).get("bases") != EXPECTED_CONTENT_HASHES["bases"]
        or manifest.get("content_hashes", {}).get("arms") != EXPECTED_CONTENT_HASHES["arms"]
        or manifest.get("content_hashes", {}).get("smoke") != EXPECTED_CONTENT_HASHES["smoke"]
    ):
        raise ValueError("C3 construction manifest changed")
    if (
        not CONSTRUCTION_AUDIT_PATH.is_file()
        or read_json(CONSTRUCTION_AUDIT_PATH).get("status") != "passed"
    ):
        raise ValueError("C3 passed construction audit is missing")
    rows = read_jsonl(ARMS_PATH)
    if object_sha256(rows) != EXPECTED_CONTENT_HASHES["arms"]:
        raise ValueError("C3 arm semantic hash changed")
    if len(rows) != 2400 or len({row.get("id") for row in rows}) != 2400:
        raise ValueError("C3 requires 2,400 unique rows")
    if {row.get("direction") for row in rows} != {"harmful", "helpful"}:
        raise ValueError("C3 direction set changed")
    if {row.get("arm") for row in rows} != {"AO", "S"}:
        raise ValueError("C3 arm set changed")
    for row in rows:
        if (
            row.get("candidate_labels") != list(LABELS)
            or row.get("initial_answer") not in LABELS
            or row.get("target_answer") not in LABELS
            or row["initial_answer"] == row["target_answer"]
        ):
            raise ValueError(f"{row.get('id')}: malformed answer mapping")
    return rows, gate, preflight


def select_rows(rows: Sequence[dict[str, Any]], spec: RunSpec) -> list[dict[str, Any]]:
    if spec.selection == "all2400":
        selected = list(rows)
        expected = 2400
    else:
        selected = read_jsonl(SMOKE_PATH)
        expected = 96
        if object_sha256(selected) != EXPECTED_CONTENT_HASHES["smoke"]:
            raise ValueError("C3 smoke semantic hash changed")
        arm_index = {row["id"]: row for row in rows}
        if any(arm_index.get(row["id"]) != row for row in selected):
            raise ValueError("C3 smoke rows are not exact members of the formal grid")
    if len(selected) != expected:
        raise ValueError(f"{spec.selection}: expected {expected} rows, found {len(selected)}")
    return selected


def completed_phase(output_root: Path, phase: str, expected: int) -> bool:
    runtime_path = output_root / phase / "runtime.json"
    results_path = output_root / phase / "results.jsonl"
    if not runtime_path.is_file() or not results_path.is_file():
        return False
    runtime = read_json(runtime_path)
    return (
        runtime.get("complete_results") == expected
        and runtime.get("content_retries") == 0
        and runtime.get("results_canonical_text_sha256") == canonical_text_sha256(results_path)
    )


def passed_audit(path: Path) -> bool:
    return path.is_file() and read_json(path).get("status") == "passed"


def validate_run_order(spec: RunSpec, output_root: Path) -> None:
    if spec.scope == "smoke":
        sequence = ("smoke-nohook", "smoke-monitor", "smoke-p25")
        position = sequence.index(spec.phase_name)
        if position and not completed_phase(output_root, sequence[position - 1], 96):
            raise ValueError(
                f"{spec.phase_name} is blocked until {sequence[position - 1]} completes"
            )
    elif spec.phase_name == "formal-uncapped":
        if not passed_audit(output_root / "SMOKE_AUDIT.json"):
            raise ValueError("Formal C3 is blocked until SMOKE_AUDIT.json passes")
    elif spec.phase_name == "formal-p25":
        if not completed_phase(output_root, "formal-uncapped", 2400):
            raise ValueError("C3 p25 is blocked until formal uncapped completes")
        if not passed_audit(output_root / "UNCAPPED_AUDIT.json"):
            raise ValueError("C3 p25 is blocked until UNCAPPED_AUDIT.json passes")


def validate_label_tokens(tokenizer: Any) -> None:
    observed = {label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS}
    if observed != {label: [token_id] for label, token_id in LABEL_TOKEN_IDS.items()}:
        raise RuntimeError(f"Qwen A/B/C/D token IDs changed: {observed}")
    if not tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False):
        raise RuntimeError("Frozen assistant prefix tokenizes to an empty sequence")


def choice_diagnostics(logits: Tensor, row: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    values = logits.detach().float()
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise FloatingPointError("C3 answer-step logits are malformed")
    log_z = torch.logsumexp(values, dim=-1)
    log_probs = values - log_z
    probabilities = torch.exp(log_probs)
    option_logits = {
        label: float(values[token_id].item()) for label, token_id in LABEL_TOKEN_IDS.items()
    }
    option_probs = {
        label: float(probabilities[token_id].item()) for label, token_id in LABEL_TOKEN_IDS.items()
    }
    option_mass = sum(option_probs.values())
    conditional = {
        label: value / option_mass if option_mass else 0.0 for label, value in option_probs.items()
    }
    top_id = int(torch.argmax(values).item())
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    return {
        "logits": option_logits,
        "full_vocab_probabilities": option_probs,
        "conditional_probabilities": conditional,
        "initial_logit": option_logits[initial],
        "target_logit": option_logits[target],
        "target_minus_initial": option_logits[target] - option_logits[initial],
        "option_argmax": max(LABELS, key=lambda label: option_logits[label]),
        "option_mass": option_mass,
        "full_vocab_entropy": float((-(probabilities * log_probs).sum()).item()),
        "full_vocab_top_token_id": top_id,
        "full_vocab_top_token_text": tokenizer.decode([top_id], skip_special_tokens=False),
        "full_vocab_top_token_logit": float(values[top_id].item()),
        "full_vocab_top_probability": float(probabilities[top_id].item()),
        "top_token_is_option": top_id in set(LABEL_TOKEN_IDS.values()),
    }


def _aggregate_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = sum(int(item["total_positions"]) for item in items)
    triggered = sum(int(item["triggered_positions"]) for item in items)
    correction = sum(float(item["projection_correction_sum"]) for item in items)
    return {
        "total_positions": total,
        "triggered_positions": triggered,
        "trigger_rate": triggered / total if total else 0.0,
        "correction_sum": correction,
        "mean_correction_per_position": correction / total if total else 0.0,
    }


def telemetry_summary(
    telemetry: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
    requested_condition: str,
) -> dict[str, Any] | None:
    if telemetry is None:
        return None
    phase_items: dict[str, dict[str, dict[str, Any]]] = {}
    if requested_condition == "p25":
        if actual is None:
            raise ValueError("C3 p25 telemetry lacks actual phase stats")
        for phase, layers in actual.items():
            phase_items[phase] = {
                str(layer): {
                    "total_positions": int(layers[str(layer)]["total_positions"]),
                    "triggered_positions": int(layers[str(layer)]["triggered_positions"]),
                    "projection_correction_sum": float(
                        layers[str(layer)]["projection_correction_sum"]
                    ),
                }
                for layer in LAYERS
            }
        mode = "actual_p25"
    else:
        for phase, payload in telemetry["phases"].items():
            exposure = payload["reference_cap_exposure"]["p25"]
            phase_items[phase] = {
                str(layer): {
                    "total_positions": int(exposure[str(layer)]["count"]),
                    "triggered_positions": int(exposure[str(layer)]["triggered"]),
                    "projection_correction_sum": float(exposure[str(layer)]["correction_sum"]),
                }
                for layer in LAYERS
            }
        mode = "uncapped_counterfactual_p25"
    by_layer = {
        str(layer): _aggregate_items([phase_items[phase][str(layer)] for phase in phase_items])
        for layer in LAYERS
    }
    prefill_phases = ("conversation_prefill", "assistant_prefix_context")
    prefill = _aggregate_items(
        [phase_items[phase][str(layer)] for phase in prefill_phases for layer in LAYERS]
    )
    answer_step = _aggregate_items([phase_items["answer_step"][str(layer)] for layer in LAYERS])
    return {
        "mode": mode,
        "trigger_rate_by_layer": {layer: item["trigger_rate"] for layer, item in by_layer.items()},
        "correction_sum_by_layer": {
            layer: item["correction_sum"] for layer, item in by_layer.items()
        },
        "mean_correction_by_layer": {
            layer: item["mean_correction_per_position"] for layer, item in by_layer.items()
        },
        "prefill_trigger": prefill,
        "answer_step_trigger": answer_step,
    }


def identity_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "direction_id",
        "base_id",
        "source_id",
        "normalized_stem_id",
        "exact_mcq_id",
        "domain",
        "difficulty",
        "direction",
        "arm",
        "template_id",
        "gold_answer",
        "foil_answer",
        "initial_answer",
        "target_answer",
        "initial_position",
        "target_position",
        "other_wrong_answers",
    )
    return {field: row[field] for field in fields}


def run_direct_one(
    model: nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    spec: RunSpec,
    axis: Tensor,
    caps: Mapping[int, LayerCap],
    seed: int,
    record_seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt, prompt_ids, conversation_tokens = render_direct(tokenizer, row)
    device = model_input_device(model)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prefix_tokens = len(prompt_ids) - conversation_tokens
    masks = {
        "conversation_prefill": torch.zeros_like(attention_mask, dtype=torch.bool),
        "assistant_prefix_context": torch.zeros_like(attention_mask, dtype=torch.bool),
        "answer_step": torch.zeros_like(attention_mask, dtype=torch.bool),
    }
    masks["conversation_prefill"][:, :conversation_tokens] = True
    masks["assistant_prefix_context"][:, conversation_tokens:-1] = True
    masks["answer_step"][:, -1] = True
    hooks_before = hook_counts(model)
    monitor = SegmentedAxisMonitor(model, axis, caps, masks) if spec.monitor_enabled else None
    capper = (
        ActivationCapper(model, caps, positions="all", track_stats=True, defer_stats_sync=True)
        if spec.condition == "p25"
        else None
    )
    transformers.set_seed(record_seed)
    with monitor if monitor is not None else nullcontext():
        with capper if capper is not None else nullcontext():
            with torch.inference_mode():
                output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hooks_after = hook_counts(model)
    if hooks_after != hooks_before:
        raise RuntimeError(f"C3 forward-hook leak: {hooks_before} -> {hooks_after}")
    diagnostics = choice_diagnostics(output.logits[0, -1], row, tokenizer)
    del output
    telemetry = monitor.materialize() if monitor is not None else None
    cap_stats = cap_stats_to_dict(capper.stats) if capper is not None else None
    actual = actual_cap_by_phase(telemetry, cap_stats)
    summary = telemetry_summary(telemetry, actual, spec.requested_condition)
    return {
        **identity_fields(row),
        "track": "direct",
        "scope": spec.scope,
        "phase": spec.phase_name,
        "condition": spec.condition,
        "instrumentation": spec.requested_condition,
        "seed": seed,
        "record_seed": record_seed,
        "prompt": prompt,
        "prompt_text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_object_sha256": object_sha256(prompt),
        "prompt_token_sha256": object_sha256(prompt_ids),
        "prompt_tokens": len(prompt_ids),
        "conversation_prefill_tokens": conversation_tokens,
        "assistant_prefix_context_tokens": prefix_tokens - 1,
        "answer_step_tokens": 1,
        "assistant_prefix": ASSISTANT_PREFIX,
        "candidate_token_ids": LABEL_TOKEN_IDS,
        "answer_logits": diagnostics,
        "logit_A": diagnostics["logits"]["A"],
        "logit_B": diagnostics["logits"]["B"],
        "logit_C": diagnostics["logits"]["C"],
        "logit_D": diagnostics["logits"]["D"],
        "target_logit": diagnostics["target_logit"],
        "initial_logit": diagnostics["initial_logit"],
        "target_initial_margin": diagnostics["target_minus_initial"],
        "option_argmax": diagnostics["option_argmax"],
        "full_vocab_top_token": diagnostics["full_vocab_top_token_text"],
        "full_vocab_top_probability": diagnostics["full_vocab_top_probability"],
        "option_probability_mass": diagnostics["option_mass"],
        "conditional_option_probabilities": diagnostics["conditional_probabilities"],
        "entropy": diagnostics["full_vocab_entropy"],
        "top_token_is_option": diagnostics["top_token_is_option"],
        "axis_telemetry": telemetry,
        "cap_stats": cap_stats,
        "actual_cap_stats_by_phase": actual,
        "telemetry_summary": summary,
        "trigger_rate_by_layer": summary["trigger_rate_by_layer"] if summary else None,
        "correction_sum_by_layer": summary["correction_sum_by_layer"] if summary else None,
        "mean_correction_by_layer": summary["mean_correction_by_layer"] if summary else None,
        "prefill_trigger": summary["prefill_trigger"] if summary else None,
        "answer_step_trigger": summary["answer_step_trigger"] if summary else None,
        "hook_counts_before": hooks_before,
        "hook_counts_after": hooks_after,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at": datetime.now().astimezone().isoformat(),
            "input_device": str(device),
        },
    }


def record_key(index: int, row: Mapping[str, Any], spec: RunSpec, seed: int) -> dict[str, Any]:
    return {"index": index, "id": row["id"], "phase": spec.phase_name, "seed": seed}


def materialize(
    store: RecordStore,
    rows: Sequence[dict[str, Any]],
    *,
    spec: RunSpec,
    seed: int,
    results_path: Path,
    runtime_path: Path,
) -> list[dict[str, Any]]:
    results = []
    payloads = []
    for index, row in enumerate(rows):
        payload = store.read(record_key(index, row, spec, seed))
        if payload is None:
            continue
        result = payload.get("result")
        if (
            payload.get("index") != index
            or not isinstance(result, dict)
            or result.get("id") != row["id"]
        ):
            raise ValueError(f"C3 checkpoint mismatch at row {index}")
        results.append(result)
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    atomic_write_jsonl(runtime_path, payloads)
    return results


def main() -> None:
    args = parse_args()
    if args.seed != 0:
        raise ValueError("C3 freezes deterministic seed at 0")
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"Run C3 inside the cedar environment, found {sys.prefix}")
    spec = resolve_spec(args.scope, args.condition)
    all_rows, review_gate, preflight = validate_inputs()
    rows = select_rows(all_rows, spec)
    validate_run_order(spec, args.output_root)
    git_state = git_provenance()
    axis, cap_sets, cap_metadata = load_cap_assets(args.cache_dir)
    caps = cap_sets["p25"]
    model_spec = QwenModelSpec()
    if model_spec.revision != MODEL_REVISION:
        raise ValueError("Pinned Qwen model spec revision changed")
    run_dir = args.output_root / spec.phase_name
    configuration = {
        "kind": "C3 SycoBench bidirectional exploratory selectivity diagnostic",
        "scope": spec.scope,
        "track": "direct",
        "phase": spec.phase_name,
        "condition": spec.condition,
        "instrumentation": spec.requested_condition,
        "monitor_enabled": spec.monitor_enabled,
        "selection": spec.selection,
        "seed": args.seed,
        "dataset": {
            "arms_path": str(ARMS_PATH.relative_to(ROOT)),
            "arms_canonical_text_sha256": canonical_text_sha256(ARMS_PATH),
            "arms_raw_sha256_diagnostic": file_sha256(ARMS_PATH),
            "arms_semantic_sha256": object_sha256(all_rows),
            "selected_rows": len(rows),
            "selected_ids_sha256": object_sha256([row["id"] for row in rows]),
            "manifest_canonical_text_sha256": canonical_text_sha256(MANIFEST_PATH),
            "construction_audit_canonical_text_sha256": canonical_text_sha256(
                CONSTRUCTION_AUDIT_PATH
            ),
            "review_gate_canonical_text_sha256": canonical_text_sha256(REVIEW_GATE_PATH),
            "execution_preflight_canonical_text_sha256": canonical_text_sha256(PREFLIGHT_PATH),
        },
        "model": asdict(model_spec),
        "measurement": {
            "enable_thinking": False,
            "free_generation": False,
            "assistant_prefix": ASSISTANT_PREFIX,
            "candidate_labels": list(LABELS),
            "candidate_token_ids": LABEL_TOKEN_IDS,
            "single_next_token_forward": True,
            "batch_size": 1,
            "content_retry": False,
            "answer_repair": False,
        },
        "axis_telemetry": {
            "layers": list(LAYERS),
            "hook_order": "monitor before same-layer cap",
            "phases": ["conversation_prefill", "assistant_prefix_context", "answer_step"],
            "reference_caps": ["p25"] if spec.monitor_enabled else [],
        },
        "capping": cap_metadata,
        "review_gate": {
            "status": review_gate["status"],
            "review_mode": review_gate["review_mode"],
            "protocol_deviation": True,
            "human_review_performed": False,
            "data_lead_review_performed": False,
            "user_review_waived": True,
            "direction_cases_reviewed": review_gate["direction_cases_reviewed"],
            "template_pairs_reviewed": review_gate["template_pairs_reviewed"],
            "dual_independent_review": False,
            "raw_agreement": None,
            "cohen_kappa": None,
            "limitations": review_gate["limitations"],
            "gpu_authorized": True,
        },
        "preflight_status": preflight["status"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": torch.cuda.device_count(),
        },
        "git": git_state,
    }
    manifest = initialize_run(run_dir, configuration)
    store = RecordStore(run_dir / "records", run_fingerprint=manifest["fingerprint"])
    results_path = run_dir / "results.jsonl"
    runtime_path = run_dir / "record_runtime.jsonl"
    results = materialize(
        store,
        rows,
        spec=spec,
        seed=args.seed,
        results_path=results_path,
        runtime_path=runtime_path,
    )
    if args.stage == "prepare":
        print(json.dumps({"phase": spec.phase_name, "rows": len(rows), "existing": len(results)}))
        return
    missing = [
        index
        for index, row in enumerate(rows)
        if not store.contains(record_key(index, row, spec, args.seed))
    ]
    if missing:
        if torch.cuda.device_count() < 4:
            raise RuntimeError("C3 Qwen3-32B plus hooks requires four visible GPUs")
        max_memory = {
            index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
        }
        update_runtime(run_dir, {"execution_started_at": datetime.now().astimezone().isoformat()})
        model, tokenizer = load_qwen_model(model_spec, max_memory=max_memory, local_files_only=True)
        validate_label_tokens(tokenizer)
        for index in missing:
            row = rows[index]
            record_seed = args.seed * 1_000_003 + index
            result = run_direct_one(
                model,
                tokenizer,
                row,
                spec=spec,
                axis=axis,
                caps=caps,
                seed=args.seed,
                record_seed=record_seed,
            )
            store.write(record_key(index, row, spec, args.seed), {"index": index, "result": result})
            results = materialize(
                store,
                rows,
                spec=spec,
                seed=args.seed,
                results_path=results_path,
                runtime_path=runtime_path,
            )
            print(
                json.dumps(
                    {"phase": spec.phase_name, "complete": len(results), "expected": len(rows)}
                ),
                flush=True,
            )
        del model, tokenizer
        release_cuda_cache()
    results = materialize(
        store,
        rows,
        spec=spec,
        seed=args.seed,
        results_path=results_path,
        runtime_path=runtime_path,
    )
    if len(results) == len(rows):
        update_runtime(
            run_dir,
            {
                "execution_completed_at": datetime.now().astimezone().isoformat(),
                "complete_results": len(results),
                "results_canonical_text_sha256": canonical_text_sha256(results_path),
                "record_runtime_canonical_text_sha256": canonical_text_sha256(runtime_path),
                "results_raw_sha256_diagnostic": file_sha256(results_path),
                "record_runtime_raw_sha256_diagnostic": file_sha256(runtime_path),
                "content_retries": 0,
                "cap_trigger_stats": aggregate_run_cap_stats(results),
            },
        )
    print(json.dumps({"phase": spec.phase_name, "complete": len(results), "expected": len(rows)}))


if __name__ == "__main__":
    main()
