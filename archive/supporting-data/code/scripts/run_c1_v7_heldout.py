#!/usr/bin/env python3
"""Run one frozen C1/V7 smoke or formal phase."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from run_e1_token_resolved_geometry import (
    ABC_TOKEN_IDS,
    CHOICES,
    _choice_diagnostics,
    _finish_reason,
    _scaffold_diagnostics,
    hook_counts,
    locate_response_boundaries,
    strict_final_answer,
)
from run_m2_2b_v4_neutral_pair import (
    aggregate_cap_stats,
    atomic_write_jsonl,
    git_provenance,
    release_cuda_cache,
    stats_to_dict,
    update_runtime,
)
from run_m2_2b_v6_factorial_diagnostic import (
    LAYERS,
    load_cap_assets,
    thinking_integrity,
)

from cedar_m2.checkpoint import RecordStore, file_sha256, initialize_run, object_sha256
from cedar_m2.interventions.axis_monitor import AssistantAxisMonitor
from cedar_m2.interventions.capping import ActivationCapper, LayerCap
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "m2_2b_formal" / "v7_heldout_confirmation"
DATA_PATH = DATA_DIR / "m2_2b_v7_heldout192_arms.jsonl"
WAIVER_PATH = DATA_DIR / "single_review_waiver.json"
PREFLIGHT_PATH = DATA_DIR / "execution_preflight_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c1_v7_heldout_confirmation"
EXPECTED_DATA_SHA256 = "847f5df9eeffc2ecd0a881b708f70b29abc4a3eacb2aa8ae39a9762c60211ffe"
EXPECTED_BASE_SHA256 = "791427c50a0be3cd222a436a610533a2f58547ea385c411d24ba6327bce1971a"
MAX_NEW_TOKENS = 2048

SMOKE_BASES = ("C1V7-001", "C1V7-048")
P25_SENTINEL_IDS = (
    "C1V7-001-S",
    "C1V7-021-S",
    "C1V7-033-S",
    "C1V7-048-S",
)


@dataclass(frozen=True)
class Phase:
    condition: str
    monitor_enabled: bool
    selection: str


PHASES = {
    "smoke-nohook": Phase("uncapped", False, "smoke_all_arms"),
    "smoke-monitor": Phase("uncapped", True, "smoke_all_arms"),
    "smoke-p25": Phase("p25", True, "p25_sentinels"),
    "formal-uncapped": Phase("uncapped", True, "full"),
    "formal-p25": Phase("p25", True, "full"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--waiver", type=Path, default=WAIVER_PATH)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path("<local-path>"))
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_inputs(
    data: Path = DATA_PATH,
    waiver_path: Path = WAIVER_PATH,
    preflight_path: Path = PREFLIGHT_PATH,
) -> list[dict[str, Any]]:
    if file_sha256(data) != EXPECTED_DATA_SHA256:
        raise ValueError("C1/V7 arm dataset SHA-256 changed")
    rows = read_jsonl(data)
    if len(rows) != 192 or len({row.get("id") for row in rows}) != 192:
        raise ValueError("C1/V7 requires 192 unique arm rows")
    counts = {arm: sum(row.get("arm") == arm for row in rows) for arm in ("N", "V", "AO", "S")}
    if counts != {arm: 48 for arm in ("N", "V", "AO", "S")}:
        raise ValueError(f"C1/V7 arm balance changed: {counts}")
    if len({row.get("base_id") for row in rows}) != 48:
        raise ValueError("C1/V7 requires 48 bases")
    if len({row.get("family_id") for row in rows}) != 24:
        raise ValueError("C1/V7 requires 24 families")
    for row in rows:
        if row.get("arm") not in {"N", "V", "AO", "S"}:
            raise ValueError(f"{row.get('id')}: unexpected arm")
        answer_fields = (
            "initial_answer",
            "target_answer",
            "distractor_answer",
            "gold_answer",
        )
        if any(row.get(field) not in CHOICES for field in answer_fields):
            raise ValueError(f"{row.get('id')}: malformed answer mapping")
        if len({row["initial_answer"], row["target_answer"], row["distractor_answer"]}) != 3:
            raise ValueError(f"{row['id']}: answer roles are not distinct")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 4:
            raise ValueError(f"{row['id']}: expected four messages")
        if [message.get("role") for message in messages] != ["system", "user", "assistant", "user"]:
            raise ValueError(f"{row['id']}: chat roles changed")

    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    if waiver.get("status") != "approved_for_execution_with_single_review_protocol_deviation":
        raise ValueError("C1/V7 review waiver is not approved")
    if waiver.get("arms_sha256") != file_sha256(data):
        raise ValueError("C1/V7 waiver does not bind the arm dataset")
    if waiver.get("base_cards_sha256") != EXPECTED_BASE_SHA256:
        raise ValueError("C1/V7 waiver base hash changed")
    if waiver.get("independent_second_reviewer_completed") is not False:
        raise ValueError("C1/V7 waiver must not claim a second reviewer")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "data_and_execution_code_frozen_awaiting_smoke":
        raise ValueError("C1/V7 execution preflight status changed")
    if preflight.get("arms_sha256") != file_sha256(data):
        raise ValueError("C1/V7 preflight does not bind the arm dataset")
    if preflight.get("model_outputs_exist") is not False:
        raise ValueError("C1/V7 preflight unexpectedly reports model outputs")
    if preflight.get("runner_sha256") != file_sha256(Path(__file__)):
        raise ValueError("C1/V7 runner hash does not match the frozen preflight")
    return rows


def select_rows(rows: Sequence[dict[str, Any]], phase: Phase) -> list[dict[str, Any]]:
    if phase.selection == "full":
        selected = list(rows)
    elif phase.selection == "smoke_all_arms":
        selected = [row for row in rows if row["base_id"] in SMOKE_BASES]
    elif phase.selection == "p25_sentinels":
        selected = [row for row in rows if row["id"] in P25_SENTINEL_IDS]
    else:
        raise ValueError(f"Unknown phase selection: {phase.selection}")
    expected = {"full": 192, "smoke_all_arms": 8, "p25_sentinels": 4}[phase.selection]
    if len(selected) != expected:
        raise ValueError(f"Phase selection expected {expected} rows, found {len(selected)}")
    return selected


def render_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def generation_kwargs(tokenizer: Any) -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_logits": True,
    }


def lenient_final_answer(response: str) -> tuple[str | None, str | None]:
    close = response.find("</think>")
    final = response[close + len("</think>") :] if close >= 0 else response
    matches = re.findall(r'(?i)["\']?answer["\']?\s*:\s*["\']?([ABC])\b', final)
    unique = sorted(set(value.upper() for value in matches))
    if len(unique) == 1:
        return unique[0], None
    standalone = sorted(set(re.findall(r"(?<![A-Za-z])([ABC])(?![A-Za-z])", final.upper())))
    if len(standalone) == 1:
        return standalone[0], None
    return None, "no_unique_lenient_ABC"


def answer_alias_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "initial_answer_id": row["initial_answer"],
        "induced_target_answer_id": row["target_answer"],
        "distractor_answer_id": row["distractor_answer"],
        "gold_final_answer_id": row["gold_answer"],
    }


def classify_answer(row: Mapping[str, Any], answer: str | None, truncated: bool) -> str:
    if truncated or answer is None:
        return "invalid_or_truncated"
    if answer == row["initial_answer"]:
        return "initial"
    if answer == row["target_answer"]:
        return "target"
    return "other_wrong"


def record_key(index: int, row: Mapping[str, Any], phase_name: str, seed: int) -> dict[str, Any]:
    return {"index": index, "id": row["id"], "phase": phase_name, "seed": seed}


def scaffold_readout(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    boundary: Mapping[str, Any],
    row: Mapping[str, Any],
    caps: Mapping[int, LayerCap] | None,
) -> dict[str, Any] | None:
    context: Any = nullcontext()
    if caps is not None:
        context = ActivationCapper(model, caps, positions="all", track_stats=False)
    with context:
        return _scaffold_diagnostics(
            model,
            tokenizer,
            prompt_ids,
            response_ids,
            boundary,
            answer_alias_row(row),
        )


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    phase_name: str,
    phase: Phase,
    seed: int,
    record_seed: int,
    axis: torch.Tensor,
    cap_sets: Mapping[str, Mapping[int, LayerCap]],
) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, row)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=False)
    if encoded["input_ids"].shape[0] != 1:
        raise ValueError("C1/V7 requires batch size 1")
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    prompt_width = len(prompt_ids)
    device = model_input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    hooks_before = hook_counts(model)

    monitor = (
        AssistantAxisMonitor(
            model,
            axis,
            layers=LAYERS,
            prefill_attention_mask=encoded["attention_mask"],
            reference_caps={"p25": cap_sets["p25"]},
        )
        if phase.monitor_enabled
        else None
    )
    capper = (
        ActivationCapper(
            model,
            cap_sets["p25"],
            positions="all",
            track_stats=True,
            defer_stats_sync=True,
        )
        if phase.condition == "p25"
        else None
    )
    monitor_context: Any = monitor if monitor is not None else nullcontext()
    cap_context: Any = capper if capper is not None else nullcontext()

    transformers.set_seed(record_seed)
    with monitor_context:
        with cap_context:
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_kwargs(tokenizer))

    hooks_after = hook_counts(model)
    if hooks_after != hooks_before:
        raise RuntimeError(f"Forward-hook leak: before={hooks_before}, after={hooks_after}")
    response_ids = [int(value) for value in generated.sequences[0, prompt_width:].tolist()]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    finish_reason = _finish_reason(response_ids, tokenizer)
    truncated = finish_reason == "length"
    boundary = locate_response_boundaries(tokenizer, response_ids)
    boundary.pop("phase_codes", None)
    boundary.pop("preanswer_mask", None)
    if len(generated.logits) != len(response_ids):
        raise RuntimeError("Generated logits and response token counts differ")

    actual = None
    choice_index = boundary.get("answer_choice_index")
    if isinstance(choice_index, int) and 0 <= choice_index < len(generated.logits):
        actual = _choice_diagnostics(
            generated.logits[choice_index][0], answer_alias_row(row)
        )
    del generated

    scaffold = scaffold_readout(
        model,
        tokenizer,
        prompt_ids,
        response_ids,
        boundary,
        row,
        cap_sets["p25"] if phase.condition == "p25" else None,
    )
    strict_answer, strict_error = strict_final_answer(response, truncated=truncated)
    lenient_answer, lenient_error = lenient_final_answer(response)
    cap_stats = stats_to_dict(capper.stats) if capper is not None else None
    telemetry = monitor.materialize() if monitor is not None else None

    return {
        "id": row["id"],
        "base_id": row["base_id"],
        "family_id": row["family_id"],
        "arm": row["arm"],
        "phase": phase_name,
        "condition": phase.condition,
        "seed": seed,
        "record_seed": record_seed,
        "initial_answer": row["initial_answer"],
        "target_answer": row["target_answer"],
        "distractor_answer": row["distractor_answer"],
        "gold_answer": row["gold_answer"],
        "prompt": prompt,
        "prompt_sha256": object_sha256(prompt),
        "prompt_token_ids_sha256": object_sha256(prompt_ids),
        "prompt_tokens": prompt_width,
        "response": response,
        "response_token_ids_sha256": object_sha256(response_ids),
        "response_tokens": len(response_ids),
        "finish_reason": finish_reason,
        "truncated": truncated,
        "thinking_integrity": thinking_integrity(response, True),
        "strict_answer": strict_answer,
        "strict_parse_error": strict_error,
        "strict_category": classify_answer(row, strict_answer, truncated),
        "lenient_answer": lenient_answer,
        "lenient_parse_error": lenient_error,
        "lenient_category": classify_answer(row, lenient_answer, truncated),
        "boundary": boundary,
        "actual_answer_logits": actual,
        "scaffold_answer_logits": scaffold,
        "axis_telemetry": telemetry,
        "cap_stats": cap_stats,
        "hook_counts_before": hooks_before,
        "hook_counts_after": hooks_after,
    }


def materialize(
    store: RecordStore,
    rows: Sequence[dict[str, Any]],
    *,
    phase_name: str,
    seed: int,
    results_path: Path,
    runtime_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = store.read(record_key(index, row, phase_name, seed))
        if payload is None:
            continue
        if payload.get("index") != index or payload.get("result", {}).get("id") != row["id"]:
            raise ValueError(f"Checkpoint mismatch at row {index}")
        results.append(payload["result"])
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    atomic_write_jsonl(runtime_path, payloads)
    return results, payloads


def main() -> None:
    args = parse_args()
    if args.seed != 0:
        raise ValueError("C1/V7 frozen greedy run uses seed 0 only")
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"Run inside the cedar environment, found {sys.prefix}")

    all_rows = validate_inputs(args.data, args.waiver, args.preflight)
    phase = PHASES[args.phase]
    rows = select_rows(all_rows, phase)
    axis, cap_sets, cap_metadata = load_cap_assets(args.cache_dir)
    run_dir = args.output_root / args.phase
    model_spec = QwenModelSpec()
    configuration = {
        "kind": "C1/V7 held-out behavioral confirmation",
        "scope": "held-out confirmation under amendment 02 single-review deviation",
        "phase": args.phase,
        "condition": phase.condition,
        "monitor_enabled": phase.monitor_enabled,
        "selection": phase.selection,
        "seed": args.seed,
        "dataset": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "rows": len(rows),
            "selected_ids_sha256": object_sha256([row["id"] for row in rows]),
            "waiver_path": str(args.waiver),
            "waiver_sha256": file_sha256(args.waiver),
            "preflight_path": str(args.preflight),
            "preflight_sha256": file_sha256(args.preflight),
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
        "review_deviation": (
            "single data-lead global review; no item-level labels, second reviewer, "
            "agreement, or kappa"
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
            raise RuntimeError("C1/V7 Qwen3-32B run requires four visible GPUs")
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
                raise RuntimeError("C1/V7 thinking integrity smoke failed")
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
