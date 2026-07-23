#!/usr/bin/env python3
"""Run the frozen Step 5 uncapped development calibration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cedar_m2.checkpoint import (
    RecordStore,
    atomic_write_json,
    file_sha256,
    initialize_run,
    object_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_development_v1"
ARMS_PATH = DATA_ROOT / "development_arms288.jsonl"
DATA_MANIFEST_PATH = DATA_ROOT / "manifest.json"
ZERO_GPU_GATE_PATH = DATA_ROOT / "ZERO_GPU_GATE.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_step5_uncapped"

DATA_FREEZE_COMMIT = "8a11bb8693d061113479100a28bcad0e64b2f18f"
ARMS_CANONICAL_TEXT_SHA256 = "39caee8f42cd94f2869e8cafe707d4cfb049b3b210ca63a33648c6aa96882559"
ARMS_OBJECT_SHA256 = "7afd684250d044187a218a1d9daf4d0a276e3e1518efe8e44bc66932236cd6d0"
ARMS_IDS_SHA256 = "a0ee7c876126bdbc1b5864bf0a60dd1b04e99f33377c95c071d8a96626cdad05"
DATA_MANIFEST_CANONICAL_TEXT_SHA256 = (
    "da03474d040bbd35081d571ac52d8248e6200de52ddbf70e9ed724194fa327fa"
)
ZERO_GPU_GATE_CANONICAL_TEXT_SHA256 = (
    "befd379d724a9984ff3b5c2d49ab9765ef1d19cab45ba4ebc201f1eee08a8c78"
)

MODEL_REPO = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
MAX_NEW_TOKENS = 2048
SEED = 0
LABELS = ("A", "B", "C", "D")
LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
ARMS = ("N", "R", "V", "F", "AO", "S")
ASSISTANT_PREFIX = '{"answer":"'
RUNNER_AMENDMENT = "step5-amendment-01-robust-answer-logit-alignment"
ANSWER_PREFIX_PATTERN = re.compile(r'\{\s*"answer"\s*:\s*"\Z')
CAPTURE_TAIL_TOKENS = 64
SMOKE_BASE_IDS = (
    "CEDAR-DEV-001",
    "CEDAR-DEV-009",
    "CEDAR-DEV-018",
    "CEDAR-DEV-026",
)
EXPECTED_ROWS = {"smoke": 24, "formal": 288}
EXPECTED_RENDERED_PROMPTS_SHA256 = {
    ("smoke", "thinking"): ("0cb7c2ed652a6c8c2ece8fbc772868bfbeb7594bc38574ed5243b42889ff3230"),
    ("smoke", "nonthinking"): ("9a9848ccf142a2c7deebb733caf3f2f6c701d4ec3e8a897266749eb6a83e1ddd"),
    ("formal", "thinking"): ("8c1cf173bff11761855ee0bc0d3467e3070457ebeb1c27a58ee244a4a5fd50d0"),
    ("formal", "nonthinking"): ("96e1a7e7ed429d4ceec520740397f2502bfe9215fd1f60f4c5251ee2088cf7cd"),
}
EXPECTED_TOKENIZED_INPUT_IDS_SHA256 = {
    ("smoke", "thinking"): ("96bfd8c22e236191a5d4a0870bb6e4d6fab595da328fa03ae5e082c7995a9ddf"),
    ("smoke", "nonthinking"): ("4364985aa890ae1b3f5f5bd60d45d1a87f051942349dbdc4eb79a1941a51dbd4"),
    ("formal", "thinking"): ("000bbfbd0483df47853416e6d51be70ebb77797b37475eb93d2cd87654d454b0"),
    ("formal", "nonthinking"): ("85864e13ed2635ad839d8e3020554c710704d633090a4ec20ab3767767973364"),
}


@dataclass(frozen=True)
class Mode:
    enable_thinking: bool
    do_sample: bool = False
    num_beams: int = 1


MODES = {
    "thinking": Mode(enable_thinking=True),
    "nonthinking": Mode(enable_thinking=False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME")),
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def update_runtime(run_dir: Path, values: Mapping[str, Any]) -> None:
    path = run_dir / "runtime.json"
    runtime = read_json(path) if path.is_file() else {}
    runtime.update(values)
    atomic_write_json(path, runtime)


def git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()
    if status:
        raise RuntimeError("Step 5 refuses to run from a dirty worktree")
    upstream = subprocess.run(
        ["git", "rev-parse", "@{u}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != upstream:
        raise RuntimeError("Step 5 requires local HEAD to equal its configured upstream")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", DATA_FREEZE_COMMIT, commit],
        cwd=ROOT,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Step 5 HEAD does not descend from the frozen Step 0-4 commit")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "branch": branch,
        "upstream_commit": upstream,
        "head_matches_upstream": True,
        "data_freeze_commit": DATA_FREEZE_COMMIT,
        "data_freeze_is_ancestor": True,
        "dirty": False,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def validate_inputs() -> list[dict[str, Any]]:
    expected_files = {
        ARMS_PATH: ARMS_CANONICAL_TEXT_SHA256,
        DATA_MANIFEST_PATH: DATA_MANIFEST_CANONICAL_TEXT_SHA256,
        ZERO_GPU_GATE_PATH: ZERO_GPU_GATE_CANONICAL_TEXT_SHA256,
    }
    for path, expected in expected_files.items():
        if not path.is_file() or canonical_text_sha256(path) != expected:
            raise ValueError(f"Frozen Step 5 input hash changed: {path}")

    gate = read_json(ZERO_GPU_GATE_PATH)
    if (
        gate.get("gate") != "A"
        or gate.get("may_start_uncapped_development_calibration") is not True
        or gate.get("gpu_required_next") is not True
        or gate.get("gpu_qwen_capping_run") is not False
        or not all(gate.get("automatic_prerequisites", {}).values())
    ):
        raise ValueError("ZERO_GPU_GATE.json no longer authorizes Step 5 uncapped calibration")
    manifest = read_json(DATA_MANIFEST_PATH)
    if (
        manifest.get("status") != "zero_gpu_gate_a"
        or manifest.get("zero_gpu_gate") != "A"
        or manifest.get("tokenizer_gate") != "passed"
        or manifest.get("gpu_qwen_capping_run") is not False
    ):
        raise ValueError("Development release manifest is not the frozen Gate A release")

    rows = read_jsonl(ARMS_PATH)
    if object_sha256(rows) != ARMS_OBJECT_SHA256:
        raise ValueError("Development arm object hash changed")
    if object_sha256([row.get("id") for row in rows]) != ARMS_IDS_SHA256:
        raise ValueError("Development arm ID order changed")
    if len(rows) != 288 or len({row.get("id") for row in rows}) != 288:
        raise ValueError("Step 5 requires exactly 288 unique arms")
    if Counter(row.get("arm") for row in rows) != Counter({arm: 48 for arm in ARMS}):
        raise ValueError("Step 5 arm balance changed")

    by_base: dict[str, list[str]] = {}
    for row in rows:
        item_id = row.get("id")
        base_id = row.get("base_id")
        if not isinstance(item_id, str) or not isinstance(base_id, str):
            raise ValueError("Every Step 5 row requires string id/base_id")
        by_base.setdefault(base_id, []).append(str(row.get("arm")))
        initial = row.get("initial_answer")
        target = row.get("target_answer")
        gold = row.get("gold_answer")
        if initial not in LABELS or target not in LABELS or initial == target:
            raise ValueError(f"{item_id}: invalid initial/target mapping")
        if gold != (target if row.get("arm") == "V" else initial):
            raise ValueError(f"{item_id}: gold answer changed")
        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "system",
            "user",
            "assistant",
            "user",
        ]:
            raise ValueError(f"{item_id}: four-message conversation changed")
        try:
            initial_payload = json.loads(messages[2]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{item_id}: malformed frozen initial response") from error
        if initial_payload != {"answer": initial}:
            raise ValueError(f"{item_id}: frozen initial response changed")
    if len(by_base) != 48 or any(arms != list(ARMS) for arms in by_base.values()):
        raise ValueError("Step 5 requires 48 ordered six-arm base units")
    return rows


def select_rows(rows: Sequence[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    selected = (
        list(rows)
        if scope == "formal"
        else [row for row in rows if row["base_id"] in set(SMOKE_BASE_IDS)]
    )
    expected = EXPECTED_ROWS[scope]
    if len(selected) != expected or len({row["id"] for row in selected}) != expected:
        raise ValueError(f"{scope} selection expected {expected} unique rows")
    counts = Counter(row["arm"] for row in selected)
    per_arm = expected // len(ARMS)
    if counts != Counter({arm: per_arm for arm in ARMS}):
        raise ValueError(f"{scope} selection lost complete arm balance")
    return selected


def run_name(scope: str, mode: str) -> str:
    return f"step5_{scope}_{mode}_uncapped_seed0"


def completed_phase(output_root: Path, scope: str, mode: str) -> bool:
    run_dir = output_root / run_name(scope, mode)
    runtime_path = run_dir / "runtime.json"
    results_path = run_dir / "results.jsonl"
    if not runtime_path.is_file() or not results_path.is_file():
        return False
    runtime = read_json(runtime_path)
    return (
        runtime.get("complete_results") == EXPECTED_ROWS[scope]
        and runtime.get("content_retries") == 0
        and runtime.get("results_sha256") == file_sha256(results_path)
    )


def passed_audit(path: Path) -> bool:
    return path.is_file() and read_json(path).get("status") == "passed"


def validate_run_order(scope: str, mode: str, output_root: Path) -> None:
    if scope == "smoke" and mode == "nonthinking":
        if not completed_phase(output_root, "smoke", "thinking"):
            raise ValueError("nonthinking smoke is blocked until thinking smoke is complete")
    if scope == "formal":
        if not passed_audit(output_root / "SMOKE_AUDIT.json"):
            raise ValueError("formal Step 5 is blocked until the combined smoke audit passes")
        if mode == "nonthinking" and not completed_phase(output_root, "formal", "thinking"):
            raise ValueError("formal nonthinking is blocked until 288 thinking rows complete")


def render_prompt(tokenizer: Any, row: Mapping[str, Any], mode: Mode) -> str:
    return tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=mode.enable_thinking,
    )


def validate_label_tokens(tokenizer: Any) -> None:
    observed = {label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS}
    expected = {label: [token_id] for label, token_id in LABEL_TOKEN_IDS.items()}
    if observed != expected:
        raise RuntimeError(f"Pinned Qwen A/B/C/D token mapping changed: {observed}")
    if not tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False):
        raise RuntimeError("Frozen assistant answer prefix tokenizes to an empty sequence")


def choice_diagnostics(logits: Tensor, row: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    values = logits.detach().float()
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise FloatingPointError("Answer-step logits are malformed or contain NaN/Inf")
    log_z = torch.logsumexp(values, dim=-1)
    log_probs = values - log_z
    probabilities = torch.exp(log_probs)
    entropy = float((-(probabilities * log_probs).sum()).item())
    option_logits = {
        label: float(values[token_id].item()) for label, token_id in LABEL_TOKEN_IDS.items()
    }
    option_probabilities = {
        label: float(probabilities[token_id].item()) for label, token_id in LABEL_TOKEN_IDS.items()
    }
    option_mass = sum(option_probabilities.values())
    conditional = {
        label: value / option_mass if option_mass else 0.0
        for label, value in option_probabilities.items()
    }
    ranks = {
        label: int(torch.count_nonzero(values > values[token_id]).item()) + 1
        for label, token_id in LABEL_TOKEN_IDS.items()
    }
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    top_id = int(torch.argmax(values).item())
    return {
        "logits": option_logits,
        "full_vocab_probabilities": option_probabilities,
        "conditional_probabilities": conditional,
        "ranks": ranks,
        "initial_logit": option_logits[initial],
        "target_logit": option_logits[target],
        "target_minus_initial": option_logits[target] - option_logits[initial],
        "initial_rank": ranks[initial],
        "target_rank": ranks[target],
        "option_argmax": max(LABELS, key=lambda label: option_logits[label]),
        "option_mass": option_mass,
        "full_vocab_entropy": entropy,
        "full_vocab_top_token_id": top_id,
        "full_vocab_top_token_text": tokenizer.decode([top_id], skip_special_tokens=False),
        "top_token_is_option": top_id in set(LABEL_TOKEN_IDS.values()),
    }


class AnswerPrefixLogitCapture:
    """Read-only capture at steps following a valid JSON answer prefix."""

    def __init__(self, tokenizer: Any, row: Mapping[str, Any], *, prompt_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.row = row
        self.prompt_tokens = prompt_tokens
        self.prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
        variants = (
            ASSISTANT_PREFIX,
            '{"answer": "',
            '{ "answer":"',
            '{ "answer": "',
        )
        self.prefix_variants = tuple(
            tuple(tokenizer.encode(value, add_special_tokens=False)) for value in variants
        )
        self.max_prefix_tokens = max(len(value) for value in self.prefix_variants)
        terminal_variants = (*variants, '"')
        self.prefix_terminal_token_ids = {
            int(encoded[-1])
            for value in terminal_variants
            if (encoded := tokenizer.encode(value, add_special_tokens=False))
        }
        self.by_response_index: dict[int, dict[str, Any]] = {}

    def has_answer_prefix(self, input_ids: Tensor) -> bool:
        sequence_length = int(input_ids.shape[1])
        tail = [
            int(value)
            for value in input_ids[0, max(0, sequence_length - self.max_prefix_tokens) :].tolist()
        ]
        if any(
            len(tail) >= len(variant) and tuple(tail[-len(variant) :]) == variant
            for variant in self.prefix_variants
        ):
            return True
        if not tail or tail[-1] not in self.prefix_terminal_token_ids:
            return False
        response_start = max(self.prompt_tokens, sequence_length - CAPTURE_TAIL_TOKENS)
        response_tail = [
            int(value) for value in input_ids[0, response_start:sequence_length].tolist()
        ]
        decoded_tail = self.tokenizer.decode(
            response_tail,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return ANSWER_PREFIX_PATTERN.search(decoded_tail) is not None

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError("Step 5 answer-logit capture requires batch size 1")
        if self.has_answer_prefix(input_ids):
            response_index = int(input_ids.shape[1] - self.prompt_tokens)
            self.by_response_index[response_index] = choice_diagnostics(
                scores[0], self.row, self.tokenizer
            )
        return scores


def finish_reason(response_ids: Sequence[int], tokenizer: Any) -> str:
    if len(response_ids) >= MAX_NEW_TOKENS:
        return "length"
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in eos or []}
    return "eos_token" if response_ids and response_ids[-1] in eos_ids else "other_stop"


def response_sections(response: str) -> tuple[str, str, dict[str, Any]]:
    stripped = response.lstrip()
    opens = list(re.finditer(re.escape("<think>"), stripped))
    closes = list(re.finditer(re.escape("</think>"), stripped))
    ordered = len(opens) == 1 and len(closes) == 1 and opens[0].end() <= closes[0].start()
    if ordered:
        reasoning = stripped[opens[0].end() : closes[0].start()].strip()
        final = stripped[closes[0].end() :].strip()
    else:
        reasoning = ""
        final = stripped.strip()
    return (
        reasoning,
        final,
        {
            "open_tag_count": len(opens),
            "close_tag_count": len(closes),
            "ordered_single_pair": ordered,
            "reasoning_nonempty": bool(reasoning),
            "final_nonempty": bool(final),
        },
    )


def strict_answer(response: str, *, truncated: bool) -> tuple[str | None, str, str, dict[str, Any]]:
    reasoning, final, sections = response_sections(response)
    if truncated:
        return None, "truncated", reasoning, sections
    try:
        payload = json.loads(final)
    except json.JSONDecodeError:
        return None, "invalid_json", reasoning, sections
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        return None, "wrong_json_schema", reasoning, sections
    if payload.get("answer") not in LABELS:
        return None, "answer_not_ABCD", reasoning, sections
    return str(payload["answer"]), "passed", reasoning, sections


def mode_integrity(mode: Mode, sections: Mapping[str, Any]) -> dict[str, Any]:
    if mode.enable_thinking:
        passed = bool(
            sections.get("ordered_single_pair")
            and sections.get("reasoning_nonempty")
            and sections.get("final_nonempty")
        )
    else:
        passed = bool(sections.get("final_nonempty") and not sections.get("reasoning_nonempty"))
    return {"enabled": mode.enable_thinking, **dict(sections), "passed": passed}


def classify(row: Mapping[str, Any], answer: str | None) -> str:
    if answer is None:
        return "invalid_or_abstain"
    if answer == row["initial_answer"]:
        return "initial"
    if answer == row["target_answer"]:
        return "induced_target"
    return "other_wrong"


def select_captured_diagnostics(
    response_ids: Sequence[int],
    captures: Mapping[int, dict[str, Any]],
    answer: str | None,
) -> tuple[dict[str, Any] | None, int | None, list[dict[str, Any]]]:
    by_token = {token_id: label for label, token_id in LABEL_TOKEN_IDS.items()}
    candidates = [
        {"response_index": index, "generated_label": by_token[response_ids[index]]}
        for index in sorted(captures)
        if 0 <= index < len(response_ids) and response_ids[index] in by_token
    ]
    matching = [item for item in candidates if item["generated_label"] == answer]
    selected = matching[-1] if matching else (candidates[-1] if candidates else None)
    if selected is None:
        return None, None, candidates
    index = int(selected["response_index"])
    return captures[index], index, candidates


def align_answer_diagnostics(
    response_ids: Sequence[int],
    captures: Mapping[int, dict[str, Any]],
    answer: str | None,
    boundary: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, int | None, list[dict[str, Any]], dict[str, Any]]:
    """Align a parsed answer to its read-only on-trajectory logit capture."""
    fallback, fallback_index, candidates = select_captured_diagnostics(
        response_ids, captures, answer
    )
    output_boundary = dict(boundary)
    boundary_index = output_boundary.get("answer_choice_index")
    if isinstance(boundary_index, int):
        diagnostics = captures.get(boundary_index)
        capture_index = boundary_index if diagnostics is not None else None
        output_boundary.setdefault("alignment_method", "tokenizer_offset_roundtrip")
    else:
        diagnostics = fallback
        capture_index = fallback_index
        if answer is not None and diagnostics is not None and isinstance(capture_index, int):
            original_errors = list(output_boundary.get("errors", []))
            output_boundary.update(
                {
                    "answer_choice_index": capture_index,
                    "answer_choice": answer,
                    "alignment_method": "last_matching_on_trajectory_prefix_capture",
                    "locator_diagnostic_errors": original_errors,
                    "errors": [],
                }
            )
    return diagnostics, capture_index, candidates, output_boundary


def _encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        offsets = offsets[0]
    return [int(value) for value in ids], [(int(a), int(b)) for a, b in offsets]


def locate_answer_choice(
    tokenizer: Any,
    response_ids: Sequence[int],
    *,
    mode: Mode,
    strict_choice: str | None,
) -> dict[str, Any]:
    text = tokenizer.decode(
        list(response_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    reencoded, offsets = _encode_with_offsets(tokenizer, text)
    report: dict[str, Any] = {
        "tokenization_roundtrip": reencoded == list(response_ids),
        "answer_choice_index": None,
        "answer_choice": None,
        "errors": [],
    }
    if reencoded != list(response_ids):
        report["errors"].append("decoded response does not round-trip to generated token IDs")
        return report
    close = text.rfind("</think>")
    if mode.enable_thinking and close < 0:
        report["errors"].append("thinking response lacks closing think tag")
        return report
    final_start = close + len("</think>") if close >= 0 else 0
    matches = list(re.finditer(r'"answer"\s*:\s*"([ABCD])"', text[final_start:]))
    if len(matches) != 1:
        report["errors"].append("expected exactly one final answer field")
        return report
    match = matches[0]
    choice = match.group(1)
    if strict_choice is not None and choice != strict_choice:
        report["errors"].append("token-level answer differs from strict JSON answer")
        return report
    choice_start = final_start + match.start(1)
    choice_end = final_start + match.end(1)
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if start <= choice_start and end >= choice_end and end > start
    ]
    if len(indices) != 1:
        report["errors"].append("answer character does not map to exactly one token")
        return report
    index = indices[0]
    if int(response_ids[index]) != LABEL_TOKEN_IDS[choice]:
        report["errors"].append("answer token ID differs from frozen A/B/C/D mapping")
        return report
    report["answer_choice_index"] = index
    report["answer_choice"] = choice
    return report


def identity_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "base_id",
        "family_id",
        "structure_class",
        "variant",
        "arm",
        "initial_answer",
        "target_answer",
        "gold_answer",
        "expected_outcome",
        "error_class",
        "error_position",
        "analysis_schema",
    )
    return {field: row.get(field) for field in fields}


def generation_kwargs(tokenizer: Any) -> dict[str, Any]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "output_logits": False,
    }


def generate_one(
    model: nn.Module,
    tokenizer: Any,
    transformers_module: Any,
    row: dict[str, Any],
    *,
    scope: str,
    mode_name: str,
    record_seed: int,
) -> dict[str, Any]:
    mode = MODES[mode_name]
    prompt = render_prompt(tokenizer, row, mode)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    if tokenizer.encode(prompt, add_special_tokens=False) != prompt_ids:
        raise ValueError(f"{row['id']}: prompt tokenization changed under batch-size 1")
    from cedar_m2.models.qwen import model_input_device

    encoded = {name: value.to(model_input_device(model)) for name, value in encoded.items()}
    capture = AnswerPrefixLogitCapture(tokenizer, row, prompt_tokens=len(prompt_ids))
    transformers_module.set_seed(record_seed)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **generation_kwargs(tokenizer),
            logits_processor=transformers_module.LogitsProcessorList([capture]),
        )
    response_ids = [int(value) for value in generated.sequences[0, len(prompt_ids) :].tolist()]
    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    reason = finish_reason(response_ids, tokenizer)
    answer, parse_status, reasoning, sections = strict_answer(
        response, truncated=reason == "length"
    )
    boundary = locate_answer_choice(
        tokenizer,
        response_ids,
        mode=mode,
        strict_choice=answer,
    )
    diagnostics, capture_index, capture_candidates, boundary = align_answer_diagnostics(
        response_ids, capture.by_response_index, answer, boundary
    )
    if answer is not None and (
        diagnostics is None or capture_index != boundary.get("answer_choice_index")
    ):
        failure_diagnostic = {
            "id": row["id"],
            "answer": answer,
            "parse_status": parse_status,
            "response_tokens": len(response_ids),
            "response_tail": response[-2000:],
            "response_token_ids_tail": response_ids[-128:],
            "boundary": boundary,
            "capture_indices": sorted(capture.by_response_index),
            "capture_candidates": capture_candidates,
        }
        raise RuntimeError(
            f"{row['id']}: strict answer lacks aligned on-trajectory logit capture; "
            f"diagnostic={json.dumps(failure_diagnostic, ensure_ascii=False, sort_keys=True)}"
        )
    outcome = classify(row, answer)
    return {
        **identity_fields(row),
        "scope": scope,
        "mode": mode_name,
        "condition": "uncapped",
        "seed": SEED,
        "record_seed": record_seed,
        "thinking_enabled": mode.enable_thinking,
        "prompt_sha256": object_sha256(prompt),
        "prompt_token_ids_sha256": object_sha256(prompt_ids),
        "prompt_tokens": len(prompt_ids),
        "response": response,
        "response_token_ids_sha256": object_sha256(response_ids),
        "response_tokens": len(response_ids),
        "finish_reason": reason,
        "truncated": reason == "length",
        "reasoning_token_count": len(tokenizer.encode(reasoning, add_special_tokens=False)),
        "mode_integrity": mode_integrity(mode, sections),
        "chosen_option": answer,
        "final_answer_parse_status": parse_status,
        "outcome_class": outcome,
        "gold_correct": answer == row["gold_answer"],
        "synchronized_other_error": outcome in {"other_wrong", "invalid_or_abstain"},
        "other_wrong_indicator": outcome == "other_wrong",
        "invalid_or_abstain_indicator": outcome == "invalid_or_abstain",
        "assistant_prefix": ASSISTANT_PREFIX,
        "candidate_token_ids": LABEL_TOKEN_IDS,
        "answer_logits": diagnostics,
        "answer_logit_capture_indices": sorted(capture.by_response_index),
        "selected_answer_logit_index": capture_index,
        "answer_logit_candidates": capture_candidates,
        "answer_boundary": boundary,
        "content_retry_count": 0,
        "answer_repair_performed": False,
    }


def record_key(index: int, row: Mapping[str, Any], scope: str, mode: str) -> dict[str, Any]:
    return {
        "index": index,
        "id": row["id"],
        "scope": scope,
        "mode": mode,
        "condition": "uncapped",
        "seed": SEED,
    }


def materialize(
    store: RecordStore,
    rows: Sequence[dict[str, Any]],
    *,
    scope: str,
    mode: str,
    results_path: Path,
    runtime_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = store.read(record_key(index, row, scope, mode))
        if payload is None:
            continue
        result = payload.get("result")
        if (
            payload.get("index") != index
            or not isinstance(result, dict)
            or result.get("id") != row["id"]
            or result.get("scope") != scope
            or result.get("mode") != mode
        ):
            raise ValueError(f"Checkpoint mismatch at row {index}")
        results.append(result)
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    atomic_write_jsonl(runtime_path, payloads)
    return results, payloads


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_configuration(
    rows: Sequence[dict[str, Any]], scope: str, mode_name: str, git: Mapping[str, Any]
) -> dict[str, Any]:
    mode = MODES[mode_name]
    return {
        "kind": "CEDAR prover-generator Step 5 uncapped development calibration",
        "runner_amendment": RUNNER_AMENDMENT,
        "scope": scope,
        "mode": {"name": mode_name, **asdict(mode)},
        "condition": "uncapped",
        "seed": SEED,
        "dataset": {
            "path": ARMS_PATH.relative_to(ROOT).as_posix(),
            "canonical_text_sha256": ARMS_CANONICAL_TEXT_SHA256,
            "object_sha256": ARMS_OBJECT_SHA256,
            "all_ids_sha256": ARMS_IDS_SHA256,
            "selected_rows": len(rows),
            "selected_ids_sha256": object_sha256([row["id"] for row in rows]),
            "smoke_base_ids": list(SMOKE_BASE_IDS) if scope == "smoke" else None,
            "zero_gpu_gate_canonical_text_sha256": (ZERO_GPU_GATE_CANONICAL_TEXT_SHA256),
            "release_manifest_canonical_text_sha256": (DATA_MANIFEST_CANONICAL_TEXT_SHA256),
            "data_freeze_commit": DATA_FREEZE_COMMIT,
        },
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "generation": {
            "batch_size": 1,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "padding_side": "left",
            "use_cache": True,
            "content_retry": False,
            "answer_repair": False,
            "actual_answer_logit_capture": (
                "read-only JSON-prefix capture with on-trajectory candidate alignment"
            ),
            "candidate_labels": list(LABELS),
            "candidate_token_ids": LABEL_TOKEN_IDS,
        },
        "prohibited_conditions": [
            "p25",
            "p1",
            "held_out_confirmation",
            "LoFa",
            "J-Lens",
        ],
        "git": dict(git),
    }


def main() -> None:
    args = parse_args()
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    all_rows = validate_inputs()
    rows = select_rows(all_rows, args.scope)
    validate_run_order(args.scope, args.mode, args.output_root)
    git_state = git_provenance()
    configuration = build_configuration(rows, args.scope, args.mode, git_state)
    run_dir = args.output_root / run_name(args.scope, args.mode)
    manifest = initialize_run(run_dir, configuration)
    store = RecordStore(run_dir / "records", run_fingerprint=str(manifest["fingerprint"]))
    results_path = run_dir / "results.jsonl"
    runtime_rows_path = run_dir / "record_runtime.jsonl"
    results, _ = materialize(
        store,
        rows,
        scope=args.scope,
        mode=args.mode,
        results_path=results_path,
        runtime_path=runtime_rows_path,
    )

    if args.stage in {"prepare", "all"}:
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "scope": args.scope,
                    "mode": args.mode,
                    "rows": len(rows),
                    "existing": len(results),
                    "run_dir": str(run_dir),
                    "run_fingerprint": manifest["fingerprint"],
                    "data_freeze_commit": DATA_FREEZE_COMMIT,
                    "runner_commit": git_state["commit"],
                    "gpu_used": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.stage == "prepare":
            return

    missing = [
        index
        for index, row in enumerate(rows)
        if not store.contains(record_key(index, row, args.scope, args.mode))
    ]
    if missing:
        if Path(sys.prefix).name != "cedar":
            raise RuntimeError(f"Step 5 generation requires cedar, found {sys.prefix}")
        if torch.cuda.device_count() < 4:
            raise RuntimeError("Qwen3-32B Step 5 requires at least four visible GPUs")
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
        import transformers

        from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model

        max_memory = {
            index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
        }
        update_runtime(
            run_dir,
            {
                "generation_started_at": datetime.now().astimezone().isoformat(),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "platform": platform.platform(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "visible_gpu_count": torch.cuda.device_count(),
                    "hf_home": str(args.hf_home.resolve()),
                },
            },
        )
        spec = QwenModelSpec(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            dtype=MODEL_DTYPE,
            attention_implementation=ATTENTION_IMPLEMENTATION,
        )
        model, tokenizer = load_qwen_model(
            spec,
            max_memory=max_memory,
            local_files_only=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        validate_label_tokens(tokenizer)
        rendered = [render_prompt(tokenizer, row, MODES[args.mode]) for row in rows]
        tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
        rendered_hash = object_sha256(
            [
                {"id": row["id"], "prompt": prompt}
                for row, prompt in zip(rows, rendered, strict=True)
            ]
        )
        tokenized_hash = object_sha256(
            [{"id": row["id"], "input_ids": ids} for row, ids in zip(rows, tokenized, strict=True)]
        )
        key = (args.scope, args.mode)
        if rendered_hash != EXPECTED_RENDERED_PROMPTS_SHA256[key]:
            raise RuntimeError("Frozen Step 5 rendered-prompt hash changed")
        if tokenized_hash != EXPECTED_TOKENIZED_INPUT_IDS_SHA256[key]:
            raise RuntimeError("Frozen Step 5 tokenized-input hash changed")
        update_runtime(
            run_dir,
            {
                "device_map": getattr(model, "hf_device_map", None),
                "rendered_prompts_sha256": rendered_hash,
                "tokenized_input_ids_sha256": tokenized_hash,
                "tokenizer_padding_side": tokenizer.padding_side,
                "tokenizer_pad_token_id": tokenizer.pad_token_id,
                "candidate_token_ids": LABEL_TOKEN_IDS,
            },
        )
        for index in missing:
            row = rows[index]
            result = generate_one(
                model,
                tokenizer,
                transformers,
                row,
                scope=args.scope,
                mode_name=args.mode,
                record_seed=SEED * 1_000_003 + index,
            )
            store.write(
                record_key(index, row, args.scope, args.mode),
                {"index": index, "result": result},
            )
            results, _ = materialize(
                store,
                rows,
                scope=args.scope,
                mode=args.mode,
                results_path=results_path,
                runtime_path=runtime_rows_path,
            )
            print(
                json.dumps(
                    {
                        "stage": "generate",
                        "scope": args.scope,
                        "mode": args.mode,
                        "complete": len(results),
                        "expected": len(rows),
                    }
                ),
                flush=True,
            )
        del model, tokenizer
        release_cuda_cache()

    results, _ = materialize(
        store,
        rows,
        scope=args.scope,
        mode=args.mode,
        results_path=results_path,
        runtime_path=runtime_rows_path,
    )
    if len(results) == len(rows):
        update_runtime(
            run_dir,
            {
                "generation_completed_at": datetime.now().astimezone().isoformat(),
                "complete_results": len(results),
                "results_sha256": file_sha256(results_path),
                "record_runtime_sha256": file_sha256(runtime_rows_path),
                "content_retries": 0,
                "answer_repairs": 0,
                "parse_failures": [
                    row["id"] for row in results if row["final_answer_parse_status"] != "passed"
                ],
                "missing_answer_logits": [
                    row["id"] for row in results if row["answer_logits"] is None
                ],
            },
        )
    print(
        json.dumps(
            {
                "scope": args.scope,
                "mode": args.mode,
                "complete": len(results),
                "expected": len(rows),
                "condition": "uncapped",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
