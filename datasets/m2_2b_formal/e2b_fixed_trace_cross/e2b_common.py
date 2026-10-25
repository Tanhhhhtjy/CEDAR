"""Frozen contracts and deterministic helpers for the E2B fixed-trace assay."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E2A_DATA = HERE.parent / "e2_phase_validity_choice"
if str(E2A_DATA) not in sys.path:
    sys.path.insert(0, str(E2A_DATA))

import e2_common as e2a  # noqa: E402

SCHEMA = "cedar.e2b_fixed_trace_cross.v1"
BASE_FREEZE_COMMIT = "10120d04d00e3c9122bf2a2ba590bd403373fa45"
ANCHOR_RESULT_COMMIT = "0c8517bb5fb4640f7d5e4d51356d0cf167046643"
ANCHOR_RESULTS = ROOT / "artifacts/m2/e2_phase_validity_choice/formal-all-seed0/results.jsonl"
ANCHOR_AUDIT = ROOT / "artifacts/m2/e2_phase_validity_choice/E2A_AUDIT.json"
PROTOCOL = ROOT / "docs/E2B_FIXED_TRACE_CROSS_PROTOCOL.md"
ANALYSIS_PLAN = ROOT / "docs/E2B_FIXED_TRACE_CROSS_ANALYSIS_PLAN.json"
FREEZE_MANIFEST = ROOT / "docs/E2_PROTOCOL_FREEZE_MANIFEST.json"

MODEL_REPO = e2a.MODEL_REPO
MODEL_REVISION = e2a.MODEL_REVISION
MODEL_DTYPE = e2a.MODEL_DTYPE
ATTENTION_IMPLEMENTATION = e2a.ATTENTION_IMPLEMENTATION
CAP_EXPERIMENT = e2a.CAP_EXPERIMENT
CAP_LAYERS = e2a.CAP_LAYERS
EXPECTED_AXIS_SHA256 = e2a.EXPECTED_AXIS_SHA256
EXPECTED_CAP_CONFIG_SHA256 = e2a.EXPECTED_CAP_CONFIG_SHA256

LABELS = ("A", "B", "C", "D")
TRACE_SOURCES = ("e2a_uncapped", "e2a_full_p25")
FORWARD_CONDITIONS = ("uncapped", "full_p25")
SOURCE_CONDITION = {"e2a_uncapped": "uncapped", "e2a_full_p25": "full_p25"}
EXPECTED_PROMPTS = 96
EXPECTED_TRACES = 192
EXPECTED_FORMAL_CELLS = 384
EXPECTED_SMOKE_PROMPTS = 8
EXPECTED_SMOKE_CELLS = 32
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260726
REPLAY_TOLERANCE = 1e-4


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    require(all(isinstance(value, dict) for value in values), f"Expected JSON objects: {path}")
    return values


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(left: float, right: float, tolerance: float = REPLAY_TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def validate_frozen_protocol_hashes() -> dict[str, str]:
    manifest = read_json(FREEZE_MANIFEST)
    observed = {
        relative: canonical_text_sha256(ROOT / relative)
        for relative in sorted(manifest["files"])
    }
    require(observed == manifest["files"], "Frozen E2 protocol hashes changed")
    require(manifest.get("gpu_authorized") is False, "Protocol freeze unexpectedly authorizes GPU")
    return observed


def validate_anchor() -> dict[str, Any]:
    require(ANCHOR_RESULTS.is_file(), "E2A formal results are missing")
    require(ANCHOR_AUDIT.is_file(), "E2A formal audit is missing")
    rows = read_jsonl(ANCHOR_RESULTS)
    audit = read_json(ANCHOR_AUDIT)
    require(len(rows) == 576, "E2A result count changed")
    require(audit.get("status") == "passed", "E2A audit is not passed")
    require(audit.get("anchor_equivalence", {}).get("passed") is True, "E2A anchor failed")
    require(not audit.get("failures"), "E2A audit contains failures")
    answer = [
        row
        for row in rows
        if row.get("assay") == "answer"
        and row.get("condition") in {"uncapped", "full_p25"}
    ]
    require(len(answer) == EXPECTED_TRACES, "E2A Answer trace count changed")
    require(
        len({(row["id"], row["condition"]) for row in answer}) == EXPECTED_TRACES,
        "E2A Answer traces are not unique",
    )
    return {
        "rows": len(rows),
        "answer_source_rows": len(answer),
        "results_canonical_lf_sha256": canonical_text_sha256(ANCHOR_RESULTS),
        "audit_canonical_lf_sha256": canonical_text_sha256(ANCHOR_AUDIT),
    }


def smoke_prompt_ids(prompt_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in prompt_rows:
        item_id = row.get("id", row.get("source_id"))
        require(isinstance(item_id, str), "Smoke candidate lacks an ID")
        grouped[(str(row["arm"]), str(row["variant"]))].append(item_id)
    expected = {(arm, variant) for arm in ("N", "R", "V", "F") for variant in ("easy", "medium")}
    require(set(grouped) == expected, "Arm-by-variant smoke strata changed")
    selected = []
    for cell in sorted(grouped):
        selected.append(
            min(
                grouped[cell],
                key=lambda item: hashlib.sha256(
                    f"cedar-e2b-smoke-v1|{item}".encode()
                ).hexdigest(),
            )
        )
    require(len(selected) == EXPECTED_SMOKE_PROMPTS, "E2B smoke selection changed")
    return selected


def ordered_cells(
    traces: Sequence[Mapping[str, Any]], *, scope: str, smoke_ids: Sequence[str]
) -> list[tuple[dict[str, Any], str]]:
    selected = [dict(trace) for trace in traces]
    if scope == "smoke":
        allowed = set(smoke_ids)
        selected = [trace for trace in selected if trace["source_id"] in allowed]
        require(len(selected) == 2 * EXPECTED_SMOKE_PROMPTS, "Smoke trace count changed")
    elif scope != "formal":
        raise ValueError(f"Unknown E2B scope: {scope}")
    cells = [(trace, condition) for trace in selected for condition in FORWARD_CONDITIONS]
    expected = EXPECTED_SMOKE_CELLS if scope == "smoke" else EXPECTED_FORMAL_CELLS
    require(len(cells) == expected, "E2B cell count changed")
    return cells


def _bytelevel_tables(tokenizer: Any) -> tuple[dict[int, bytes], dict[bytes, list[int]]]:
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    inverse = {character: byte for byte, character in bytes_to_unicode().items()}
    id_to_bytes: dict[int, bytes] = {}
    bytes_to_ids: dict[bytes, list[int]] = defaultdict(list)
    for token, token_id in tokenizer.get_vocab().items():
        try:
            raw = bytes(inverse[character] for character in token)
        except KeyError:
            continue
        id_to_bytes[int(token_id)] = raw
        bytes_to_ids[raw].append(int(token_id))
    for token_ids in bytes_to_ids.values():
        token_ids.sort()
    return id_to_bytes, dict(bytes_to_ids)


def _split_once_candidates(
    ids: Sequence[int], id_to_bytes: Mapping[int, bytes], bytes_to_ids: Mapping[bytes, list[int]]
) -> list[tuple[int, int, tuple[int, int]]]:
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for position, token_id in enumerate(ids):
        raw = id_to_bytes.get(int(token_id))
        if raw is None:
            continue
        for cut in range(1, len(raw)):
            for left in bytes_to_ids.get(raw[:cut], []):
                for right in bytes_to_ids.get(raw[cut:], []):
                    candidates.append((position, int(token_id), (left, right)))
    return candidates


def reconstruct_response_ids(tokenizer: Any, row: Mapping[str, Any]) -> tuple[list[int], str]:
    """Invert the stored decoded response and prove identity with its frozen token hash.

    Byte-level BPE decoding is not injective: a generated space token followed by a
    newline token may re-encode as the merged space-newline token. The frozen SHA-256
    and token count provide an unambiguous identity check. This routine performs a
    deterministic, model-free split search; it never invents or repairs text.
    """

    canonical = [
        int(value)
        for value in tokenizer.encode(row["response"], add_special_tokens=False)
    ]
    suffix: list[int] = []
    if row.get("finish_reason") == "eos_token":
        require(isinstance(tokenizer.eos_token_id, int), "Pinned tokenizer has no EOS token")
        suffix = [int(tokenizer.eos_token_id)]
    expected_count = int(row["response_tokens"])
    expected_hash = str(row["response_token_ids_sha256"])
    expected_body_count = expected_count - len(suffix)

    def matches(body: Sequence[int]) -> bool:
        full = [*body, *suffix]
        return len(full) == expected_count and object_sha256(full) == expected_hash

    if matches(canonical):
        return [*canonical, *suffix], "canonical_reencode"

    delta = expected_body_count - len(canonical)
    require(delta in {1, 2}, f"Unsupported noncanonical token-count delta: {delta}")
    id_to_bytes, bytes_to_ids = _bytelevel_tables(tokenizer)
    splits = _split_once_candidates(canonical, id_to_bytes, bytes_to_ids)
    matches_found: list[tuple[list[int], str]] = []

    if delta == 1:
        for position, _old, replacement in splits:
            candidate = [*canonical[:position], *replacement, *canonical[position + 1 :]]
            if matches(candidate):
                matches_found.append((candidate, "bytelevel_single_split"))
    else:
        # First cover a single token represented by three byte-level vocabulary tokens.
        for position, token_id in enumerate(canonical):
            raw = id_to_bytes.get(token_id)
            if raw is None:
                continue
            for first in range(1, len(raw) - 1):
                for second in range(first + 1, len(raw)):
                    for left in bytes_to_ids.get(raw[:first], []):
                        for middle in bytes_to_ids.get(raw[first:second], []):
                            for right in bytes_to_ids.get(raw[second:], []):
                                replacement = (left, middle, right)
                                candidate = [
                                    *canonical[:position],
                                    *replacement,
                                    *canonical[position + 1 :],
                                ]
                                if matches(candidate):
                                    matches_found.append((candidate, "bytelevel_triple_split"))
        # Generated traces can repeat the same noncanonical split. Search every pair
        # sharing the same original token and replacement, which covers the frozen
        # Qwen space/newline ambiguity without any row-specific exception.
        grouped: dict[tuple[int, tuple[int, int]], list[int]] = defaultdict(list)
        for position, old, replacement in splits:
            grouped[(old, replacement)].append(position)
        for (_old, replacement), positions in sorted(grouped.items()):
            for left_index, left_position in enumerate(positions):
                for right_position in positions[left_index + 1 :]:
                    candidate: list[int] = []
                    for position, token_id in enumerate(canonical):
                        if position in {left_position, right_position}:
                            candidate.extend(replacement)
                        else:
                            candidate.append(token_id)
                    if matches(candidate):
                        matches_found.append((candidate, "bytelevel_repeated_split"))

    unique = {tuple(candidate): method for candidate, method in matches_found}
    require(len(unique) == 1, f"Response token reconstruction is not unique: {len(unique)}")
    body_tuple, method = next(iter(unique.items()))
    full = [*body_tuple, *suffix]
    decoded = tokenizer.decode(full, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    require(decoded == row["response"], "Reconstructed response text changed")
    return full, method


def geometry(row: Mapping[str, Any]) -> dict[str, float]:
    diagnostics = row.get("answer_logits") or row.get("logit_diagnostics")
    logits = diagnostics.get("logits") if isinstance(diagnostics, Mapping) else None
    require(isinstance(logits, Mapping) and set(logits) == set(LABELS), "Missing A/B/C/D logits")
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    other = [label for label in LABELS if label not in {initial, target}]
    initial_logit = float(logits[initial])
    target_relative = float(logits[target]) - initial_logit
    other_relative = [float(logits[label]) - initial_logit for label in other]
    background = statistics.fmean(other_relative)
    return {
        "m_target": target_relative,
        "m_background": background,
        "m_general": statistics.fmean([target_relative, *other_relative]),
        "m_specific": target_relative - background,
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "Cannot compute an empty percentile")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def family_bootstrap(scores: Mapping[str, float]) -> dict[str, Any]:
    families = sorted(scores)
    require(len(families) == 12, "E2B bootstrap requires exactly 12 families")
    estimate = statistics.fmean(scores.values())
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        statistics.fmean(scores[rng.choice(families)] for _ in families)
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "estimate": estimate,
        "family_cluster_bootstrap_95ci": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "families": len(families),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "family_scores": dict(sorted(scores.items())),
        "probability_positive": sum(value > 0 for value in draws) / len(draws),
        "probability_negative": sum(value < 0 for value in draws) / len(draws),
    }
