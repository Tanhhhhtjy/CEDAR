"""Frozen runtime selection and hashing helpers for E2 new-salt confirmation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Keep the GPU runtime path independent of workbook/solver build dependencies.
# These values are duplicated from the frozen construction module and bound by
# CPU_GATE.json, so importing this module in the cedar GPU environment does not
# require openpyxl or the theorem-prover build stack.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("N", "R", "V", "F")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026072502
CAP_EXPERIMENT = "layers_46:54-p0.25"
CAP_LAYERS = tuple(range(46, 54))
ERROR_CLASSES = (
    "cross_entity_transfer",
    "converse_error",
    "conjunction_weakening",
    "unsupported_bridge",
    "scope_transfer",
    "support_path_generalization",
)
EXPECTED_AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
EXPECTED_CAP_CONFIG_SHA256 = (
    "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
)
MODEL_REPO = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
SALT = "cedar-e2-new-salt-confirmation-v1-20260725"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_jsonl_sha256(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
CONDITIONS = ("uncapped", "full_p25")
FORMAL_INPUT_ROWS = 96
FORMAL_RESULT_ROWS = 192
SMOKE_BASES = 6
SMOKE_INPUT_ROWS = 24
SMOKE_RESULT_ROWS = 48
MAX_NEW_TOKENS = 4096
SEED = 0
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
CPU_GATE_PATH = HERE / "CPU_GATE.json"
MANIFEST_PATH = HERE / "manifest.json"
ARMS_PATH = HERE / "arms96.jsonl"
BASES_PATH = HERE / "bases24.jsonl"
TOKENIZER_PATH = HERE / "tokenizer_preflight.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    require(all(isinstance(row, dict) for row in rows), f"Expected JSON objects: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_smoke_base_ids(bases: Sequence[dict[str, Any]] | None = None) -> list[str]:
    source = list(bases) if bases is not None else read_jsonl(BASES_PATH)
    selected = []
    for index, error_class in enumerate(sorted(ERROR_CLASSES)):
        variant = "easy" if index % 2 == 0 else "medium"
        candidates = [
            row
            for row in source
            if row["error_class"] == error_class and row["variant"] == variant
        ]
        require(len(candidates) == 2, f"Smoke cell {error_class}/{variant} expected 2 bases")
        chosen = min(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{SALT}|smoke|{row['base_id']}".encode()
            ).hexdigest(),
        )
        selected.append(chosen["base_id"])
    require(len(selected) == len(set(selected)) == SMOKE_BASES, "Smoke base selection changed")
    return selected


def select_rows(scope: str) -> list[dict[str, Any]]:
    require(scope in {"smoke", "formal"}, f"Unknown scope: {scope}")
    arms = read_jsonl(ARMS_PATH)
    require(len(arms) == FORMAL_INPUT_ROWS, "Formal input row count changed")
    require(
        Counter(row["arm"] for row in arms) == Counter({arm: 24 for arm in ARMS}),
        "Arm balance changed",
    )
    if scope == "formal":
        selected = arms
        expected = FORMAL_INPUT_ROWS
    else:
        by_base: dict[str, dict[str, dict[str, Any]]] = {}
        for row in arms:
            by_base.setdefault(row["base_id"], {})[row["arm"]] = row
        selected = [
            by_base[base_id][arm]
            for base_id in select_smoke_base_ids()
            for arm in ARMS
        ]
        expected = SMOKE_INPUT_ROWS
    require(
        len(selected) == len({row["id"] for row in selected}) == expected,
        "Selection row count changed",
    )
    return selected


def ordered_run_cells(rows: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    return [(row, condition) for row in rows for condition in CONDITIONS]


def selection_fingerprint(scope: str) -> dict[str, Any]:
    rows = select_rows(scope)
    return {
        "scope": scope,
        "rows": len(rows),
        "result_cells": len(rows) * len(CONDITIONS),
        "ids_sha256": object_sha256([row["id"] for row in rows]),
        "messages_sha256": object_sha256(
            [{"id": row["id"], "messages": row["messages"]} for row in rows]
        ),
    }


def validate_cpu_gate() -> dict[str, Any]:
    gate = read_json(CPU_GATE_PATH)
    manifest = read_json(MANIFEST_PATH)
    tokenizer = read_json(TOKENIZER_PATH)
    require(gate.get("gate") == "E2-new-salt-confirmation", "CPU Gate identity changed")
    require(gate.get("gpu_authorized") is True, "CPU Gate does not authorize GPU")
    require(gate.get("may_run_smoke") is True, "CPU Gate does not authorize smoke")
    require(gate.get("p1_authorized") is False, "CPU Gate unexpectedly authorizes p1")
    require(gate.get("conditions") == list(CONDITIONS), "CPU Gate conditions changed")
    require(gate.get("formal_rows") == FORMAL_RESULT_ROWS, "CPU Gate formal count changed")
    require(manifest.get("status") == "ready_for_gpu_confirmation", "Manifest is not release-ready")
    require(manifest.get("gpu_authorized") is True, "Manifest does not authorize GPU")
    require(tokenizer.get("status") == "passed", "Tokenizer preflight is not passed")
    require(
        canonical_jsonl_sha256(ARMS_PATH)
        == gate["data_hashes"]["arms96_canonical_sha256"],
        "Arms changed after Gate",
    )
    code_paths = {
        "run_common_raw_sha256": HERE / "e2_new_salt_run_common.py",
        "runner_raw_sha256": ROOT / "scripts" / "run_e2_new_salt_confirmation.py",
        "auditor_raw_sha256": ROOT / "scripts" / "audit_e2_new_salt_confirmation.py",
        "analyzer_raw_sha256": ROOT / "scripts" / "analyze_e2_new_salt_confirmation.py",
    }
    for field, path in code_paths.items():
        require(
            file_sha256(path) == gate["code_hashes"][field],
            f"Frozen runtime code changed after Gate: {path.name}",
        )
    for scope in ("smoke", "formal"):
        require(
            selection_fingerprint(scope) == gate["selection_fingerprints"][scope],
            f"{scope} selection changed after Gate",
        )
    return gate


__all__ = [
    "ARMS",
    "ATTENTION_IMPLEMENTATION",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CAP_EXPERIMENT",
    "CAP_LAYERS",
    "CONDITIONS",
    "EXPECTED_AXIS_SHA256",
    "EXPECTED_CAP_CONFIG_SHA256",
    "FORMAL_INPUT_ROWS",
    "FORMAL_RESULT_ROWS",
    "MAX_NEW_TOKENS",
    "MODEL_DTYPE",
    "MODEL_REPO",
    "MODEL_REVISION",
    "SEED",
    "SMOKE_INPUT_ROWS",
    "SMOKE_RESULT_ROWS",
    "file_sha256",
    "ordered_run_cells",
    "read_json",
    "read_jsonl",
    "select_rows",
    "select_smoke_base_ids",
    "selection_fingerprint",
    "validate_cpu_gate",
    "write_json",
]
