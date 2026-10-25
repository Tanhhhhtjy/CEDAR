"""Shared v1.1 helpers built on the frozen v1 formal-logic infrastructure."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V1_DIR = HERE.parent / "prover_generator_feasibility_v1"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from formal_horn import (  # noqa: E402,F401
    apply_edit,
    canonical_json,
    canonical_json_file_sha256,
    canonical_jsonl_sha256,
    canonical_sha256,
    decisive_literal,
    edit_count,
    evaluate_fixed_point,
    fact,
    file_sha256,
    label_from_evaluation,
    literal,
    literal_key,
    locate_solver,
    opposite,
    proof_count,
    read_jsonl,
    rule,
    run_mace4_countermodel,
    shortest_proof,
    stable_seed,
    validate_with_both,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = "cedar.prover_generator_hardening.v1_1"
ARCHITECTURE = "proof_graph_plus_atomic_edit_certification"
MASTER_SEED = 20260726
MAX_CANDIDATES = 100
ERROR_CLASSES = (
    "cross_entity_transfer",
    "converse_error",
    "conjunction_weakening",
    "unsupported_bridge",
    "scope_transfer",
    "support_path_generalization",
)


def record_semantics(record: dict[str, Any]) -> dict[str, Any]:
    if record["kind"] == "fact":
        return {"kind": "fact", "literal": copy.deepcopy(record["literal"])}
    return {
        "kind": "rule",
        "body": copy.deepcopy(record["body"]),
        "head": copy.deepcopy(record["head"]),
    }


def record_semantic_hash(record: dict[str, Any]) -> str:
    return canonical_sha256(record_semantics(record))


def literal_from_key(key: str) -> dict[str, Any]:
    negated = key.startswith("-")
    if negated:
        key = key[1:]
    predicate, arguments = key[:-1].split("(", maxsplit=1)
    return literal(predicate, *([] if not arguments else arguments.split(",")), negated=negated)


def canonical_tree_hash(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".jsonl":
            hashes[relative] = canonical_jsonl_sha256(path)
        elif path.suffix == ".json":
            hashes[relative] = canonical_json_file_sha256(path)
        elif path.suffix in {".md", ".py"}:
            normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            hashes[relative] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashes


def json_roundtrip(value: Any) -> Any:
    return json.loads(canonical_json(value))
