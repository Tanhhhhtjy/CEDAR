"""Route-neutral finite Horn semantics and external prover adapters.

This module is a clean-room CEDAR implementation. It does not import or copy code from either
upstream generator repository.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cedar.prover_generator_feasibility.v1"
MASTER_SEED = 20260725
MAX_PROOFS_PER_LITERAL = 64


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def literal(predicate: str, *arguments: str, negated: bool = False) -> dict[str, Any]:
    return {
        "predicate": predicate,
        "arguments": list(arguments),
        "negated": bool(negated),
    }


def opposite(item: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(item)
    result["negated"] = not result["negated"]
    return result


def literal_key(item: dict[str, Any]) -> str:
    sign = "-" if item["negated"] else ""
    return f"{sign}{item['predicate']}({','.join(item['arguments'])})"


def fact(
    record_id: str,
    item: dict[str, Any],
    *,
    role: str = "structural",
) -> dict[str, Any]:
    return {"record_id": record_id, "kind": "fact", "literal": item, "role": role}


def rule(
    record_id: str,
    body: list[dict[str, Any]],
    head: dict[str, Any],
    *,
    role: str = "structural",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "kind": "rule",
        "body": body,
        "head": head,
        "role": role,
    }


def _variables_in_literal(item: dict[str, Any]) -> set[str]:
    return {arg for arg in item["arguments"] if arg.startswith("?")}


def variables_in_rule(record: dict[str, Any]) -> list[str]:
    variables: set[str] = set()
    for item in record["body"] + [record["head"]]:
        variables.update(_variables_in_literal(item))
    return sorted(variables)


def ground_literal(item: dict[str, Any], substitution: dict[str, str]) -> dict[str, Any]:
    return {
        "predicate": item["predicate"],
        "arguments": [substitution.get(arg, arg) for arg in item["arguments"]],
        "negated": item["negated"],
    }


def evaluate_fixed_point(world: dict[str, Any]) -> dict[str, Any]:
    """Compute the signed least fixed point and bounded distinct proof-record sets."""

    constants = sorted(world["constants"])
    proofs: dict[str, dict[tuple[str, ...], int]] = {}

    for record in world["records"]:
        if record["kind"] != "fact":
            continue
        key = literal_key(record["literal"])
        proofs.setdefault(key, {})[(record["record_id"],)] = 0

    changed = True
    while changed:
        changed = False
        for record in world["records"]:
            if record["kind"] != "rule":
                continue
            variables = variables_in_rule(record)
            substitutions = (
                [{}]
                if not variables
                else [
                    dict(zip(variables, values, strict=True))
                    for values in itertools.product(constants, repeat=len(variables))
                ]
            )
            for substitution in substitutions:
                body_keys = [
                    literal_key(ground_literal(item, substitution)) for item in record["body"]
                ]
                if any(key not in proofs for key in body_keys):
                    continue
                body_options = [list(proofs[key].items()) for key in body_keys]
                for option_tuple in itertools.product(*body_options):
                    used_records = {record["record_id"]}
                    body_depth = 0
                    for proof_records, depth in option_tuple:
                        used_records.update(proof_records)
                        body_depth = max(body_depth, depth)
                    proof_key = tuple(sorted(used_records))
                    head_key = literal_key(ground_literal(record["head"], substitution))
                    head_proofs = proofs.setdefault(head_key, {})
                    new_depth = body_depth + 1
                    if proof_key not in head_proofs:
                        if len(head_proofs) < MAX_PROOFS_PER_LITERAL:
                            head_proofs[proof_key] = new_depth
                            changed = True
                    elif new_depth < head_proofs[proof_key]:
                        head_proofs[proof_key] = new_depth
                        changed = True

    normalized_proofs: dict[str, list[dict[str, Any]]] = {}
    for key, entries in sorted(proofs.items()):
        normalized_proofs[key] = sorted(
            (
                {"record_ids": list(record_ids), "depth": depth}
                for record_ids, depth in entries.items()
            ),
            key=lambda entry: (entry["depth"], entry["record_ids"]),
        )

    return {
        "closure": sorted(proofs),
        "proofs": normalized_proofs,
        "proof_cap": MAX_PROOFS_PER_LITERAL,
    }


def label_from_evaluation(evaluation: dict[str, Any], query: dict[str, Any]) -> str:
    positive = literal_key(query) in evaluation["closure"]
    negative = literal_key(opposite(query)) in evaluation["closure"]
    if positive and negative:
        return "invalid"
    if positive:
        return "true"
    if negative:
        return "false"
    return "unknown"


def shortest_proof(evaluation: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    entries = evaluation["proofs"].get(literal_key(item), [])
    return copy.deepcopy(entries[0]) if entries else None


def proof_count(evaluation: dict[str, Any], item: dict[str, Any]) -> int:
    return len(evaluation["proofs"].get(literal_key(item), []))


def decisive_literal(query: dict[str, Any], label: str) -> dict[str, Any] | None:
    if label == "true":
        return copy.deepcopy(query)
    if label == "false":
        return opposite(query)
    return None


def apply_edit(world: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(world)
    records = result["records"]
    operation = edit["operation"]
    if operation == "add":
        if any(record["record_id"] == edit["after"]["record_id"] for record in records):
            raise ValueError("add edit reuses an existing record_id")
        records.append(copy.deepcopy(edit["after"]))
    elif operation == "delete":
        before = edit["before"]
        matches = [index for index, record in enumerate(records) if record == before]
        if len(matches) != 1:
            raise ValueError("delete edit must match exactly one complete record")
        records.pop(matches[0])
    elif operation == "replace":
        before = edit["before"]
        matches = [index for index, record in enumerate(records) if record == before]
        if len(matches) != 1:
            raise ValueError("replace edit must match exactly one complete record")
        records[matches[0]] = copy.deepcopy(edit["after"])
    else:
        raise ValueError(f"unsupported edit operation: {operation}")
    result["records"] = sorted(records, key=lambda record: record["record_id"])
    return result


def edit_count(edit: dict[str, Any]) -> int:
    return 1 if edit["operation"] in {"add", "delete", "replace"} else 0


def _prover9_atom(item: dict[str, Any]) -> str:
    arguments = [arg[1:] if arg.startswith("?") else arg for arg in item["arguments"]]
    atom = f"{item['predicate']}({','.join(arguments)})"
    return f"-{atom}" if item["negated"] else atom


def _prover9_record(record: dict[str, Any]) -> str:
    if record["kind"] == "fact":
        return f"{_prover9_atom(record['literal'])}."
    variables = [variable[1:] for variable in variables_in_rule(record)]
    body = " & ".join(_prover9_atom(item) for item in record["body"])
    formula = f"({body}) -> {_prover9_atom(record['head'])}"
    if variables:
        quantifiers = " ".join(f"all {variable}" for variable in variables)
        formula = f"{quantifiers} ({formula})"
    return f"{formula}."


def prover9_input(world: dict[str, Any], goal: dict[str, Any]) -> str:
    assumptions = "\n".join(_prover9_record(record) for record in world["records"])
    return (
        "set(auto).\n"
        "assign(max_seconds, 5).\n"
        "formulas(assumptions).\n"
        f"{assumptions}\n"
        "end_of_list.\n"
        "formulas(goals).\n"
        f"{_prover9_atom(goal)}.\n"
        "end_of_list.\n"
    )


def mace4_input(world: dict[str, Any], required_literal: dict[str, Any]) -> str:
    assumptions = "\n".join(_prover9_record(record) for record in world["records"])
    return (
        "assign(max_seconds, 5).\n"
        "formulas(assumptions).\n"
        f"{assumptions}\n"
        f"{_prover9_atom(required_literal)}.\n"
        "end_of_list.\n"
    )


def locate_solver(name: str) -> Path:
    environment_name = f"CEDAR_{name.upper()}"
    if os.environ.get(environment_name):
        candidate = Path(os.environ[environment_name]).expanduser().resolve()
        if candidate.is_file():
            return candidate
    repo_root = Path(__file__).resolve().parents[3]
    sibling = repo_root.parent / "external_refs" / "tools" / "prover9_gui" / "bin-win32"
    candidate = sibling / f"{name}.exe"
    if candidate.is_file():
        return candidate
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / (f"{name}.exe" if os.name == "nt" else name)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{name} not found; set {environment_name} or install it outside the repository"
    )


def _solver_section(output: str, name: str, end_name: str) -> str | None:
    pattern = re.compile(
        rf"^=+ {re.escape(name)} =+\n(.*?)^=+ {re.escape(end_name)} =+$",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(output)
    return match.group(1).strip() if match else None


def _sanitize_solver_output(output: str, solver: str) -> str:
    """Keep deterministic evidence sections and omit process/time/path telemetry."""

    output = output.replace("\x07", "").replace("\r\n", "\n")
    if solver == "prover9":
        version_match = re.search(r"Prover9 \(32\) version [^\n]+", output)
        input_section = _solver_section(output, "INPUT", "end of input")
        proof_section = _solver_section(output, "PROOF", "end of proof")
        if proof_section:
            proof_section = re.sub(
                r"^% Proof \d+ at .*? seconds\.$",
                "% Proof timing omitted.",
                proof_section,
                flags=re.MULTILINE,
            )
        if "THEOREM PROVED" in output:
            outcome = "THEOREM PROVED"
        elif "SEARCH FAILED" in output:
            outcome = "SEARCH FAILED"
        else:
            outcome = "UNRECOGNIZED SOLVER OUTCOME"
        parts = [version_match.group(0) if version_match else "Prover9 version unknown"]
        if input_section:
            parts.append("INPUT\n" + input_section)
        if proof_section:
            parts.append("PROOF\n" + proof_section)
        parts.append(outcome)
        return "\n\n".join(parts).strip() + "\n"

    version_match = re.search(r"Mace4 \(32\) version [^\n]+", output)
    input_section = _solver_section(output, "INPUT", "end of input")
    model_section = _solver_section(output, "MODEL", "end of model")
    if model_section:
        model_section = re.sub(
            r"interpretation\(\s*([0-9]+),\s*\[number=[0-9]+, seconds=[0-9]+\]",
            r"interpretation( \1, [number=<N>, seconds=<TIME>]",
            model_section,
        )
    parts = [version_match.group(0) if version_match else "Mace4 version unknown"]
    if input_section:
        parts.append("INPUT\n" + input_section)
    if model_section:
        parts.append("MODEL\n" + model_section)
    parts.append("MODEL FOUND" if model_section else "MODEL NOT FOUND")
    return "\n\n".join(parts).strip() + "\n"


def run_prover9(world: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    executable = locate_solver("prover9")
    payload = prover9_input(world, goal)
    completed = subprocess.run(
        [str(executable)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    combined = _sanitize_solver_output(completed.stdout + completed.stderr, "prover9")
    proved = "THEOREM PROVED" in combined
    search_failed = "SEARCH FAILED" in combined
    if not proved and not search_failed:
        raise RuntimeError(f"unrecognized Prover9 outcome (exit={completed.returncode})")
    return {
        "goal": literal_key(goal),
        "input_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "proved": proved,
        "search_failed": search_failed,
        "trace": combined,
    }


def run_mace4_countermodel(world: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    executable = locate_solver("mace4")
    required = opposite(target)
    payload = mace4_input(world, required)
    completed = subprocess.run(
        [str(executable), "-n", "2", "-N", "4", "-m", "1", "-t", "5"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    raw_output = completed.stdout + completed.stderr
    combined = _sanitize_solver_output(raw_output, "mace4")
    model_found = "interpretation(" in raw_output and "Exiting with 1 model" in raw_output
    if not model_found:
        raise RuntimeError("Mace4 did not produce a countermodel for an underived target")
    return {
        "target_refuted": literal_key(target),
        "required_counterliteral": literal_key(required),
        "input_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "model_found": True,
        "trace": combined,
    }


def prover9_label(world: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    query_result = run_prover9(world, query)
    opposite_result = run_prover9(world, opposite(query))
    if query_result["proved"] and opposite_result["proved"]:
        label = "invalid"
    elif query_result["proved"]:
        label = "true"
    elif opposite_result["proved"]:
        label = "false"
    else:
        label = "unknown"
    return {
        "label": label,
        "query_result": query_result,
        "opposite_result": opposite_result,
    }


def validate_with_both(world: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    fixed_point = evaluate_fixed_point(world)
    python_label = label_from_evaluation(fixed_point, query)
    external = prover9_label(world, query)
    return {
        "python_label": python_label,
        "prover9_label": external["label"],
        "agreement": python_label == external["label"],
        "fixed_point": fixed_point,
        "prover9": external,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_jsonl_sha256(path: Path) -> str:
    rows = read_jsonl(path)
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json_file_sha256(path: Path) -> str:
    return canonical_sha256(json.loads(path.read_text(encoding="utf-8")))
