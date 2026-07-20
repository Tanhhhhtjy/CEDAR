#!/usr/bin/env python3
"""Deterministically expand approved C1/V7 family/base inputs without invoking a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from validate_m2_2b_v7_heldout import (
    MAX_SINGLE_FAMILY_HARMFUL_SHARE,
    MIN_HARMFUL_FAMILIES,
    REQUIRE_POSITIVE_LOFO,
    REVIEW_COLUMNS,
    VERSION,
    canonical_sha256,
    expand_arms,
    load_jsonl,
    render_frozen_arms,
    validate_base_cards,
    validate_family_cards,
    validate_family_feasibility_audit,
    validate_flat_arms,
    validate_overlap_audit,
    validate_overlap_matrix,
)

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs" / "C1_V7_HELDOUT_PROTOCOL.md"
METHOD_AUDIT = ROOT / "docs" / "C1_V7_REFERENCE_METHOD_AUDIT.md"
EXECUTION_RUNBOOK = ROOT / "docs" / "C1_V7_PHASE2_TO_CONFIRMATION_RUNBOOK.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
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


def construction_audit_rows(bases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "base_id": base["base_id"],
            "family_id": base["family_id"],
            "review_type": "author_construction_audit",
            "independent_human_review": False,
            "human_review_status": "not_started",
            "initial_gold_answer": base["initial_answer"],
            "target_answer": base["target_answer"],
            "distractor_answer": base["distractor_answer"],
            "original_gold_rationale": base["original_gold_rationale"],
            "updated_gold_rationale": base["updated_gold_rationale"],
            "construction_certificate": base["construction_certificate"],
            "N_gold": base["initial_answer"],
            "AO_gold": base["initial_answer"],
            "S_gold": base["initial_answer"],
            "V_gold": base["target_answer"],
            "notice": "Construction annotation; not independent human review.",
        }
        for base in sorted(bases, key=lambda value: value["base_id"])
    ]


def _extract_development_text(row: dict[str, Any]) -> str | None:
    if isinstance(row.get("initial_user"), str):
        return str(row["initial_user"])
    messages = row.get("messages") or row.get("messages_common")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return None


def normalize_surface(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", text)
    text = re.sub(r"\b[a-z]{1,4}-?\d+\b", "<entity>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lexical_overlap_report(
    bases: list[dict[str, Any]], development_root: Path
) -> dict[str, Any]:
    current = {
        base["base_id"]: str(base["messages_common"][1]["content"])
        for base in bases
    }
    exact_index: dict[str, list[str]] = {}
    surface_index: dict[str, list[str]] = {}
    files_scanned = 0
    rows_scanned = 0
    for path in sorted(development_root.rglob("*.jsonl")):
        if HERE in path.parents:
            continue
        files_scanned += 1
        try:
            rows = load_jsonl(path)
        except (OSError, ValueError):
            continue
        for row in rows:
            text = _extract_development_text(row)
            if not text:
                continue
            rows_scanned += 1
            identifier = str(row.get("id") or row.get("base_id") or f"{path.name}:{rows_scanned}")
            exact_index.setdefault(re.sub(r"\s+", " ", text.strip().lower()), []).append(identifier)
            surface_index.setdefault(normalize_surface(text), []).append(identifier)
    exact_matches: list[dict[str, Any]] = []
    normalized_matches: list[dict[str, Any]] = []
    for base_id, text in current.items():
        exact = exact_index.get(re.sub(r"\s+", " ", text.strip().lower()), [])
        surface = surface_index.get(normalize_surface(text), [])
        if exact:
            exact_matches.append({"base_id": base_id, "development_ids": sorted(exact)})
        if surface:
            normalized_matches.append(
                {"base_id": base_id, "development_ids": sorted(surface)}
            )
    return {
        "status": "passed" if not exact_matches and not normalized_matches else "failed",
        "scope": "lexical exact and number/entity-normalized surfaces only",
        "manual_structural_audit_still_required": True,
        "development_files_scanned": files_scanned,
        "development_rows_with_text": rows_scanned,
        "exact_matches": exact_matches,
        "normalized_surface_matches": normalized_matches,
        "qwen_or_gpu_execution": False,
    }


def _blind_id(salt: bytes, base_id: str) -> str:
    digest = hmac.new(salt, base_id.encode(), hashlib.sha256).hexdigest()[:16].upper()
    return f"C1V7-BLIND-{digest}"


def _review_row(base: dict[str, Any], blind_id: str) -> dict[str, str]:
    arms = render_frozen_arms(base)
    return {
        "blind_item_id": blind_id,
        "original_user": base["messages_common"][1]["content"],
        "initial_assistant": base["messages_common"][2]["content"],
        "n_followup": arms["N"]["followup"],
        "v_followup": arms["V"]["followup"],
        "ao_followup": arms["AO"]["followup"],
        "s_followup": arms["S"]["followup"],
        **{column: "" for column in REVIEW_COLUMNS[7:]},
    }


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite review sheet: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_review_packets(output_dir: Path, bases: list[dict[str, Any]]) -> dict[str, Any]:
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    protected_outputs = (
        output_dir / "blind_review_a.csv",
        output_dir / "blind_review_b.csv",
        output_dir / "adjudication_template.csv",
        private_dir / "blind_review_key.jsonl",
    )
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite an existing review packet: "
            + ", ".join(str(path) for path in existing)
        )
    salt_path = private_dir / "review_salt.txt"
    if salt_path.exists():
        salt = bytes.fromhex(salt_path.read_text(encoding="utf-8").strip())
    else:
        salt = secrets.token_bytes(32)
        salt_path.write_text(salt.hex() + "\n", encoding="utf-8", newline="\n")
    keyed = [
        (_blind_id(salt, base["base_id"]), base)
        for base in sorted(bases, key=lambda value: value["base_id"])
    ]
    dataset_hash = sha256(output_dir / "m2_2b_v7_heldout48_base.jsonl")
    key_rows = [
        {
            "blind_item_id": blind_id,
            "base_id": base["base_id"],
            "family_id": base["family_id"],
            "initial_answer": base["initial_answer"],
            "target_answer": base["target_answer"],
            "distractor_answer": base["distractor_answer"],
            "base_content_sha256": canonical_sha256(base),
            "dataset48_sha256": dataset_hash,
        }
        for blind_id, base in keyed
    ]
    write_jsonl(private_dir / "blind_review_key.jsonl", key_rows)
    for reviewer, reverse in (("a", False), ("b", True)):
        ordered = sorted(
            keyed,
            key=lambda item: hmac.new(
                salt,
                f"{reviewer}:{item[0]}".encode(),
                hashlib.sha256,
            ).hexdigest(),
            reverse=reverse,
        )
        _write_review_csv(
            output_dir / f"blind_review_{reviewer}.csv",
            [_review_row(base, blind_id) for blind_id, base in ordered],
        )
    adjudication = output_dir / "adjudication_template.csv"
    if adjudication.exists():
        raise RuntimeError(f"refusing to overwrite adjudication template: {adjudication}")
    with adjudication.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "blind_item_id",
                "disputed_field",
                "reviewer_a_raw_label",
                "reviewer_b_raw_label",
                "adjudicated_label",
                "adjudicator_rationale",
                "adjudication_timestamp",
            )
        )
    return {
        "review_salt_sha256": sha256(salt_path),
        "blind_key_sha256": sha256(private_dir / "blind_review_key.jsonl"),
        "dataset48_sha256": dataset_hash,
        "base_content_hashes_bound": 48,
        "review_a_sha256": sha256(output_dir / "blind_review_a.csv"),
        "review_b_sha256": sha256(output_dir / "blind_review_b.csv"),
        "adjudication_template_sha256": sha256(adjudication),
        "review_status": "not_started",
    }


def build(
    data_dir: Path,
    *,
    development_root: Path,
    prepare_review: bool,
) -> dict[str, Any]:
    cards_path = data_dir / "family_cards24.jsonl"
    bases_path = data_dir / "m2_2b_v7_heldout48_base.jsonl"
    cards = load_jsonl(cards_path)
    bases = load_jsonl(bases_path)
    family_report = validate_family_cards(cards, require_approved=True)
    overlap_report = validate_overlap_matrix(
        data_dir / "DEVELOPMENT_OVERLAP_MATRIX.md",
        require_approved=True,
    )
    overlap_audit_report = validate_overlap_audit(
        data_dir / "development_overlap_audit24.jsonl",
        cards,
        require_approved=True,
    )
    feasibility_report = validate_family_feasibility_audit(
        data_dir / "family_feasibility_audit24.jsonl",
        cards,
    )
    base_report = validate_base_cards(
        bases,
        cards,
        require_family_approval=True,
    )
    rows = expand_arms(bases)
    validate_flat_arms(rows, bases)
    arms_path = data_dir / "m2_2b_v7_heldout192_arms.jsonl"
    audit_path = data_dir / "construction_audit48.jsonl"
    leakage_path = data_dir / "leakage_report.json"
    write_jsonl(arms_path, rows)
    write_jsonl(audit_path, construction_audit_rows(bases))
    leakage = lexical_overlap_report(bases, development_root)
    write_json(leakage_path, leakage)
    review = prepare_review_packets(data_dir, bases) if prepare_review else {
        "review_status": "not_prepared"
    }
    hashed_names = (
        "family_cards24.jsonl",
        "FAMILY_BLUEPRINT.md",
        "DEVELOPMENT_OVERLAP_MATRIX.md",
        "development_overlap_audit24.jsonl",
        "m2_2b_v7_heldout48_base.jsonl",
        "m2_2b_v7_heldout192_arms.jsonl",
        "construction_audit48.jsonl",
        "leakage_report.json",
        "README.md",
        "build_m2_2b_v7_heldout.py",
        "validate_m2_2b_v7_heldout.py",
        "validate_phase2_sentinel4.py",
        "PHASE2_CLAUDE_CONSTRUCTION_BRIEF.md",
        "PHASE1_1_F17_LOGIC_REPAIR.md",
        "phase2_sentinel4_draft.jsonl",
        "PHASE2_SENTINEL_REVIEW.md",
        "CODEX_SENTINEL_APPROVAL.md",
        "family_feasibility_audit24.jsonl",
        "PHASE2_FAMILY_FEASIBILITY_AUDIT.md",
    )
    optional_hashed_names = (
        "author_m2_2b_v7_full48.py",
        "verify_phase2_full48_solutions.py",
        "write_phase2_construction_review.py",
        "PHASE2_CONSTRUCTION_REVIEW.md",
    )
    hashed_names = (
        *hashed_names,
        *(name for name in optional_hashed_names if (data_dir / name).is_file()),
    )
    if prepare_review:
        hashed_names = (
            *hashed_names,
            "blind_review_a.csv",
            "blind_review_b.csv",
            "adjudication_template.csv",
        )
    files = {
        name: {"sha256": sha256(data_dir / name), "bytes": (data_dir / name).stat().st_size}
        for name in hashed_names
    }
    manifest = {
        "status": "awaiting_independent_human_review",
        "dataset_version": VERSION,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "construction_method_audit": str(METHOD_AUDIT.relative_to(ROOT)),
        "construction_method_audit_sha256": sha256(METHOD_AUDIT),
        "execution_runbook": str(EXECUTION_RUNBOOK.relative_to(ROOT)),
        "execution_runbook_sha256": sha256(EXECUTION_RUNBOOK),
        "family_report": family_report,
        "base_report": base_report,
        "structural_overlap_review": overlap_report,
        "structural_overlap_audit": overlap_audit_report,
        "family_feasibility": feasibility_report,
        "arm_rows": len(rows),
        "leakage": leakage,
        "human_review": review,
        "qwen_or_gpu_execution": False,
        "model_outputs_exist": False,
        "gate_a_family_robustness": {
            "min_harmful_families": MIN_HARMFUL_FAMILIES,
            "max_single_family_harmful_share": MAX_SINGLE_FAMILY_HARMFUL_SHARE,
            "require_positive_leave_one_family_out": REQUIRE_POSITIVE_LOFO,
        },
        "files": files,
    }
    write_json(data_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE)
    parser.add_argument(
        "--development-root",
        type=Path,
        default=ROOT / "datasets" / "m2_2b_formal",
    )
    parser.add_argument("--prepare-review", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        args.data_dir,
        development_root=args.development_root,
        prepare_review=args.prepare_review,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
