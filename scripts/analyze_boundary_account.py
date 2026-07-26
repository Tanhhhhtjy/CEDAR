#!/usr/bin/env python3
"""Post-hoc, zero-GPU decision-boundary diagnostic for frozen CEDAR results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import subprocess
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs" / "BOUNDARY_ACCOUNT_ANALYSIS_PLAN.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "m2" / "boundary_account_posthoc_v1"
LABELS = tuple("ABCD")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_726
EPS = 1e-12


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8", newline="\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob(commit: str, path: str) -> bytes:
    command = ["git", "show", f"{commit}:{path}"]
    try:
        return subprocess.check_output(command, cwd=ROOT)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Cannot load frozen blob {commit}:{path}") from error


def load_json_blob(spec: Mapping[str, str]) -> Any:
    return json.loads(git_blob(spec["commit"], spec["path"]).decode("utf-8"))


def load_jsonl_blob(spec: Mapping[str, str]) -> list[dict[str, Any]]:
    text = git_blob(spec["commit"], spec["path"]).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def load_csv_blob(spec: Mapping[str, str]) -> list[dict[str, str]]:
    text = git_blob(spec["commit"], spec["path"]).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def stable_hash_fraction(text: str) -> float:
    integer = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return integer / float(16**16 - 1)


def percentile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    return float(np.quantile(np.asarray(finite), probability))


def ci95(values: Sequence[float]) -> list[float | None]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def summarize(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": mean(finite) if finite else None,
        "sd": stdev(finite) if len(finite) > 1 else None,
        "median": median(finite) if finite else None,
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
    }


def option_logits(row: Mapping[str, Any], field: str = "answer_logits") -> dict[str, float] | None:
    payload = row.get(field)
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("logits")
    if not isinstance(values, Mapping) or set(values) != set(LABELS):
        return None
    output = {label: float(values[label]) for label in LABELS}
    return output if all(math.isfinite(value) for value in output.values()) else None


def choice_from_row(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    return str(value) if value in LABELS else None


def option_geometry(
    values: Mapping[str, float] | None, initial: str | None, target: str | None
) -> dict[str, Any]:
    empty = {
        "winner": None,
        "runner_up": None,
        "g_top": None,
        "g_initial": None,
        "g_target": None,
        "m_general": None,
        "m_specific": None,
    }
    if values is None:
        return empty
    ordered = sorted(LABELS, key=lambda label: (-float(values[label]), label))
    winner, runner_up = ordered[:2]
    result = {
        "winner": winner,
        "runner_up": runner_up,
        "g_top": float(values[winner] - values[runner_up]),
        "g_initial": None,
        "g_target": None,
        "m_general": None,
        "m_specific": None,
    }
    if initial in LABELS:
        competitors = [float(values[label]) for label in LABELS if label != initial]
        result["g_initial"] = float(values[initial] - max(competitors))
        result["m_general"] = float(mean(competitors) - values[initial])
    if target in LABELS:
        result["g_target"] = float(values[winner] - values[target])
        if initial in LABELS and target != initial:
            background = [label for label in LABELS if label not in {initial, target}]
            result["m_specific"] = float(
                values[target] - mean(float(values[label]) for label in background)
            )
    return result


def classify_transition(
    uncapped: str | None, p25: str | None, initial: str | None, target: str | None
) -> str:
    if uncapped not in LABELS or p25 not in LABELS:
        return "invalid_or_missing"
    if uncapped == initial and p25 == initial:
        return "stable_initial"
    if uncapped == initial and p25 == target:
        return "initial_to_target"
    if uncapped == initial:
        return "initial_to_other"
    if uncapped == target and p25 == initial:
        return "target_to_initial"
    if uncapped == target and p25 != target:
        return "target_to_other"
    if uncapped == p25:
        return "other_transition"
    return "other_transition"


def add_outcomes(item: dict[str, Any]) -> None:
    uncapped = item["choice_uncapped"]
    p25 = item["choice_p25"]
    initial = item.get("initial_answer")
    target = item.get("target_answer")
    item["transition"] = classify_transition(uncapped, p25, initial, target)
    valid = uncapped in LABELS and p25 in LABELS
    item["any_flip"] = int(uncapped != p25) if valid else None
    item["initial_exit"] = (
        int(p25 != initial) if valid and uncapped == initial and initial in LABELS else None
    )
    item["target_entry"] = (
        int(p25 == target) if valid and uncapped != target and target in LABELS else None
    )


def family_slot(family_id: str) -> str:
    match = re.search(r"F(\d+)$", family_id)
    if not match:
        raise ValueError(f"Cannot extract structural family slot from {family_id}")
    return f"F{int(match.group(1)):02d}"


def base_item(
    *,
    dataset: str,
    readout: str,
    item_id: str,
    cluster_id: str,
    arm: str,
    initial: str | None,
    target: str | None,
    uncapped_choice: str | None,
    p25_choice: str | None,
    uncapped_logits: Mapping[str, float] | None,
    p25_logits: Mapping[str, float] | None,
) -> dict[str, Any]:
    uncapped_geometry = option_geometry(uncapped_logits, initial, target)
    p25_geometry = option_geometry(p25_logits, initial, target)
    delta_m_general = None
    if uncapped_geometry["m_general"] is not None and p25_geometry["m_general"] is not None:
        delta_m_general = float(
            p25_geometry["m_general"] - uncapped_geometry["m_general"]
        )
    target_minus_initial_uncapped = None
    target_minus_initial_p25 = None
    if uncapped_logits is not None and initial in uncapped_logits and target in uncapped_logits:
        target_minus_initial_uncapped = float(
            uncapped_logits[target] - uncapped_logits[initial]
        )
    if p25_logits is not None and initial in p25_logits and target in p25_logits:
        target_minus_initial_p25 = float(p25_logits[target] - p25_logits[initial])
    item = {
        "dataset": dataset,
        "readout": readout,
        "item_id": item_id,
        "stable_id": f"{dataset}::{readout}::{item_id}",
        "cluster_id": cluster_id,
        "arm": arm,
        "domain": None,
        "sample_seed": None,
        "variant": None,
        "initial_answer": initial,
        "target_answer": target,
        "choice_uncapped": uncapped_choice,
        "choice_p25": p25_choice,
        "logit_argmax_uncapped": uncapped_geometry["winner"],
        "logit_argmax_p25": p25_geometry["winner"],
        "g_top": uncapped_geometry["g_top"],
        "g_initial": uncapped_geometry["g_initial"],
        "g_target": uncapped_geometry["g_target"],
        "m_general_uncapped": uncapped_geometry["m_general"],
        "m_general_p25": p25_geometry["m_general"],
        "m_specific_uncapped": uncapped_geometry["m_specific"],
        "m_specific_p25": p25_geometry["m_specific"],
        "delta_m_general": delta_m_general,
        "target_minus_initial_uncapped": target_minus_initial_uncapped,
        "target_minus_initial_p25": target_minus_initial_p25,
        "delta_baseline_winner_advantage": None,
        "z_g_top": None,
        "z_g_initial": None,
        "z_g_target": None,
        "z_delta_m_general": None,
        "gap_quantile": None,
        "predictor_source": "uncapped option logits",
        "outcome_source": readout,
    }
    add_outcomes(item)
    return item


def pair_index(rows: Sequence[dict[str, Any]], condition_field: str = "condition") -> dict:
    index = {(str(row["id"]), str(row[condition_field])): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate id-condition cells")
    return index


def build_e2_items(
    rows: Sequence[dict[str, Any]], dataset: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset == "E2A":
        selected = [
            row
            for row in rows
            if row.get("assay") == "answer"
            and row.get("condition") in {"uncapped", "full_p25"}
        ]
    else:
        selected = [row for row in rows if row.get("condition") in {"uncapped", "full_p25"}]
    index = pair_index(selected)
    ids = sorted({str(row["id"]) for row in selected})
    items: list[dict[str, Any]] = []
    invariant_failures = []
    answer_position_failures = []
    for item_id in ids:
        uncapped = index[(item_id, "uncapped")]
        p25 = index[(item_id, "full_p25")]
        for field in (
            "base_id",
            "arm",
            "family_id",
            "initial_answer",
            "target_answer",
            "candidate_token_ids",
            "prompt_sha256",
            "prompt_token_ids_sha256",
        ):
            if uncapped.get(field) != p25.get(field):
                invariant_failures.append(f"{dataset}:{item_id}:{field}")
        for condition_row in (uncapped, p25):
            if option_logits(condition_row) is None:
                continue
            selected_index = condition_row.get("selected_answer_logit_index")
            capture_indices = condition_row.get("answer_logit_capture_indices", [])
            boundary = condition_row.get("answer_boundary", {})
            if (
                selected_index not in capture_indices
                or boundary.get("errors")
                or boundary.get("answer_choice") != condition_row.get("chosen_option")
            ):
                answer_position_failures.append(
                    f"{dataset}:{item_id}:{condition_row['condition']}"
                )
        item = base_item(
            dataset=dataset,
            readout="thinking_generation",
            item_id=item_id,
            cluster_id=str(uncapped["family_id"]),
            arm=str(uncapped["arm"]),
            initial=choice_from_row(uncapped, "initial_answer"),
            target=choice_from_row(uncapped, "target_answer"),
            uncapped_choice=choice_from_row(uncapped, "chosen_option"),
            p25_choice=choice_from_row(p25, "chosen_option"),
            uncapped_logits=option_logits(uncapped),
            p25_logits=option_logits(p25),
        )
        item.update(
            {
                "base_id": str(uncapped["base_id"]),
                "family_slot": family_slot(str(uncapped["family_id"])),
                "structure_class": str(uncapped["structure_class"]),
                "variant": uncapped.get("variant"),
                "stratum": f"{dataset}|{uncapped['arm']}",
                "standardization_group": f"{dataset}|{uncapped['arm']}|thinking_generation",
                "pooled_resample_source": "E2",
                "pooled_cluster_id": str(uncapped["structure_class"]),
                "source_condition_uncapped": "uncapped",
                "source_condition_p25": "full_p25",
                "uncapped_truncated": bool(uncapped.get("truncated")),
                "p25_truncated": bool(p25.get("truncated")),
                "candidate_token_ids": uncapped.get("candidate_token_ids"),
                "uncapped_parse_status": uncapped.get("final_answer_parse_status"),
                "p25_parse_status": p25.get("final_answer_parse_status"),
            }
        )
        items.append(item)
    audit = {
        "selected_rows": len(selected),
        "pairs": len(items),
        "invariant_failures": invariant_failures,
        "answer_position_failures": answer_position_failures,
        "content_retries": sum(int(row.get("content_retry_count", 0)) for row in selected),
        "answer_repairs": sum(bool(row.get("answer_repair_performed")) for row in selected),
        "missing_uncapped_logits": [
            item["item_id"] for item in items if item["g_top"] is None
        ],
        "missing_p25_logits": [
            item["item_id"] for item in items if item["m_general_p25"] is None
        ],
    }
    return items, audit


def build_c2_items(
    direct_uncapped: Sequence[dict[str, Any]],
    direct_p25: Sequence[dict[str, Any]],
    free_uncapped: Sequence[dict[str, Any]],
    free_p25: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maps = {
        name: {str(row["id"]): row for row in rows}
        for name, rows in {
            "direct_uncapped": direct_uncapped,
            "direct_p25": direct_p25,
            "free_uncapped": free_uncapped,
            "free_p25": free_p25,
        }.items()
    }
    if any(len(mapping) != len(rows) for mapping, rows in zip(maps.values(), (
        direct_uncapped, direct_p25, free_uncapped, free_p25
    ), strict=True)):
        raise ValueError("Duplicate C2 IDs")
    shared = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    items: list[dict[str, Any]] = []
    invariant_failures = []
    direct_answer_position_failures = []
    for item_id in shared:
        du = maps["direct_uncapped"][item_id]
        dp = maps["direct_p25"][item_id]
        fu = maps["free_uncapped"][item_id]
        fp = maps["free_p25"][item_id]
        quartet = (du, dp, fu, fp)
        for field in (
            "base_id",
            "arm",
            "domain",
            "normalized_stem_id",
            "initial_answer",
            "target_answer",
            "candidate_token_ids",
        ):
            if len({canonical_json(row.get(field)) for row in quartet}) != 1:
                invariant_failures.append(f"C2:{item_id}:{field}")
        if du.get("answer_step_tokens") != 1 or dp.get("answer_step_tokens") != 1:
            direct_answer_position_failures.append(f"C2:{item_id}:answer_step_tokens")
        item = base_item(
            dataset="C2",
            readout="free_generation",
            item_id=item_id,
            cluster_id=str(du["normalized_stem_id"]),
            arm=str(du["arm"]),
            initial=choice_from_row(du, "initial_answer"),
            target=choice_from_row(du, "target_answer"),
            uncapped_choice=choice_from_row(fu, "strict_answer"),
            p25_choice=choice_from_row(fp, "strict_answer"),
            uncapped_logits=option_logits(du),
            p25_logits=option_logits(dp),
        )
        item.update(
            {
                "base_id": str(du["base_id"]),
                "family_slot": None,
                "domain": str(du["domain"]),
                "variant": du.get("difficulty"),
                "stratum": f"C2|{du['arm']}|{du['domain']}",
                "standardization_group": f"C2|{du['arm']}|free_generation",
                "pooled_resample_source": "C2",
                "pooled_cluster_id": str(du["normalized_stem_id"]),
                "predictor_source": "matched direct uncapped option logits",
                "outcome_source": "strict free generation",
                "direct_choice_uncapped": option_geometry(
                    option_logits(du),
                    choice_from_row(du, "initial_answer"),
                    choice_from_row(du, "target_answer"),
                )["winner"],
                "direct_choice_p25": option_geometry(
                    option_logits(dp),
                    choice_from_row(dp, "initial_answer"),
                    choice_from_row(dp, "target_answer"),
                )["winner"],
                "direct_any_flip": int(
                    option_geometry(option_logits(du), None, None)["winner"]
                    != option_geometry(option_logits(dp), None, None)["winner"]
                ),
                "candidate_token_ids": du.get("candidate_token_ids"),
                "uncapped_parse_status": (
                    "passed" if choice_from_row(fu, "strict_answer") else "failed"
                ),
                "p25_parse_status": (
                    "passed" if choice_from_row(fp, "strict_answer") else "failed"
                ),
                "uncapped_truncated": bool(fu.get("truncated")),
                "p25_truncated": bool(fp.get("truncated")),
            }
        )
        items.append(item)
    audit = {
        "raw_counts": {name: len(mapping) for name, mapping in maps.items()},
        "shared_ids": len(shared),
        "invariant_failures": invariant_failures,
        "direct_answer_position_failures": direct_answer_position_failures,
        "direct_free_uncapped_choice_agreement": {
            "agree": sum(
                item["direct_choice_uncapped"] == item["choice_uncapped"] for item in items
            ),
            "n": len(items),
        },
    }
    return items, audit


def build_e_minus1b_items(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = {
        (str(row["pair_id"]), str(row["variant"]), str(row["condition"])): row
        for row in rows
    }
    if len(index) != len(rows):
        raise ValueError("Duplicate E-1b pair-variant-condition cells")
    keys = sorted({(str(row["pair_id"]), str(row["variant"])) for row in rows})
    items: list[dict[str, Any]] = []
    invariant_failures = []
    for pair_id, variant in keys:
        uncapped = index[(pair_id, variant, "uncapped_monitor")]
        p25 = index[(pair_id, variant, "p25_cap")]
        for field in (
            "pair_id",
            "orig_id",
            "sample_seed",
            "variant",
            "A_token_id",
            "B_token_id",
            "sycophantic_answer",
            "non_sycophantic_answer",
            "prompt_sha256",
        ):
            if uncapped.get(field) != p25.get(field):
                invariant_failures.append(f"E-1b:{pair_id}:{variant}:{field}")
        uncapped_logits = {"A": float(uncapped["A_logit"]), "B": float(uncapped["B_logit"])}
        p25_logits = {"A": float(p25["A_logit"]), "B": float(p25["B_logit"])}
        ordered = sorted(("A", "B"), key=lambda label: (-uncapped_logits[label], label))
        winner, other = ordered
        g_top = float(uncapped_logits[winner] - uncapped_logits[other])
        p25_oriented_gap = float(p25_logits[winner] - p25_logits[other])
        target = choice_from_row(uncapped, "sycophantic_answer")
        g_target = float(uncapped_logits[winner] - uncapped_logits[target]) if target else None
        item = {
            "dataset": "E-1b",
            "readout": "forced_choice",
            "item_id": f"{pair_id}::{variant}",
            "stable_id": f"E-1b::forced_choice::{pair_id}::{variant}",
            "cluster_id": pair_id,
            "arm": variant,
            "domain": None,
            "sample_seed": int(uncapped["sample_seed"]),
            "variant": variant,
            "base_id": str(uncapped["base_id"]),
            "family_slot": None,
            "initial_answer": None,
            "target_answer": target,
            "choice_uncapped": choice_from_row(uncapped, "predicted_forced_choice"),
            "choice_p25": choice_from_row(p25, "predicted_forced_choice"),
            "logit_argmax_uncapped": winner,
            "logit_argmax_p25": max(("A", "B"), key=lambda label: p25_logits[label]),
            "g_top": g_top,
            "g_initial": None,
            "g_target": g_target,
            "m_general_uncapped": None,
            "m_general_p25": None,
            "m_specific_uncapped": None,
            "m_specific_p25": None,
            "delta_m_general": None,
            "delta_baseline_winner_advantage": p25_oriented_gap - g_top,
            "z_g_top": None,
            "z_g_initial": None,
            "z_g_target": None,
            "z_delta_m_general": None,
            "gap_quantile": None,
            "stratum": f"E-1b|seed{uncapped['sample_seed']}|{variant}",
            "standardization_group": f"E-1b|{variant}|forced_choice",
            "pooled_resample_source": "E-1b",
            "pooled_cluster_id": pair_id,
            "predictor_source": "uncapped A/B logits",
            "outcome_source": "forced-choice argmax",
            "candidate_token_ids": {
                "A": int(uncapped["A_token_id"]),
                "B": int(uncapped["B_token_id"]),
            },
            "uncapped_full_vocab_entropy": float(uncapped["full_vocab_entropy"]),
            "p25_full_vocab_entropy": float(p25["full_vocab_entropy"]),
            "uncapped_option_mass": float(uncapped["AB_probability_mass"]),
            "p25_option_mass": float(p25["AB_probability_mass"]),
            "uncapped_top_token_is_option": bool(uncapped["full_vocab_top1_is_AB"]),
            "p25_top_token_is_option": bool(p25["full_vocab_top1_is_AB"]),
            "uncapped_truncated": bool(uncapped.get("prompt_truncated")),
            "p25_truncated": bool(p25.get("prompt_truncated")),
            "uncapped_parse_status": "forced_choice",
            "p25_parse_status": "forced_choice",
        }
        add_outcomes(item)
        items.append(item)
    audit = {
        "rows": len(rows),
        "pair_variant_cells": len(items),
        "pair_clusters": len({item["cluster_id"] for item in items}),
        "variants": dict(Counter(item["variant"] for item in items)),
        "sample_seeds": dict(Counter(str(item["sample_seed"]) for item in items)),
        "invariant_failures": invariant_failures,
    }
    return items, audit


def manifest_checks(plan: Mapping[str, Any]) -> dict[str, Any]:
    inputs = plan["inputs"]
    e2a_runtime = json.loads(
        git_blob(
            inputs["e2a_results"]["commit"],
            "artifacts/m2/e2_phase_validity_choice/formal-all-seed0/runtime.json",
        ).decode("utf-8")
    )
    e2c_delivery = json.loads(
        git_blob(
            inputs["e2c_results"]["commit"],
            "artifacts/m2/e2_new_salt_confirmation/DELIVERY_MANIFEST.json",
        ).decode("utf-8")
    )
    c2_delivery = json.loads(
        git_blob(
            inputs["c2_direct_uncapped"]["commit"],
            "artifacts/m2/c2_v8_natural_social_confirmation/delivery_manifest.json",
        ).decode("utf-8")
    )
    e1_runtime = json.loads(
        git_blob(
            inputs["e_minus1b_rows"]["commit"],
            "artifacts/m2/e_minus1b_philpapers_cap_audit/runtime_manifest.json",
        ).decode("utf-8")
    )
    c2_runtimes = {
        name: json.loads(
            git_blob(
                inputs["c2_direct_uncapped"]["commit"],
                f"artifacts/m2/c2_v8_natural_social_confirmation/{name}/runtime.json",
            ).decode("utf-8")
        )
        for name in ("direct-uncapped", "direct-p25", "free-uncapped", "free-p25")
    }
    c2_names = {
        "c2_direct_uncapped": "direct-uncapped/results.jsonl",
        "c2_direct_p25": "direct-p25/results.jsonl",
        "c2_free_uncapped": "free-uncapped/results.jsonl",
        "c2_free_p25": "free-p25/results.jsonl",
    }
    checks = {
        "e2a_runtime_results_hash": (
            e2a_runtime.get("results_sha256") == inputs["e2a_results"]["sha256"]
        ),
        "e2c_delivery_results_hash": (
            e2c_delivery["files"].get(
                "formal-paired-uncapped-full-p25-seed0/results.jsonl"
            )
            == inputs["e2c_results"]["sha256"]
        ),
        "e2c_delivery_analysis_hash": (
            e2c_delivery["files"].get("E2_NEW_SALT_ANALYSIS.json")
            == inputs["e2c_analysis"]["sha256"]
        ),
        "e_minus1b_runtime_results_hash": (
            e1_runtime.get("row_results_sha256") == inputs["e_minus1b_rows"]["sha256"]
        ),
        "e_minus1b_runtime_zero_content_retries": e1_runtime.get("content_retries") == 0,
        "c2_runtime_zero_content_retries": all(
            runtime.get("content_retries") == 0 for runtime in c2_runtimes.values()
        ),
    }
    for key, delivery_name in c2_names.items():
        checks[f"{key}_delivery_hash"] = (
            c2_delivery["files"][delivery_name]["canonical_text_sha256"]
            == inputs[key]["sha256"]
        )
    return checks


def audit_inputs(
    plan: Mapping[str, Any],
    datasets: Mapping[str, Sequence[dict[str, Any]]],
    build_audits: Mapping[str, Any],
    c2_addendum: Sequence[dict[str, str]],
) -> dict[str, Any]:
    blob_audit = {}
    for name, spec in plan["inputs"].items():
        payload = git_blob(spec["commit"], spec["path"])
        observed = sha256_bytes(payload)
        blob_audit[name] = {
            "commit": spec["commit"],
            "path": spec["path"],
            "expected_sha256": spec["sha256"],
            "observed_sha256": observed,
            "bytes": len(payload),
            "passed": observed == spec["sha256"],
        }
    e2a = list(datasets["E2A"])
    e2c = list(datasets["E2C"])
    c2 = list(datasets["C2"])
    e1 = list(datasets["E-1b"])
    addendum_by_base = {row["base_id"]: row for row in c2_addendum}
    c2_crosscheck_failures = []
    for item in c2:
        addendum = addendum_by_base.get(str(item["base_id"]))
        arm = item["arm"]
        if addendum is None:
            c2_crosscheck_failures.append(f"{item['item_id']}:missing_addendum_base")
            continue
        comparisons = {
            f"margin_{arm}_uncapped": item["target_minus_initial_uncapped"],
            f"margin_{arm}_p25": item["target_minus_initial_p25"],
        }
        for field, expected in comparisons.items():
            if expected is None or not math.isclose(
                float(addendum[field]), float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                c2_crosscheck_failures.append(f"{item['item_id']}:{field}")
    expected_missing = [item for item in e2c if item["g_top"] is None]
    checks = {
        "all_source_blobs_match_frozen_sha256": all(
            item["passed"] for item in blob_audit.values()
        ),
        "available_manifest_hashes_match": all(manifest_checks(plan).values()),
        "e2a_answer_pairs_96": len(e2a) == 96,
        "e2a_pair_invariants": not build_audits["E2A"]["invariant_failures"],
        "e2a_answer_position_consistent": not build_audits["E2A"][
            "answer_position_failures"
        ],
        "e2a_no_missing_logits": not build_audits["E2A"]["missing_uncapped_logits"]
        and not build_audits["E2A"]["missing_p25_logits"],
        "e2c_pairs_96": len(e2c) == 96,
        "e2c_pair_invariants": not build_audits["E2C"]["invariant_failures"],
        "e2c_answer_position_consistent_when_present": not build_audits["E2C"][
            "answer_position_failures"
        ],
        "e2c_exact_expected_missing": len(expected_missing) == 1
        and expected_missing[0]["item_id"] == "CEDAR-E2C-022-F"
        and expected_missing[0]["uncapped_truncated"],
        "c2_cross_readout_pairs_480": len(c2) == 480,
        "c2_pair_invariants": not build_audits["C2"]["invariant_failures"],
        "c2_direct_answer_position_consistent": not build_audits["C2"][
            "direct_answer_position_failures"
        ],
        "c2_addendum_crosscheck_rows_600": len(c2_addendum) == 600,
        "c2_addendum_values_match_raw_direct_records": not c2_crosscheck_failures,
        "e_minus1b_pair_variant_rows_900": len(e1) == 900,
        "e_minus1b_pair_invariants": not build_audits["E-1b"]["invariant_failures"],
        "no_content_retry_or_answer_repair": (
            build_audits["E2A"]["content_retries"] == 0
            and build_audits["E2A"]["answer_repairs"] == 0
            and build_audits["E2C"]["content_retries"] == 0
            and build_audits["E2C"]["answer_repairs"] == 0
            and manifest_checks(plan)["c2_runtime_zero_content_retries"]
            and manifest_checks(plan)["e_minus1b_runtime_zero_content_retries"]
        ),
        "readout_modes_separate": {item["readout"] for item in e2a + e2c + c2 + e1}
        == {"thinking_generation", "free_generation", "forced_choice"},
    }
    agreement = {}
    for dataset, items in datasets.items():
        eligible = [item for item in items if item["logit_argmax_uncapped"] is not None]
        agreement[dataset] = {
            "uncapped_argmax_matches_discrete": sum(
                item["logit_argmax_uncapped"] == item["choice_uncapped"]
                for item in eligible
            ),
            "eligible": len(eligible),
            "readout": eligible[0]["readout"] if eligible else None,
        }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "cedar.boundary_account_posthoc.input_audit.v1",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "source_blobs": blob_audit,
        "manifest_checks": manifest_checks(plan),
        "build_audits": build_audits,
        "c2_addendum_crosscheck_failures": c2_crosscheck_failures,
        "output_argmax_agreement": agreement,
        "readout_warning": (
            "C2 direct logits and free-generation choices are distinct readouts; "
            "their agreement is diagnostic, not an audit requirement."
        ),
    }


def standardize_field(
    items: Sequence[dict[str, Any]], source_field: str, target_field: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        value = item.get(source_field)
        if value is not None and math.isfinite(float(value)):
            grouped[str(item["standardization_group"])].append(item)
    metadata = {}
    for group, rows in sorted(grouped.items()):
        values = [float(row[source_field]) for row in rows]
        center = mean(values)
        scale = stdev(values) if len(values) > 1 else 0.0
        metadata[group] = {"n": len(values), "mean": center, "sd": scale}
        for row in rows:
            row[target_field] = (float(row[source_field]) - center) / scale if scale else 0.0
    return metadata


def assign_gap_quantiles(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("g_top") is not None and item.get("any_flip") is not None:
            grouped[f"{item['dataset']}|{item['readout']}"] .append(item)
    metadata = {}
    for group, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (float(row["g_top"]), row["stable_id"]))
        bins = 4 if len(ordered) >= 40 else 2
        for index, row in enumerate(ordered):
            row["gap_quantile"] = min(bins, math.floor(index * bins / len(ordered)) + 1)
        metadata[group] = {
            "n": len(ordered),
            "bins": bins,
            "tie_break": "stable_id",
            "basis": "uncapped_g_top_only",
        }
    return metadata


def _design_matrix(
    rows: Sequence[Mapping[str, Any]], outcome: str, predictors: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]] | None:
    eligible = [
        row
        for row in rows
        if row.get(outcome) in {0, 1}
        and all(row.get(predictor) is not None for predictor in predictors)
    ]
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_stratum[str(row["stratum"])].append(row)
    varying = {
        stratum
        for stratum, stratum_rows in by_stratum.items()
        if len({int(row[outcome]) for row in stratum_rows}) == 2
    }
    eligible = [row for row in eligible if str(row["stratum"]) in varying]
    if not eligible or len({int(row[outcome]) for row in eligible}) < 2:
        return None
    strata = sorted(varying)
    columns = [np.ones(len(eligible), dtype=float)]
    names = ["intercept"]
    for predictor in predictors:
        columns.append(np.asarray([float(row[predictor]) for row in eligible], dtype=float))
        names.append(predictor)
    for stratum in strata[1:]:
        columns.append(
            np.asarray([float(str(row["stratum"]) == stratum) for row in eligible])
        )
        names.append(f"stratum:{stratum}")
    matrix = np.column_stack(columns)
    response = np.asarray([int(row[outcome]) for row in eligible], dtype=float)
    metadata = {
        "eligible_rows": len(eligible),
        "events": int(response.sum()),
        "non_events": int(len(response) - response.sum()),
        "varying_strata": strata,
        "dropped_zero_event_or_all_event_strata": sorted(set(by_stratum) - varying),
    }
    return matrix, response, names, metadata


def _expit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_fixed_effect_logit(
    rows: Sequence[Mapping[str, Any]], outcome: str, predictors: Sequence[str]
) -> dict[str, Any]:
    design = _design_matrix(rows, outcome, predictors)
    if design is None:
        return {"status": "not_estimable", "reason": "no varying fixed-effect strata"}
    matrix, response, names, metadata = design
    beta = np.zeros(matrix.shape[1], dtype=float)
    converged = False
    for _ in range(100):
        fitted = _expit(matrix @ beta)
        weights = np.maximum(fitted * (1.0 - fitted), 1e-9)
        gradient = matrix.T @ (response - fitted)
        hessian = matrix.T @ (weights[:, None] * matrix)
        try:
            step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-10, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(
                hessian + np.eye(hessian.shape[0]) * 1e-8, gradient, rcond=None
            )[0]
        beta += step
        if float(np.max(np.abs(step))) < 1e-8:
            converged = True
            break
        if not np.all(np.isfinite(beta)) or float(np.max(np.abs(beta))) > 50:
            return {"status": "not_estimable", "reason": "complete_or_quasi_separation"}
    if not converged:
        return {"status": "not_estimable", "reason": "IRLS did not converge"}
    coefficients = {name: float(beta[index]) for index, name in enumerate(names)}
    return {"status": "estimated", "coefficients": coefficients, **metadata}


def roc_auc(outcomes: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for outcome, score in zip(outcomes, scores, strict=True) if outcome == 1]
    negatives = [score for outcome, score in zip(outcomes, scores, strict=True) if outcome == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def descriptive_boundary(
    rows: Sequence[Mapping[str, Any]], gap_field: str, *, include_quantiles: bool = True
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("any_flip") in {0, 1} and row.get(gap_field) is not None
    ]
    flips = [float(row[gap_field]) for row in eligible if row["any_flip"] == 1]
    stable = [float(row[gap_field]) for row in eligible if row["any_flip"] == 0]
    quantiles = (
        sorted({int(row["gap_quantile"]) for row in eligible}) if include_quantiles else []
    )
    q_counts = {
        f"Q{quantile}": {
            "n": sum(int(row["gap_quantile"]) == quantile for row in eligible),
            "flips": sum(
                int(row["gap_quantile"]) == quantile and row["any_flip"] == 1
                for row in eligible
            ),
        }
        for quantile in quantiles
    }
    for value in q_counts.values():
        value["flip_rate"] = value["flips"] / value["n"] if value["n"] else None
    auc = roc_auc(
        [int(row["any_flip"]) for row in eligible],
        [-float(row[gap_field]) for row in eligible],
    )
    return {
        "n": len(eligible),
        "flips": len(flips),
        "stable": len(stable),
        "flip_rate": len(flips) / len(eligible) if eligible else None,
        "gap_flip": summarize(flips),
        "gap_stable": summarize(stable),
        "median_flip_minus_stable": (
            median(flips) - median(stable) if flips and stable else None
        ),
        "auc_negative_gap": auc,
        "quantiles": q_counts,
        "nearest_minus_farthest_flip_rate": (
            q_counts[f"Q{min(quantiles)}"]["flip_rate"]
            - q_counts[f"Q{max(quantiles)}"]["flip_rate"]
            if quantiles
            else None
        ),
    }


def resample_clusters(
    rows: Sequence[Mapping[str, Any]], rng: random.Random, *, pooled: bool
) -> list[Mapping[str, Any]]:
    grouped_sources: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        source = str(row["pooled_resample_source"] if pooled else row["dataset"])
        cluster = str(row["pooled_cluster_id"] if pooled else row["cluster_id"])
        grouped_sources[source][cluster].append(row)
    output: list[Mapping[str, Any]] = []
    for source in sorted(grouped_sources):
        clusters = sorted(grouped_sources[source])
        for _ in clusters:
            selected = clusters[rng.randrange(len(clusters))]
            output.extend(grouped_sources[source][selected])
    return output


def _bootstrap_metrics(
    rows: Sequence[Mapping[str, Any]], gap_field: str, *, include_quantiles: bool
) -> tuple[float | None, float | None, float | None]:
    eligible = [
        row
        for row in rows
        if row.get("any_flip") in {0, 1} and row.get(gap_field) is not None
    ]
    flips = [float(row[gap_field]) for row in eligible if row["any_flip"] == 1]
    stable = [float(row[gap_field]) for row in eligible if row["any_flip"] == 0]
    auc = roc_auc(
        [int(row["any_flip"]) for row in eligible],
        [-float(row[gap_field]) for row in eligible],
    )
    median_difference = median(flips) - median(stable) if flips and stable else None
    if not eligible or not include_quantiles:
        return auc, median_difference, None
    quantiles = sorted({int(row["gap_quantile"]) for row in eligible})
    nearest = [row for row in eligible if int(row["gap_quantile"]) == min(quantiles)]
    farthest = [row for row in eligible if int(row["gap_quantile"]) == max(quantiles)]
    q_difference = (
        mean(int(row["any_flip"]) for row in nearest)
        - mean(int(row["any_flip"]) for row in farthest)
        if nearest and farthest
        else None
    )
    return auc, median_difference, q_difference


def bootstrap_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    outcome: str,
    predictors: Sequence[str],
    gap_field: str,
    draws: int,
    seed: int,
    pooled: bool,
    include_quantiles: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    point = fit_fixed_effect_logit(rows, outcome, predictors)
    rng = random.Random(seed)
    records = []
    coefficient_draws: dict[str, list[float]] = {predictor: [] for predictor in predictors}
    auc_draws: list[float] = []
    median_draws: list[float] = []
    quantile_draws: list[float] = []
    for draw in range(draws):
        sampled = resample_clusters(rows, rng, pooled=pooled)
        fitted = fit_fixed_effect_logit(sampled, outcome, predictors)
        coefficients = fitted.get("coefficients", {}) if fitted["status"] == "estimated" else {}
        auc, median_difference, quantile_difference = _bootstrap_metrics(
            sampled, gap_field, include_quantiles=include_quantiles
        )
        record = {
            "scope": scope,
            "draw": draw,
            "fit_status": fitted["status"],
            "auc_negative_gap": auc,
            "median_flip_minus_stable": median_difference,
            "nearest_minus_farthest_flip_rate": quantile_difference,
        }
        for predictor in predictors:
            value = coefficients.get(predictor)
            record[f"coefficient::{predictor}"] = value
            if value is not None and math.isfinite(value):
                coefficient_draws[predictor].append(float(value))
        if auc is not None and math.isfinite(auc):
            auc_draws.append(float(auc))
        if median_difference is not None and math.isfinite(median_difference):
            median_draws.append(float(median_difference))
        if quantile_difference is not None and math.isfinite(quantile_difference):
            quantile_draws.append(float(quantile_difference))
        records.append(record)
    descriptive = descriptive_boundary(
        rows, gap_field, include_quantiles=include_quantiles
    )
    coefficient_summary = {}
    for predictor in predictors:
        point_value = (
            point.get("coefficients", {}).get(predictor)
            if point.get("status") == "estimated"
            else None
        )
        values = coefficient_draws[predictor]
        coefficient_summary[predictor] = {
            "estimate": point_value,
            "bootstrap_95ci": ci95(values),
            "valid_draws": len(values),
            "requested_draws": draws,
            "odds_ratio": math.exp(point_value) if point_value is not None else None,
            "odds_ratio_95ci": (
                [math.exp(value) if value is not None else None for value in ci95(values)]
                if values
                else [None, None]
            ),
            "bootstrap_two_sided_sign_p": (
                min(
                    1.0,
                    2
                    * min(
                        sum(value <= 0 for value in values) / len(values),
                        sum(value >= 0 for value in values) / len(values),
                    ),
                )
                if values
                else None
            ),
        }
    low_events = descriptive["flips"] < 5
    result = {
        "scope": scope,
        "outcome": outcome,
        "predictors": list(predictors),
        "point_fit": point,
        "coefficients": coefficient_summary,
        "descriptive": descriptive,
        "auc_negative_gap_bootstrap_95ci": ci95(auc_draws),
        "median_flip_minus_stable_bootstrap_95ci": ci95(median_draws),
        "nearest_minus_farthest_bootstrap_95ci": ci95(quantile_draws),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "pooled_hierarchical_resampling": pooled,
            "valid_auc_draws": len(auc_draws),
        },
        "inferential_status": (
            "descriptive_only_fewer_than_5_flips" if low_events else "cluster_bootstrap"
        ),
    }
    return result, records


def holm_adjust(raw: Mapping[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((name, float(value)) for name, value in raw.items() if value is not None),
        key=lambda pair: pair[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in raw}
    running = 0.0
    count = len(available)
    for rank, (name, value) in enumerate(available):
        candidate = min(1.0, value * (count - rank))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def leave_one_cluster_out(
    rows: Sequence[Mapping[str, Any]], outcome: str, predictor: str
) -> dict[str, Any]:
    clusters = sorted(
        {
            f"{row['pooled_resample_source']}::{row['pooled_cluster_id']}"
            for row in rows
        }
    )
    estimates = {}
    for cluster in clusters:
        kept = [
            row
            for row in rows
            if f"{row['pooled_resample_source']}::{row['pooled_cluster_id']}" != cluster
        ]
        fitted = fit_fixed_effect_logit(kept, outcome, [predictor])
        estimates[cluster] = (
            fitted.get("coefficients", {}).get(predictor)
            if fitted.get("status") == "estimated"
            else None
        )
    finite = [float(value) for value in estimates.values() if value is not None]
    return {
        "clusters": len(clusters),
        "estimable": len(finite),
        "negative": sum(value < 0 for value in finite),
        "positive": sum(value > 0 for value in finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "all_estimable_directions_negative": bool(finite)
        and len(finite) == len(clusters)
        and all(value < 0 for value in finite),
        "estimates": estimates,
    }


def leave_one_source_out(
    rows: Sequence[Mapping[str, Any]], outcome: str, predictor: str
) -> dict[str, Any]:
    estimates = {}
    for omitted in sorted({str(row["dataset"]) for row in rows}):
        fitted = fit_fixed_effect_logit(
            [row for row in rows if row["dataset"] != omitted], outcome, [predictor]
        )
        estimates[omitted] = {
            "status": fitted["status"],
            "estimate": (
                fitted.get("coefficients", {}).get(predictor)
                if fitted.get("status") == "estimated"
                else None
            ),
        }
    return estimates


def transition_counts(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (item["dataset"], item["readout"], item["arm"], item["transition"])
        for item in items
    )
    return [
        {
            "dataset": key[0],
            "readout": key[1],
            "arm": key[2],
            "transition": key[3],
            "count": value,
        }
        for key, value in sorted(counts.items())
    ]


def quantile_bootstrap_summary(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    point = descriptive_boundary(rows, "g_top")["quantiles"]
    by_quantile: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(draws):
        sampled = resample_clusters(rows, rng, pooled=False)
        eligible = [
            row
            for row in sampled
            if row.get("any_flip") in {0, 1} and row.get("gap_quantile") is not None
        ]
        for quantile in sorted({int(row["gap_quantile"]) for row in eligible}):
            selected = [row for row in eligible if int(row["gap_quantile"]) == quantile]
            if selected:
                by_quantile[f"Q{quantile}"].append(
                    mean(int(row["any_flip"]) for row in selected)
                )
    return {
        quantile: {**values, "cluster_bootstrap_95ci": ci95(by_quantile[quantile])}
        for quantile, values in point.items()
    }


def source_rows(items: Sequence[Mapping[str, Any]], dataset: str) -> list[Mapping[str, Any]]:
    return [item for item in items if item["dataset"] == dataset]


def build_gate(
    rq1: Mapping[str, Mapping[str, Any]],
    loco: Mapping[str, Any],
    loso: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pooled = rq1["Pooled"]["coefficients"]["z_g_top"]
    source_names = ("E2A", "E2C", "C2", "E-1b")
    source_coefficients = {
        name: rq1[name]["coefficients"]["z_g_top"] for name in source_names
    }
    negative_sources = sum(
        value["estimate"] is not None and value["estimate"] < 0
        for value in source_coefficients.values()
    )
    reverse_supported = [
        name
        for name, value in source_coefficients.items()
        if value["bootstrap_95ci"][0] is not None and value["bootstrap_95ci"][0] > 0
    ]
    c2 = rq1["C2"]
    c2_same_direction = (
        c2["coefficients"]["z_g_top"]["estimate"] is not None
        and c2["coefficients"]["z_g_top"]["estimate"] < 0
        and c2["descriptive"]["auc_negative_gap"] is not None
        and c2["descriptive"]["auc_negative_gap"] > 0.5
    )
    conditions = {
        "pooled_slope_ci_below_zero": pooled["bootstrap_95ci"][1] is not None
        and pooled["bootstrap_95ci"][1] < 0,
        "at_least_three_sources_negative": negative_sources >= 3,
        "no_source_ci_fully_supports_reverse": not reverse_supported,
        "nearest_flip_rate_exceeds_farthest": (
            rq1["Pooled"]["descriptive"]["nearest_minus_farthest_flip_rate"] is not None
            and rq1["Pooled"]["descriptive"]["nearest_minus_farthest_flip_rate"] > 0
        ),
        "c2_cross_readout_directionally_concordant": c2_same_direction,
        "leave_one_cluster_out_direction_stable": loco[
            "all_estimable_directions_negative"
        ],
    }
    if all(conditions.values()):
        gate = "Gate A"
        allowed = (
            "Across assays, intervention-induced flips concentrated among low-margin "
            "instances. This supports a boundary account in which a stable continuous "
            "displacement yields heterogeneous discrete effects depending on an "
            "instance's pre-intervention distance from the decision boundary."
        )
    elif pooled["estimate"] is not None and pooled["estimate"] < 0:
        gate = "Gate B"
        allowed = (
            "The results are consistent with a decision-boundary account, but the "
            "post-hoc evidence is not conclusive."
        )
    else:
        gate = "Gate C"
        allowed = "Continuous and discrete effects need not coincide."
    return {
        "schema": "cedar.boundary_account_posthoc.claim_gate.v1",
        "gate": gate,
        "conditions": conditions,
        "negative_source_point_estimates": negative_sources,
        "source_cis_supporting_reverse": reverse_supported,
        "leave_one_source_out_point_estimates": loso,
        "pooled_without_e_minus1b_slope": loso["E-1b"]["estimate"],
        "reconciles_wording_allowed": gate == "Gate A",
        "consistent_with_wording_allowed": gate in {"Gate A", "Gate B"},
        "allowed_wording": allowed,
        "forbidden_wording": [
            "We proved the internal mechanism of capping.",
            "The boundary account was preregistered or independently confirmed.",
            "Delta m is a causal mediator of answer changes.",
            (
                "C2 provides strong cross-readout evidence when its free-generation "
                "event count is low."
            ),
        ],
        "post_hoc": True,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_figure(fig: Any, base_path: Path) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pubfig as pf

        pf.batch_export(
            fig,
            str(base_path),
            formats=("pdf", "png"),
            spec="science",
            width="double",
            dpi=600,
        )
    except (ImportError, TypeError, ValueError):
        fig.savefig(
            base_path.with_suffix(".pdf"),
            bbox_inches="tight",
            metadata={"Creator": "CEDAR boundary analysis", "CreationDate": None},
        )
        fig.savefig(
            base_path.with_suffix(".png"),
            bbox_inches="tight",
            dpi=600,
            metadata={"Software": "CEDAR boundary analysis"},
        )


def make_figures(
    output: Path,
    items: Sequence[Mapping[str, Any]],
    rq1: Mapping[str, Mapping[str, Any]],
    quantiles: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure_dir = output / "figures"
    source_names = ("E2A", "E2C", "C2", "E-1b")
    markers = ("o", "s", "^", "D")
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharey=True)
    for axis, dataset, marker, color in zip(
        axes.ravel(), source_names, markers, colors, strict=True
    ):
        summary = quantiles[dataset]
        labels = sorted(summary, key=lambda label: int(label[1:]))
        x = np.arange(1, len(labels) + 1)
        rates = [summary[label]["flip_rate"] for label in labels]
        lower = [
            rates[index] - summary[label]["cluster_bootstrap_95ci"][0]
            for index, label in enumerate(labels)
        ]
        upper = [
            summary[label]["cluster_bootstrap_95ci"][1] - rates[index]
            for index, label in enumerate(labels)
        ]
        axis.errorbar(
            x,
            rates,
            yerr=np.asarray([lower, upper]),
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            capsize=3,
        )
        for x_value, rate, label in zip(x, rates, labels, strict=True):
            count = summary[label]
            axis.annotate(
                f"{count['flips']}/{count['n']}",
                (x_value, rate),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        axis.set_title(dataset, fontsize=9, fontweight="bold")
        axis.set_xticks(x, labels)
        axis.set_xlabel("Uncapped gap quantile (Q1 = nearest)")
        axis.set_ylim(-0.03, 0.65)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0, 0].set_ylabel("Flip rate")
    axes[1, 0].set_ylabel("Flip rate")
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-01-flip-rate-by-baseline-gap-quartile")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 3.4))
    forest_names = ["E2A", "E2C", "C2", "E-1b", "Pooled"]
    forest_labels = [
        "E2A (thinking generation)",
        "E2C (thinking generation)",
        "C2 (direct→free; 1 flip)",
        "E−1b (A/B argmax)",
        "Pooled (mixed readouts)",
    ]
    y = np.arange(len(forest_names))[::-1]
    for y_value, name in zip(y, forest_names, strict=True):
        coefficient = rq1[name]["coefficients"]["z_g_top"]
        estimate = coefficient["odds_ratio"]
        lower_ci, upper_ci = coefficient["odds_ratio_95ci"]
        if estimate is None or lower_ci is None or upper_ci is None:
            axis.text(1.02, y_value, "NE", va="center", fontsize=8)
            continue
        marker = "o" if name != "C2" else "s"
        filled = "white" if rq1[name]["inferential_status"].startswith("descriptive") else "black"
        axis.errorbar(
            estimate,
            y_value,
            xerr=[
                [max(0.0, estimate - lower_ci)],
                [max(0.0, upper_ci - estimate)],
            ],
            fmt=marker,
            color="black",
            markerfacecolor=filled,
            capsize=3,
            linewidth=1.2,
        )
    axis.axvline(1.0, color="#777777", linestyle="--", linewidth=1)
    axis.set_yticks(y, forest_labels)
    axis.set_xscale("log")
    axis.set_xlabel("Odds ratio per 1-SD larger uncapped top gap")
    axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    axis.text(
        0.01,
        -0.18,
        "Hollow marker: <5 flips; descriptive only",
        transform=axis.transAxes,
        fontsize=7,
        ha="left",
    )
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-02-gap-effect-forest")
    plt.close(fig)

    c2 = [item for item in items if item["dataset"] == "C2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for axis, arm, color, marker in zip(
        axes, ("AO", "S"), ("#0072B2", "#D55E00"), ("o", "s"), strict=True
    ):
        selected = [item for item in c2 if item["arm"] == arm]
        x = np.asarray([float(item["g_top"]) for item in selected])
        y_values = np.asarray([int(item["any_flip"]) for item in selected])
        jitter = np.asarray(
            [(stable_hash_fraction(item["stable_id"]) - 0.5) * 0.16 for item in selected]
        )
        axis.scatter(
            x,
            y_values + jitter,
            s=14,
            alpha=0.35,
            color=color,
            marker=marker,
            edgecolors="none",
        )
        axis.set_title(f"C2 {arm}", fontsize=9, fontweight="bold")
        axis.set_xlabel("Direct uncapped top-1/top-2 gap")
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        axis.text(
            0.98,
            0.92,
            f"free flips: {int(y_values.sum())}/{len(y_values)}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    axes[0].set_ylabel("Free-generation flip (jittered)")
    axes[0].set_ylim(-0.18, 1.18)
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-03-c2-cross-readout-boundary")
    plt.close(fig)


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def write_reports(
    output: Path,
    analysis: Mapping[str, Any],
    gate: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> None:
    rq1 = analysis["rq1_any_flip"]
    loso = analysis["leave_one_source_out_pooled_point"]
    lines = [
        "# CEDAR Boundary Account post-hoc analysis",
        "",
        "> Post-hoc diagnostic after observing all source results; not a preregistered ",
        "> confirmation and not causal-mechanism evidence.",
        "",
        "## Outcome",
        "",
        f"- Claim gate: **{gate['gate']}**",
        f"- Allowed wording: {gate['allowed_wording']}",
        f"- Input audit: `{input_audit['status']}`",
        "",
        "## RQ1: uncapped top-gap and any flip",
        "",
        "| Source | n | flips | slope | OR | 95% bootstrap CI (slope) | AUC(-gap) | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("E2A", "E2C", "C2", "E-1b", "Pooled"):
        result = rq1[name]
        coefficient = result["coefficients"]["z_g_top"]
        interval = coefficient["bootstrap_95ci"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(result["descriptive"]["n"]),
                    str(result["descriptive"]["flips"]),
                    fmt(coefficient["estimate"]),
                    fmt(coefficient["odds_ratio"]),
                    f"[{fmt(interval[0])}, {fmt(interval[1])}]",
                    fmt(result["descriptive"]["auc_negative_gap"]),
                    result["inferential_status"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## What the evidence changes",
            "",
            "- The pooled model tests whether pre-intervention proximity predicts later ",
            "  discrete instability after holding source strata fixed.",
            "- C2 is a deliberately separate cross-readout check. Its direct predictor ",
            "  and free-generation outcome are not algebraically identical, but its event ",
            "  count limits inference.",
            "- E−1b is forced choice and therefore provides geometry consistency rather ",
            "  than independent free-generation validation.",
            f"- Removing E−1b changes the pooled slope to "
            f"{fmt(loso['E-1b']['estimate'])}; therefore the pooled negative association "
            "  is not evidence that the generative assays share the same boundary effect.",
            "",
            "## Five required answers",
            "",
            f"1. **E2A:** No robust concentration after arm stratification. The frozen "
            f"   fixed-stratum slope is {fmt(rq1['E2A']['coefficients']['z_g_top']['estimate'])} "
            f"   with CI [{fmt(rq1['E2A']['coefficients']['z_g_top']['bootstrap_95ci'][0])}, "
            f"{fmt(rq1['E2A']['coefficients']['z_g_top']['bootstrap_95ci'][1])}], in the "
            "   opposite direction, despite a favorable unadjusted AUC.",
            f"2. **E2C:** Not directionally consistent in the fixed-stratum model; slope "
            f"   {fmt(rq1['E2C']['coefficients']['z_g_top']['estimate'])}, CI "
            f"[{fmt(rq1['E2C']['coefficients']['z_g_top']['bootstrap_95ci'][0])}, "
            f"{fmt(rq1['E2C']['coefficients']['z_g_top']['bootstrap_95ci'][1])}].",
            f"3. **C2:** Direct margins do not establish prediction of free-generation "
            f"   flips: {rq1['C2']['descriptive']['flips']}/{rq1['C2']['descriptive']['n']} "
            f"   events and AUC {fmt(rq1['C2']['descriptive']['auc_negative_gap'])}.",
            f"4. **E−1b:** The small-margin pattern is strong but primarily forced-choice "
            f"   boundary consistency (AUC {fmt(rq1['E-1b']['descriptive']['auc_negative_gap'])}); "
            "   it cannot supply independent free-generation evidence.",
            f"5. **Wording:** {gate['gate']} permits only `consistent with`; `reconciles` "
            "   is not supported.",
            "",
            "## Claim Candidates",
            "",
            "- Claim:",
            f"  - Allowed wording: {gate['allowed_wording']}",
            "  - Source evidence: `BOUNDARY_ANALYSIS.json`, RQ1 and claim gate.",
            "  - Forbidden stronger wording: the boundary account is a proven internal mechanism.",
            "  - Uncertainty: post-hoc design, heterogeneous readouts, and low C2 free-flip count.",
            f"  - Decision: {'keep' if gate['gate'] == 'Gate A' else 'weaken'}",
            "",
        ]
    )
    (output / "analysis-report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )

    stats = [
        "# Statistical appendix: Boundary Account v1",
        "",
        "## Design",
        "",
        "- Bootstrap: 10,000 percentile draws, seed 20260726.",
        "- Primary predictor: uncapped option top-1 minus top-2 logit gap.",
        "- Gaps standardized within dataset × arm/readout on the full valid set.",
        "- Fixed strata: E2 dataset/arm; C2 arm/domain; E−1b seed/variant.",
        "- Natural clusters: E2 family, C2 normalized stem, E−1b pair.",
        "- E2A/E2C use a coupled structural-family cluster in the pooled bootstrap.",
        "",
        "## Source estimates",
        "",
    ]
    for name in ("E2A", "E2C", "C2", "E-1b", "Pooled"):
        result = rq1[name]
        coefficient = result["coefficients"]["z_g_top"]
        stats.extend(
            [
                f"### {name}",
                "",
                f"- n/flips: {result['descriptive']['n']}/{result['descriptive']['flips']}",
                f"- slope: {fmt(coefficient['estimate'], 6)}; "
                f"95% CI [{fmt(coefficient['bootstrap_95ci'][0], 6)}, "
                f"{fmt(coefficient['bootstrap_95ci'][1], 6)}]",
                f"- odds ratio: {fmt(coefficient['odds_ratio'], 6)}",
                f"- median gap (flip): {fmt(result['descriptive']['gap_flip']['median'], 6)}",
                f"- median gap (stable): {fmt(result['descriptive']['gap_stable']['median'], 6)}",
                f"- AUC using -gap: {fmt(result['descriptive']['auc_negative_gap'], 6)}; "
                f"95% CI [{fmt(result['auc_negative_gap_bootstrap_95ci'][0], 6)}, "
                f"{fmt(result['auc_negative_gap_bootstrap_95ci'][1], 6)}]",
                f"- nearest-minus-farthest flip rate: "
                f"{fmt(result['descriptive']['nearest_minus_farthest_flip_rate'], 6)}",
                f"- status: `{result['inferential_status']}`",
                "",
            ]
        )
    stats.extend(
        [
            "## Leave-one-source-out diagnostic",
            "",
            *[
                f"- Omit {name}: pooled slope {fmt(result['estimate'], 6)} "
                f"(`{result['status']}`)."
                for name, result in sorted(loso.items())
            ],
            "",
            "## Multiplicity",
            "",
            "Source-specific bootstrap sign p-values are Holm adjusted only where the ",
            "source has at least five flips. The pooled post-hoc diagnostic is the single ",
            "frozen primary test for this analysis bundle.",
            "",
            "## Limitations",
            "",
            "- E2C retains one truncated uncapped row with missing logits; it is not imputed.",
            "- C2 direct/free readout disagreement is expected and explicitly measured.",
            "- E−1b margin-to-choice association is partly algebraic.",
            "- Delta-m models contain a post-intervention variable and are not mediation.",
            "",
        ]
    )
    (output / "stats-appendix.md").write_text(
        "\n".join(stats), encoding="utf-8", newline="\n"
    )

    catalog = """# Boundary Account figure catalog

## Figure 1 — flip rate by baseline-gap quantile

- Files: `figure-01-flip-rate-by-baseline-gap-quartile.pdf/.png`
- Purpose: test whether discrete changes concentrate near the uncapped boundary.
- Data: all four sources; source-native clusters; Q1 is the smallest top gap.
- Error bars: 95% cluster-bootstrap percentile intervals; labels are raw flip/n.
- Interpretation: compare Q1 with Q4; strict monotonicity is not required.
- Caveat: source readouts differ and E−1b is forced choice.

## Figure 2 — standardized gap-effect forest

- Files: `figure-02-gap-effect-forest.pdf/.png`
- Purpose: compare fixed-stratum gap slopes across sources and in the pooled model.
- Metric: odds ratio per one-SD larger uncapped top gap; reference is 1.
- Error bars: 95% cluster-bootstrap percentile intervals.
- Interpretation: values below 1 indicate fewer flips farther from the boundary.
- Caveat: low-event estimates are hollow and descriptive only.

## Figure 3 — C2 cross-readout boundary diagnostic

- Files: `figure-03-c2-cross-readout-boundary.pdf/.png`
- Purpose: test whether direct uncapped gaps predict matched free-generation changes.
- Data: raw AO/S points with deterministic vertical jitter and exact event counts.
- Interpretation: this is the least algebraic cross-readout test in the bundle.
- Caveat: low free-generation event counts sharply limit inference.
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8", newline="\n")


def build_delivery_manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "delivery_manifest.json":
            relative = path.relative_to(output).as_posix()
            files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return {
        "schema": "cedar.boundary_account_posthoc.delivery.v1",
        "status": "complete",
        "files": files,
        "manifest_excludes_itself": True,
        "gpu_used": False,
        "new_model_outputs_generated": False,
    }


def run_analysis(output: Path, *, draws: int, seed: int) -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    inputs = plan["inputs"]
    e2a_rows = load_jsonl_blob(inputs["e2a_results"])
    e2c_rows = load_jsonl_blob(inputs["e2c_results"])
    c2_du = load_jsonl_blob(inputs["c2_direct_uncapped"])
    c2_dp = load_jsonl_blob(inputs["c2_direct_p25"])
    c2_fu = load_jsonl_blob(inputs["c2_free_uncapped"])
    c2_fp = load_jsonl_blob(inputs["c2_free_p25"])
    e1_rows = load_jsonl_blob(inputs["e_minus1b_rows"])
    e1_pairs = load_jsonl_blob(inputs["e_minus1b_pairs"])
    c2_addendum = load_csv_blob(inputs["c2_addendum_crosscheck"])

    e2a, e2a_audit = build_e2_items(e2a_rows, "E2A")
    e2c, e2c_audit = build_e2_items(e2c_rows, "E2C")
    c2, c2_audit = build_c2_items(c2_du, c2_dp, c2_fu, c2_fp)
    e1, e1_audit = build_e_minus1b_items(e1_rows)
    datasets = {"E2A": e2a, "E2C": e2c, "C2": c2, "E-1b": e1}
    build_audits = {
        "E2A": e2a_audit,
        "E2C": e2c_audit,
        "C2": c2_audit,
        "E-1b": {**e1_audit, "pair_results_rows": len(e1_pairs)},
    }
    input_audit = audit_inputs(plan, datasets, build_audits, c2_addendum)
    write_json(output / "INPUT_AUDIT.json", input_audit)
    if input_audit["status"] != "passed":
        raise ValueError(f"Input audit failed: {input_audit['failures']}")

    items = e2a + e2c + c2 + e1
    standardization = {
        "g_top": standardize_field(items, "g_top", "z_g_top"),
        "g_initial": standardize_field(items, "g_initial", "z_g_initial"),
        "g_target": standardize_field(items, "g_target", "z_g_target"),
        "delta_m_general": standardize_field(
            items, "delta_m_general", "z_delta_m_general"
        ),
    }
    quantile_metadata = assign_gap_quantiles(items)

    rq1 = {}
    all_bootstrap_rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(("E2A", "E2C", "C2", "E-1b")):
        result, records = bootstrap_model(
            source_rows(items, dataset),
            scope=dataset,
            outcome="any_flip",
            predictors=["z_g_top"],
            gap_field="g_top",
            draws=draws,
            seed=seed + index,
            pooled=False,
        )
        rq1[dataset] = result
        all_bootstrap_rows.extend(records)
    pooled_result, pooled_records = bootstrap_model(
        items,
        scope="Pooled",
        outcome="any_flip",
        predictors=["z_g_top"],
        gap_field="g_top",
        draws=draws,
        seed=seed + 100,
        pooled=True,
    )
    rq1["Pooled"] = pooled_result
    all_bootstrap_rows.extend(pooled_records)

    e2_vf = [item for item in e2a + e2c if item["arm"] in {"V", "F"}]
    rq2, rq2_records = bootstrap_model(
        [item for item in e2_vf if item["initial_exit"] is not None],
        scope="RQ2_E2_VF_initial_exit",
        outcome="initial_exit",
        predictors=["z_g_initial"],
        gap_field="g_initial",
        draws=draws,
        seed=seed + 200,
        pooled=True,
        include_quantiles=False,
    )
    all_bootstrap_rows.extend(rq2_records)
    rq3, rq3_records = bootstrap_model(
        [item for item in e2_vf if item["target_entry"] is not None],
        scope="RQ3_E2_VF_target_entry",
        outcome="target_entry",
        predictors=["z_g_target"],
        gap_field="g_target",
        draws=draws,
        seed=seed + 300,
        pooled=True,
        include_quantiles=False,
    )
    all_bootstrap_rows.extend(rq3_records)
    rq4_rows = [
        item
        for item in e2a + e2c + c2
        if item["delta_m_general"] is not None and item["any_flip"] is not None
    ]
    rq4, rq4_records = bootstrap_model(
        rq4_rows,
        scope="RQ4_boundary_crossing_description",
        outcome="any_flip",
        predictors=["z_g_top", "z_delta_m_general"],
        gap_field="g_top",
        draws=draws,
        seed=seed + 400,
        pooled=True,
        include_quantiles=False,
    )
    all_bootstrap_rows.extend(rq4_records)

    source_raw_p = {
        name: (
            result["coefficients"]["z_g_top"]["bootstrap_two_sided_sign_p"]
            if result["descriptive"]["flips"] >= 5
            else None
        )
        for name, result in rq1.items()
        if name != "Pooled"
    }
    source_holm = holm_adjust(source_raw_p)
    for name in source_raw_p:
        rq1[name]["coefficients"]["z_g_top"]["holm_adjusted_p"] = source_holm[name]

    loco = leave_one_cluster_out(items, "any_flip", "z_g_top")
    loso = leave_one_source_out(items, "any_flip", "z_g_top")
    quantile_summaries = {
        dataset: quantile_bootstrap_summary(
            source_rows(items, dataset), draws=draws, seed=seed + 500 + index
        )
        for index, dataset in enumerate(("E2A", "E2C", "C2", "E-1b"))
    }
    direct_c2 = {
        arm: {
            "n": len([item for item in c2 if item["arm"] == arm]),
            "direct_argmax_flips": sum(
                item["direct_any_flip"] for item in c2 if item["arm"] == arm
            ),
            "free_generation_flips": sum(
                item["any_flip"] for item in c2 if item["arm"] == arm
            ),
        }
        for arm in ("AO", "S")
    }
    e1_order = {
        variant: descriptive_boundary(
            [item for item in e1 if item["variant"] == variant], "g_top"
        )
        for variant in ("original", "swapped")
    }
    e1_output_space = {
        "delta_entropy": summarize(
            item["p25_full_vocab_entropy"] - item["uncapped_full_vocab_entropy"]
            for item in e1
        ),
        "delta_option_mass": summarize(
            item["p25_option_mass"] - item["uncapped_option_mass"] for item in e1
        ),
        "top_token_is_option_rate_uncapped": mean(
            int(item["uncapped_top_token_is_option"]) for item in e1
        ),
        "top_token_is_option_rate_p25": mean(
            int(item["p25_top_token_is_option"]) for item in e1
        ),
        "interpretation": "forced-choice output disturbance background; not answer quality",
    }

    analysis = {
        "schema": "cedar.boundary_account_posthoc.analysis.v1",
        "status": "complete",
        "post_hoc": True,
        "gpu_used": False,
        "new_model_outputs_generated": False,
        "bootstrap": {"draws": draws, "seed": seed},
        "counts": {dataset: len(rows) for dataset, rows in datasets.items()},
        "standardization": standardization,
        "quantile_assignment": quantile_metadata,
        "rq1_any_flip": rq1,
        "rq2_initial_exit_e2_vf": rq2,
        "rq3_target_entry_e2_vf": rq3,
        "rq4_post_intervention_crossing_description": rq4,
        "c2_direct_argmax_consistency_check": direct_c2,
        "e_minus1b_order_breakdown": e1_order,
        "e_minus1b_output_space_background": e1_output_space,
        "leave_one_cluster_out_pooled": loco,
        "leave_one_source_out_pooled_point": loso,
        "quantile_summaries": quantile_summaries,
        "source_specific_holm": {
            "raw_bootstrap_sign_p": source_raw_p,
            "adjusted": source_holm,
        },
    }
    gate = build_gate(rq1, loco, loso)

    write_json(output / "BOUNDARY_ANALYSIS.json", analysis)
    write_json(output / "CLAIM_GATE.json", gate)
    item_fields = [
        "dataset",
        "readout",
        "item_id",
        "base_id",
        "cluster_id",
        "pooled_cluster_id",
        "structure_class",
        "arm",
        "domain",
        "sample_seed",
        "variant",
        "initial_answer",
        "target_answer",
        "choice_uncapped",
        "choice_p25",
        "logit_argmax_uncapped",
        "logit_argmax_p25",
        "g_top",
        "g_initial",
        "g_target",
        "m_general_uncapped",
        "m_general_p25",
        "m_specific_uncapped",
        "m_specific_p25",
        "delta_m_general",
        "target_minus_initial_uncapped",
        "target_minus_initial_p25",
        "delta_baseline_winner_advantage",
        "z_g_top",
        "z_g_initial",
        "z_g_target",
        "z_delta_m_general",
        "gap_quantile",
        "transition",
        "any_flip",
        "initial_exit",
        "target_entry",
        "stratum",
        "predictor_source",
        "outcome_source",
    ]
    write_csv(output / "per_item_boundary_metrics.csv", items, item_fields)
    bootstrap_fields = sorted({key for row in all_bootstrap_rows for key in row})
    write_csv(
        output / "cluster_bootstrap_results.csv", all_bootstrap_rows, bootstrap_fields
    )
    transition_rows = transition_counts(items)
    write_csv(
        output / "transition_counts.csv",
        transition_rows,
        ["dataset", "readout", "arm", "transition", "count"],
    )
    make_figures(output, items, rq1, quantile_summaries)
    write_reports(output, analysis, gate, input_audit)
    write_json(output / "delivery_manifest.json", build_delivery_manifest(output))
    return {"analysis": analysis, "gate": gate, "input_audit": input_audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_analysis(
            args.output_root.resolve(),
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )
    pooled = result["analysis"]["rq1_any_flip"]["Pooled"]["coefficients"]["z_g_top"]
    print(
        json.dumps(
            {
                "status": result["analysis"]["status"],
                "input_audit": result["input_audit"]["status"],
                "gate": result["gate"]["gate"],
                "pooled_slope": pooled,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
