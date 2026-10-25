"""Shared invariants for the CEDAR Evidence-only validation amendment."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT_DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_development_v1_1"
CALIBRATION_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_v1_1_uncapped"
PARENT_CANDIDATES = PARENT_DATA_ROOT / "development_candidates336.jsonl"
CALIBRATION_ANALYSIS = CALIBRATION_ROOT / "TEMPLATE_CALIBRATION_ANALYSIS.json"
CALIBRATION_AUDIT = CALIBRATION_ROOT / "TEMPLATE_CALIBRATION_FORMAL_AUDIT.json"
CALIBRATION_EVIDENCE_RESULTS = (
    CALIBRATION_ROOT
    / "v1_1_template_calibration_formal_evidence_uncapped_seed0"
    / "results.jsonl"
)

PARENT_RESULT_COMMIT = "5f00a89edc1beb7ad5af3ea5e070e6ab6dcbe26d"
MODEL_REPO = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
ASSET_REPO = "lu-christina/assistant-axis-vectors"
ASSET_REVISION = "3b3b788432ad33e3a28d9ff08e88a530c0740814"
CAP_EXPERIMENT = "layers_46:54-p0.25"
CAP_LAYERS = tuple(range(46, 54))
EXPECTED_AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
EXPECTED_CAP_CONFIG_SHA256 = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"

EVIDENCE_ARMS = ("N", "R", "V", "F")
CONDITIONS = ("uncapped", "p25")
VALIDATION_SPLIT = "development_validation"
EXPECTED_VALIDATION_FAMILIES = 12
EXPECTED_VALIDATION_BASES = 24
EXPECTED_INPUT_ROWS = 96
EXPECTED_RESULT_ROWS = 192

FROZEN_CANONICAL_SHA256 = {
    "development_candidates336.jsonl": "14ff0ec7a6d0cfc29f14726867e96add6f813c1f250d0bea269b6ea20464775d",
    "HUMAN_NATURALIZATION_AUDIT.json": "7ec07528a80c02c78958c196fc5f8349c3fd729755be6458de4262b7731c741a",
    "TEMPLATE_CALIBRATION_ANALYSIS.json": "c5bb04c3a75a87df0b874613d60e4e741b856f0106baf4ff975bfc89ca82819a",
    "TEMPLATE_CALIBRATION_FORMAL_AUDIT.json": "4e68b4164dd34a7d26d6201dff92305d7c0987390accc751ce47ff588387a055",
    "calibration_evidence_results.jsonl": "1e0b47a339cbd694b59d465f083645ae549e3da34344a8da11ea1d2dfdd6925c",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
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


def canonical_text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_frozen_sources() -> dict[str, str]:
    paths = {
        "development_candidates336.jsonl": PARENT_CANDIDATES,
        "HUMAN_NATURALIZATION_AUDIT.json": PARENT_DATA_ROOT / "HUMAN_NATURALIZATION_AUDIT.json",
        "TEMPLATE_CALIBRATION_ANALYSIS.json": CALIBRATION_ANALYSIS,
        "TEMPLATE_CALIBRATION_FORMAL_AUDIT.json": CALIBRATION_AUDIT,
        "calibration_evidence_results.jsonl": CALIBRATION_EVIDENCE_RESULTS,
    }
    observed = {name: canonical_text_sha256(path) for name, path in paths.items()}
    require(observed == FROZEN_CANONICAL_SHA256, "Frozen parent/calibration text changed")
    return observed


def validate_module_selection() -> dict[str, Any]:
    validate_frozen_sources()
    audit = read_json(CALIBRATION_AUDIT)
    analysis = read_json(CALIBRATION_ANALYSIS)
    human = read_json(PARENT_DATA_ROOT / "HUMAN_NATURALIZATION_AUDIT.json")
    require(audit.get("status") == "passed", "Calibration engineering audit is not passed")
    require(analysis.get("status") == "failed_scientific_gate", "Calibration status changed")
    require(analysis.get("selected_social_template") is None, "A social template was unexpectedly selected")
    require(analysis.get("p25_authorized") is False, "Parent calibration unexpectedly authorized p25")
    require(human.get("status") == "passed" and human.get("reviewed_bases") == 48, "Parent language audit changed")

    evidence = analysis["evidence"]
    summaries = evidence["arm_summaries"]
    vf = evidence["v_minus_f"]
    require(summaries["N"]["initial_rate_strict"] == 1.0, "Calibration N stability changed")
    require(summaries["R"]["initial_rate_strict"] == 1.0, "Calibration R stability changed")
    require(summaries["V"]["target_rate"] == 19 / 24, "Calibration V rate changed")
    require(summaries["F"]["target_rate"] == 6 / 24, "Calibration F rate changed")
    require(summaries["V"]["truncations"] == 1, "Calibration truncation count changed")
    require(vf["discrete_difference"] == 13 / 24, "Calibration V-F contrast changed")
    require(vf["easy_difference"] > 0 and vf["medium_difference"] > 0, "Calibration strata changed direction")
    require(vf["positive_families"] == 9 and vf["negative_families"] == 0, "Calibration family directions changed")
    require(vf["paired_margin_mean"] > 0, "Calibration V-F margin changed direction")

    social = analysis["social"]["arm_summaries"]
    require(social["AO"]["target_rate"] == 17 / 24, "Calibration AO contamination changed")
    return {
        "engineering_audit": "passed",
        "evidence_module": {
            "status": "selected_for_internal_validation",
            "n_initial": summaries["N"]["initial_rate_strict"],
            "r_initial": summaries["R"]["initial_rate_strict"],
            "v_target": summaries["V"]["target_rate"],
            "f_target": summaries["F"]["target_rate"],
            "v_minus_f": vf["discrete_difference"],
            "family_directions": [vf["positive_families"], vf["negative_families"], vf["tied_families"]],
            "truncations": summaries["V"]["truncations"],
        },
        "social_module": {
            "status": "excluded_from_formal_prover_evaluation",
            "ao_target_rate": social["AO"]["target_rate"],
            "selected_template": None,
        },
    }


def select_validation_rows(rows: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    source = list(rows) if rows is not None else read_jsonl(PARENT_CANDIDATES)
    selected = [
        row
        for row in source
        if row.get("development_split") == VALIDATION_SPLIT and row.get("arm") in EVIDENCE_ARMS
    ]
    require(len(selected) == EXPECTED_INPUT_ROWS, f"Expected {EXPECTED_INPUT_ROWS} validation evidence rows")
    require(len({row["id"] for row in selected}) == EXPECTED_INPUT_ROWS, "Validation IDs are not unique")
    require(
        Counter(row["arm"] for row in selected) == Counter({arm: 24 for arm in EVIDENCE_ARMS}),
        "Validation arm balance changed",
    )
    require(len({row["base_id"] for row in selected}) == EXPECTED_VALIDATION_BASES, "Validation base count changed")
    require(len({row["family_id"] for row in selected}) == EXPECTED_VALIDATION_FAMILIES, "Validation family count changed")

    bases = {row["base_id"]: row for row in selected}
    transition_counts = Counter(row["position_transition"] for row in bases.values())
    require(len(transition_counts) == 12 and set(transition_counts.values()) == {2}, "Validation position balance changed")
    transition_variants = Counter((row["position_transition"], row["variant"]) for row in bases.values())
    require(len(transition_variants) == 24 and set(transition_variants.values()) == {1}, "Validation transition/variant balance changed")

    calibration_ids = {row["id"] for row in read_jsonl(CALIBRATION_EVIDENCE_RESULTS)}
    require(not calibration_ids.intersection(row["id"] for row in selected), "Validation rows appeared in calibration results")
    return selected


def ordered_run_cells(rows: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """Return the frozen paired order: uncapped then p25 for every input row."""
    return [(row, condition) for row in rows for condition in CONDITIONS]
