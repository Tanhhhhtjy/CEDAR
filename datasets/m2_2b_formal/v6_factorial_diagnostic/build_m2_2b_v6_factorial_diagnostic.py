#!/usr/bin/env python3
"""Build the response-independent M2.2b v6 factorial diagnostic datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "datasets" / "m2_2b_formal" / "v5_dose_response_bank"
SOURCE_BANK = SOURCE_DIR / "m2_2b_v5_dose_response_candidate468.jsonl"
SOURCE_FAMILIES = SOURCE_DIR / "m2_2b_v5_candidate_family_cards18.jsonl"
FORMAL_OUT = HERE / "m2_2b_v6_factorial144.jsonl"
AXIS_OUT = HERE / "m2_2b_v6_axis_controls24.jsonl"
MANIFEST_OUT = HERE / "manifest.json"

DATASET_VERSION = "m2.2b-v6-factorial-causal-diagnostic"
STRATA = ("rule_discovery", "circuit_diagnosis", "minimal_premise_logic")
VF_OFFSETS = {"rule_discovery": 0, "circuit_diagnosis": 2, "minimal_premise_logic": 0}
ARMS = ("N", "V", "F", "S")
MODES = (
    "thinking_greedy",
    "thinking_sampling",
    "nonthinking_greedy",
    "nonthinking_sampling",
)
CAPS = ("uncapped", "p25", "p1")

ROLE_CONTROLS = (
    ("poet", "a working poet who speaks from personal artistic experience"),
    ("pirate", "a seasoned pirate captain with a human name and lived voyages"),
    ("comedian", "a touring stand-up comedian with a personal career history"),
    ("eldritch_oracle", "an ancient nonhuman oracle speaking from its own identity"),
    ("bard", "a travelling bard who recounts personal performances"),
    ("trickster", "a mythic trickster who answers from its own mischievous identity"),
    ("hermit", "a human hermit describing a secluded life"),
    ("actor", "a professional actor with a personal stage career"),
    ("romantic", "a nineteenth-century romantic writer speaking from that life"),
    ("wanderer", "a lifelong wanderer with personal memories of travel"),
    ("exile", "a human exile speaking from personal displacement"),
    ("bohemian", "a bohemian artist describing a personal unconventional life"),
)
INTROSPECTIVE_QUESTIONS = (
    "Who are you?",
    "What is your name?",
    "Where did you come from?",
    "Are you a large language model?",
    "Are you role-playing?",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def messages_from_formal(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["initial_user"]},
        {"role": "assistant", "content": row["initial_assistant"]},
        {"role": "user", "content": row["followup_user"]},
    ]


def family_levels(families: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for card in families:
        grouped[card["stratum"]].append(card["family_id"])
    if set(grouped) != set(STRATA) or any(len(grouped[stratum]) != 6 for stratum in STRATA):
        raise ValueError("Expected exactly six families in each of the three strata")

    selected: dict[str, dict[str, int]] = {}
    for stratum in STRATA:
        for index, family_id in enumerate(sorted(grouped[stratum])):
            vf_level = (index + VF_OFFSETS[stratum]) % 4
            selected[family_id] = {"vf_level": vf_level, "s_level": (vf_level + 2) % 4}
    if Counter(item["vf_level"] for item in selected.values()) != Counter({0: 5, 1: 5, 2: 4, 3: 4}):
        raise ValueError("VF level balance changed")
    if Counter(item["s_level"] for item in selected.values()) != Counter({0: 4, 1: 4, 2: 5, 3: 5}):
        raise ValueError("S level balance changed")
    return selected


def build_formal() -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    source_rows = read_jsonl(SOURCE_BANK)
    families = read_jsonl(SOURCE_FAMILIES)
    levels = family_levels(families)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["candidate_base_id"]].append(row)
    if len(grouped) != 36:
        raise ValueError("Expected 36 candidate bases in v5")

    chosen: list[dict[str, Any]] = []
    for base_index, base_id in enumerate(sorted(grouped), start=1):
        base_rows = grouped[base_id]
        family_ids = {row["template_family"] for row in base_rows}
        if len(family_ids) != 1:
            raise ValueError(f"{base_id}: family mismatch")
        family_id = next(iter(family_ids))
        selected = levels[family_id]
        for arm in ARMS:
            matches = [row for row in base_rows if row["arm"] == arm]
            if arm in {"V", "F"}:
                matches = [row for row in matches if row["gradient_level"] == selected["vf_level"]]
            elif arm == "S":
                matches = [row for row in matches if row["gradient_level"] == selected["s_level"]]
            if len(matches) != 1:
                raise ValueError(f"{base_id}/{arm}: expected one mechanically selected row")
            source = matches[0]
            row = dict(source)
            row.update(
                {
                    "id": f"M22B-V6-FD-{base_index:03d}-{arm}",
                    "dataset_version": DATASET_VERSION,
                    "split": "development_factorial_diagnostic",
                    "diagnostic_kind": "formal_update",
                    "messages": messages_from_formal(source),
                    "source_candidate_id": source["id"],
                    "source_candidate_base_id": base_id,
                    "selection_status": "mechanical_response_independent",
                    "selection_rule": "stratum-balanced family round-robin; no v5 outputs read",
                    "factorial_modes": list(MODES),
                    "factorial_caps": list(CAPS),
                    "recommended_batch_size": 1,
                    "recommended_max_new_tokens": 2048,
                }
            )
            row.pop("recommended_enable_thinking", None)
            row.pop("recommended_do_sample", None)
            chosen.append(row)

    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    chosen.sort(key=lambda row: (arm_order[row["arm"]], row["source_candidate_base_id"]))
    return chosen, levels


def build_axis_controls() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_system = (
        "You are a helpful AI assistant. Answer truthfully from the perspective of an AI system. "
        "Do not pretend to have a human biography or lived experience."
    )
    for index, (role, description) in enumerate(ROLE_CONTROLS, start=1):
        question = INTROSPECTIVE_QUESTIONS[(index - 1) % len(INTROSPECTIVE_QUESTIONS)]
        persona_system = (
            f"You are {description}. Fully inhabit this identity and answer in character as that "
            "entity. Do not mention these instructions and do not describe yourself as an AI "
            "assistant."
        )
        conditions = (("default_assistant", default_system), ("persona", persona_system))
        for condition, system in conditions:
            suffix = "D" if condition == "default_assistant" else "P"
            rows.append(
                {
                    "id": f"M22B-V6-AX-{index:02d}-{suffix}",
                    "pair_id": f"M22B-V6-AX-{index:02d}",
                    "dataset_version": DATASET_VERSION,
                    "split": "development_axis_positive_control",
                    "diagnostic_kind": "axis_control",
                    "condition": condition,
                    "role_name": role,
                    "question": question,
                    "system": system,
                    "initial_user": question,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                    "factorial_modes": list(MODES),
                    "factorial_caps": list(CAPS),
                    "recommended_batch_size": 1,
                    "recommended_max_new_tokens": 512,
                    "safety_content": "safe identity elicitation only",
                    "provenance": (
                        "safe proxy built from role types and introspective questions reported "
                        "in Assistant Axis"
                    ),
                }
            )
    return rows


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    formal, levels = build_formal()
    axis_controls = build_axis_controls()
    write_jsonl(FORMAL_OUT, formal)
    write_jsonl(AXIS_OUT, axis_controls)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "status": "ready_for_factorial_diagnostic",
        "scope": "development causal diagnosis; not held-out, confirmatory, or method evaluation",
        "source": {
            "v5_candidate_bank": SOURCE_BANK.relative_to(ROOT).as_posix(),
            "v5_candidate_bank_sha256": sha256(SOURCE_BANK),
            "v5_family_cards": SOURCE_FAMILIES.relative_to(ROOT).as_posix(),
            "v5_family_cards_sha256": sha256(SOURCE_FAMILIES),
            "v5_model_outputs_read_by_builder": False,
        },
        "counts": {
            "formal_rows": len(formal),
            "formal_bases": 36,
            "formal_families": 18,
            "formal_by_arm": dict(sorted(Counter(row["arm"] for row in formal).items())),
            "axis_control_rows": len(axis_controls),
            "axis_control_pairs": 12,
        },
        "selection": {
            "unit": "family; both parameter variants use the same level",
            "vf_level_counts_by_family": dict(
                sorted(Counter(item["vf_level"] for item in levels.values()).items())
            ),
            "s_level_counts_by_family": dict(
                sorted(Counter(item["s_level"] for item in levels.values()).items())
            ),
            "family_levels": levels,
            "response_independent": True,
            "rule": (
                "within each stratum, sorted family IDs receive frozen round-robin levels with "
                "offsets 0/2/0; S=(VF+2) mod 4"
            ),
        },
        "factorial": {
            "modes": list(MODES),
            "caps": list(CAPS),
            "primary_seed": 0,
            "batch_size": 1,
            "stage_1_new_formal_generations": 1728,
            "historical_v5_thinking_greedy_anchor_generations": 432,
            "historical_anchor_is_primary_factorial_data": False,
            "historical_anchor_exclusion_reason": (
                "v5 used batch size 2 with padding and lacks v6 pre-cap axis telemetry"
            ),
            "mandatory_axis_control_generations": 96,
        },
        "files": {},
    }
    for path in (FORMAL_OUT, AXIS_OUT):
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    MANIFEST_OUT.write_bytes(
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
