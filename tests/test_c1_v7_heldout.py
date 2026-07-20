from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "m2_2b_formal" / "v7_heldout_confirmation"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_module(
    "validate_m2_2b_v7_heldout",
    DATA_DIR / "validate_m2_2b_v7_heldout.py",
)
builder = _load_module(
    "build_m2_2b_v7_heldout",
    DATA_DIR / "build_m2_2b_v7_heldout.py",
)


def _families() -> list[dict[str, Any]]:
    strata = (
        *("arithmetic_accounting" for _ in range(8)),
        *("unit_label_rule_application" for _ in range(8)),
        *("multistep_logic" for _ in range(8)),
    )
    return [
        {
            "dataset_version": validator.VERSION,
            "family_id": f"C1V7-F{index:02d}",
            "stratum": stratum,
            "family_title": f"Synthetic schema family {index}",
            "reasoning_structure": f"synthetic structure {index}",
            "initial_key_fact": f"synthetic key fact {index}",
            "answer_generation_mechanism": f"synthetic answer mechanism {index}",
            "v_changed_node": f"synthetic changed node {index}",
            "v_update_mechanism": f"synthetic update mechanism {index}",
            "ao_targeting_mechanism": validator.AO_TARGETING_MECHANISM,
            "s_targeting_mechanism": validator.S_TARGETING_MECHANISM,
            "migration_structure": f"synthetic migration structure {index}",
            "nearest_development_families": [f"development-neighbor-{index}"],
            "nonisomorphism_rationale": f"synthetic nonisomorphism rationale {index}",
            "variants": [1, 2],
            "new_structure_attestation": True,
            "development_overlap_review_status": "approved_by_data_lead",
        }
        for index, stratum in enumerate(strata, 1)
    ]


def _bases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    migrations = validator.MIGRATIONS * 8
    strata = (
        *("arithmetic_accounting" for _ in range(16)),
        *("unit_label_rule_application" for _ in range(16)),
        *("multistep_logic" for _ in range(16)),
    )
    for index, (migration, stratum) in enumerate(zip(migrations, strata, strict=True), 1):
        initial, target = migration.split("_to_")
        distractor = next(label for label in validator.LABELS if label not in {initial, target})
        family_index = (index - 1) // 2 + 1
        variant = (index - 1) % 2 + 1
        initial_assistant = json.dumps(
            {"answer": initial, "confidence": 95, "reason": "Synthetic unique derivation."},
            separators=(",", ":"),
        )
        options = {
            label: f"Synthetic option {index}-{label}"
            for label in validator.LABELS
        }
        original_user = (
            f"Synthetic task {index} with a uniquely determined answer.\n"
            + "\n".join(f"({label}) {options[label]}" for label in validator.LABELS)
            + f"\n\n{validator.RESPONSE_CONTRACT}"
        )
        new_evidence = f"Updated task fact {index}: the decisive state has changed."
        rows.append(
            {
                "dataset_version": validator.VERSION,
                "base_id": f"C1V7-{index:03d}",
                "family_id": f"C1V7-F{family_index:02d}",
                "variant": variant,
                "stratum": stratum,
                "initial_answer": initial,
                "target_answer": target,
                "distractor_answer": distractor,
                "answer_migration": migration,
                "options": options,
                "original_gold_rationale": "Synthetic original rationale.",
                "updated_gold_rationale": "Synthetic updated rationale.",
                "messages_common": [
                    {"role": "system", "content": validator.EXPECTED_SYSTEM},
                    {
                        "role": "user",
                        "content": original_user,
                    },
                    {"role": "assistant", "content": initial_assistant},
                ],
                "v_new_evidence": new_evidence,
                "v_followup": (
                    f"{new_evidence} Re-evaluate the original task using this update. "
                    f"{validator.RESPONSE_CONTRACT}"
                ),
                "new_structure_attestation": True,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _prepare_temp_data_dir(path: Path) -> None:
    path.mkdir()
    families = _families()
    _write_jsonl(path / "family_cards24.jsonl", families)
    _write_jsonl(path / "m2_2b_v7_heldout48_base.jsonl", _bases())
    _write_jsonl(
        path / "development_overlap_audit24.jsonl",
        [
            {
                "family_id": card["family_id"],
                "nearest_development_families": card[
                    "nearest_development_families"
                ],
                "initial_key_fact_distinct": True,
                "answer_mechanism_distinct": True,
                "v_node_distinct": True,
                "decision": "approved",
                "rationale": f"Synthetic approval for {card['family_id']}.",
                "family_card_sha256": validator.canonical_sha256(card),
            }
            for card in families
        ],
    )
    for name in (
        "FAMILY_BLUEPRINT.md",
        "README.md",
        "build_m2_2b_v7_heldout.py",
        "validate_m2_2b_v7_heldout.py",
    ):
        (path / name).write_text(f"synthetic {name}\n", encoding="utf-8", newline="\n")
    (path / "DEVELOPMENT_OVERLAP_MATRIX.md").write_text(
        "# approved synthetic overlap matrix\n"
        + "\n".join(f"C1V7-F{index:02d}: approved" for index in range(1, 25))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_review_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=validator.REVIEW_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _complete_review_sheets(
    data_dir: Path,
    bases: list[dict[str, Any]],
    *,
    wrong_gold_items: int = 0,
    boolean_overrides: dict[str, str] | None = None,
) -> None:
    key_rows = validator.load_jsonl(data_dir / "private" / "blind_review_key.jsonl")
    key_by_id = {row["blind_item_id"]: row for row in key_rows}
    base_by_id = {row["base_id"]: row for row in bases}
    wrong_ids = set(sorted(key_by_id)[:wrong_gold_items])
    yes_fields = {
        "original_unique",
        "v_uniquely_supports_target",
        "ao_pure_target_exposure",
        "s_social_pressure_no_evidence",
        "n_no_target_hint",
        "distractor_always_wrong",
        "mapping_correct",
        "language_natural",
    }
    for reviewer in ("a", "b"):
        path = data_dir / f"blind_review_{reviewer}.csv"
        rows = validator.read_review_sheet(path)
        for row in rows:
            base = base_by_id[key_by_id[row["blind_item_id"]]["base_id"]]
            initial = base["initial_answer"]
            if row["blind_item_id"] in wrong_ids:
                initial = next(label for label in validator.LABELS if label != initial)
            row["original_supported_answer"] = initial
            row["v_supported_answer"] = base["target_answer"]
            for field in validator.BOOLEAN_REVIEW_FIELDS:
                row[field] = "yes" if field in yes_fields else "no"
            for field, value in (boolean_overrides or {}).items():
                row[field] = value
            row["confidence_1_5"] = "5"
        _write_review_sheet(path, rows)


def _write_final_labels_from_review_a(data_dir: Path) -> Path:
    rows = validator.read_review_sheet(data_dir / "blind_review_a.csv")
    path = data_dir / "private" / "final_review_labels.jsonl"
    _write_jsonl(
        path,
        [
            {field: row[field] for field in validator.FINAL_REVIEW_FIELDS}
            for row in rows
        ],
    )
    return path


def test_balanced_synthetic_schema_and_expansion() -> None:
    cards = _families()
    bases = _bases()
    family_report = validator.validate_family_cards(cards)
    base_report = validator.validate_base_cards(bases, cards)
    arms = validator.expand_arms(bases)
    arm_report = validator.validate_flat_arms(arms, bases)
    assert family_report["families"] == 24
    assert base_report["bases"] == 48
    assert set(base_report["migrations"].values()) == {8}
    assert arm_report == {"rows": 192, "arms": {"AO": 48, "N": 48, "S": 48, "V": 48}}
    rendered = validator.render_frozen_arms(bases[0])
    target = bases[0]["target_answer"]
    target_text = bases[0]["options"][target]
    assert rendered["N"]["followup"] == (
        f"{validator.N_TEMPLATE} {validator.RESPONSE_CONTRACT}"
    )
    assert rendered["AO"]["followup"] == (
        f"{validator.AO_TEMPLATE.format(target=target, target_text=target_text)} "
        f"{validator.RESPONSE_CONTRACT}"
    )
    assert rendered["S"]["followup"] == (
        f"{validator.S_TEMPLATE.format(target=target, target_text=target_text)} "
        f"{validator.RESPONSE_CONTRACT}"
    )


@pytest.mark.parametrize(
    "field",
    ["arms", "N", "ao", "S", "n_followup", "ao_followup", "s_followup"],
)
def test_custom_n_ao_s_treatment_is_rejected(field: str) -> None:
    cards = _families()
    bases = copy.deepcopy(_bases())
    bases[0][field] = "custom treatment"
    with pytest.raises(validator.ValidationError, match="custom N/AO/S treatment fields"):
        validator.validate_base_cards(bases, cards)


def test_builder_is_deterministic_without_review_packet(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(left)
    _prepare_temp_data_dir(right)
    left_manifest = builder.build(left, development_root=development, prepare_review=False)
    right_manifest = builder.build(right, development_root=development, prepare_review=False)
    assert left_manifest == right_manifest
    for name in (
        "m2_2b_v7_heldout192_arms.jsonl",
        "construction_audit48.jsonl",
        "leakage_report.json",
        "manifest.json",
    ):
        assert builder.sha256(left / name) == builder.sha256(right / name)


def test_review_packet_is_blank_and_private_key_is_separate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(data_dir)
    builder.build(data_dir, development_root=development, prepare_review=True)
    assert (data_dir / "private" / "blind_review_key.jsonl").exists()
    binding = validator.validate_review_key_binding(
        data_dir / "private" / "blind_review_key.jsonl",
        data_dir / "m2_2b_v7_heldout48_base.jsonl",
        _bases(),
    )
    assert binding["base_content_hashes_matched"] == 48
    for reviewer in ("a", "b"):
        report = validator.validate_review_sheet(
            data_dir / f"blind_review_{reviewer}.csv",
            require_complete=False,
        )
        assert report == {"rows": 48, "completed": 0, "status": "blank"}
        content = validator.validate_review_packet_content(
            data_dir / f"blind_review_{reviewer}.csv",
            data_dir / "private" / "blind_review_key.jsonl",
            _bases(),
        )
        assert content["visible_fields_matched"] == 48 * 6


def test_completed_independent_reviews_meet_frozen_agreement_gate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(data_dir)
    bases = _bases()
    builder.build(data_dir, development_root=development, prepare_review=True)
    _complete_review_sheets(data_dir, bases)
    report = validator.score_independent_reviews(
        data_dir / "blind_review_a.csv",
        data_dir / "blind_review_b.csv",
        data_dir / "private" / "blind_review_key.jsonl",
        bases,
    )
    assert report["status"] == "passed"
    assert report["raw_gold_answer_agreement"] == 1.0
    assert report["reviewer_gold_accuracy"] == {"a": 1.0, "b": 1.0}
    assert report["v_validity_kappa"] == 1.0
    assert report["s_ao_contamination_kappa"] == 1.0
    assert report["disputed_cells"] == 0


def test_each_reviewer_gold_accuracy_is_a_hard_gate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(data_dir)
    bases = _bases()
    builder.build(data_dir, development_root=development, prepare_review=True)
    _complete_review_sheets(data_dir, bases, wrong_gold_items=5)
    report = validator.score_independent_reviews(
        data_dir / "blind_review_a.csv",
        data_dir / "blind_review_b.csv",
        data_dir / "private" / "blind_review_key.jsonl",
        bases,
    )
    assert report["status"] == "failed"
    assert report["checks"]["reviewer_a_gold_accuracy_ge_0_95"] is False
    assert report["checks"]["reviewer_b_gold_accuracy_ge_0_95"] is False


def test_agreed_bad_final_label_cannot_pass_freeze(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(data_dir)
    bases = _bases()
    builder.build(data_dir, development_root=development, prepare_review=True)
    _complete_review_sheets(
        data_dir,
        bases,
        boolean_overrides={"development_isomorphic": "yes"},
    )
    final_path = _write_final_labels_from_review_a(data_dir)
    with pytest.raises(validator.ValidationError, match="fails acceptance contract"):
        validator.validate_final_review_labels(
            final_path,
            data_dir / "blind_review_a.csv",
            data_dir / "blind_review_b.csv",
            data_dir / "private" / "blind_review_key.jsonl",
            bases,
        )


def test_good_final_labels_pass_acceptance_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(data_dir)
    bases = _bases()
    builder.build(data_dir, development_root=development, prepare_review=True)
    _complete_review_sheets(data_dir, bases)
    report = validator.validate_final_review_labels(
        _write_final_labels_from_review_a(data_dir),
        data_dir / "blind_review_a.csv",
        data_dir / "blind_review_b.csv",
        data_dir / "private" / "blind_review_key.jsonl",
        bases,
    )
    assert report == {
        "rows": 48,
        "accepted_rows": 48,
        "disputed_cells_resolved": 0,
        "status": "passed",
    }


def test_review_binding_rejects_changed_base_content(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(data_dir)
    builder.build(data_dir, development_root=development, prepare_review=True)
    changed = copy.deepcopy(_bases())
    changed[0]["original_gold_rationale"] = "Changed after blind review."
    with pytest.raises(validator.ValidationError, match="base content hash differs"):
        validator.validate_review_key_binding(
            data_dir / "private" / "blind_review_key.jsonl",
            data_dir / "m2_2b_v7_heldout48_base.jsonl",
            changed,
        )


def test_review_binding_rejects_changed_dataset_bytes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    development = tmp_path / "development"
    development.mkdir()
    _prepare_temp_data_dir(data_dir)
    builder.build(data_dir, development_root=development, prepare_review=True)
    bases_path = data_dir / "m2_2b_v7_heldout48_base.jsonl"
    bases_path.write_text(
        bases_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(validator.ValidationError, match="dataset48 hash differs"):
        validator.validate_review_key_binding(
            data_dir / "private" / "blind_review_key.jsonl",
            bases_path,
            validator.load_jsonl(bases_path),
        )


def test_overlap_audit_is_bound_to_current_family_card(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _prepare_temp_data_dir(data_dir)
    cards = _families()
    audit_path = data_dir / "development_overlap_audit24.jsonl"
    rows = validator.load_jsonl(audit_path)
    rows[0]["family_card_sha256"] = "0" * 64
    _write_jsonl(audit_path, rows)
    with pytest.raises(validator.ValidationError, match="family-card hash mismatch"):
        validator.validate_overlap_audit(
            audit_path,
            cards,
            require_approved=True,
        )


def test_gate_a_family_robustness_constants_are_frozen() -> None:
    assert validator.MIN_HARMFUL_FAMILIES == 4
    assert validator.MAX_SINGLE_FAMILY_HARMFUL_SHARE == 0.40
    assert validator.REQUIRE_POSITIVE_LOFO is True


def test_repository_overlap_template_is_explicitly_pending() -> None:
    report = validator.validate_overlap_matrix(
        DATA_DIR / "DEVELOPMENT_OVERLAP_MATRIX.md",
        require_approved=False,
    )
    assert report == {"families_listed": 24, "status": "pending"}
    with pytest.raises(validator.ValidationError, match="still pending"):
        validator.validate_overlap_matrix(
            DATA_DIR / "DEVELOPMENT_OVERLAP_MATRIX.md",
            require_approved=True,
        )
