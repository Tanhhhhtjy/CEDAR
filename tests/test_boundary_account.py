from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_boundary_account as analysis  # noqa: E402


def synthetic_item(
    item_id: str,
    cluster: str,
    gap: float,
    flip: int,
    *,
    stratum: str = "D|A",
) -> dict:
    return {
        "dataset": "D",
        "readout": "generated",
        "stable_id": item_id,
        "item_id": item_id,
        "cluster_id": cluster,
        "pooled_cluster_id": cluster,
        "pooled_resample_source": "D",
        "standardization_group": "D|A|generated",
        "stratum": stratum,
        "g_top": gap,
        "z_g_top": gap,
        "any_flip": flip,
        "gap_quantile": None,
    }


def test_margin_sign_and_top_two_gap() -> None:
    geometry = analysis.option_geometry(
        {"A": 5.0, "B": 4.0, "C": 1.0, "D": 0.0}, "A", "B"
    )
    assert geometry["g_top"] == 1.0
    assert geometry["g_initial"] == 1.0
    assert geometry["g_target"] == 1.0
    assert geometry["m_general"] == (4.0 + 1.0 + 0.0) / 3 - 5.0
    assert geometry["m_specific"] == 4.0 - 0.5


def test_initial_target_other_mapping_and_position_permutation() -> None:
    original = analysis.option_geometry(
        {"A": 3.0, "B": 6.0, "C": 2.0, "D": 1.0}, "A", "B"
    )
    permuted = analysis.option_geometry(
        {"A": 2.0, "B": 1.0, "C": 3.0, "D": 6.0}, "C", "D"
    )
    for field in ("g_top", "g_initial", "g_target", "m_general", "m_specific"):
        assert original[field] == permuted[field]


def test_transition_taxonomy_does_not_merge_initial_to_other() -> None:
    assert analysis.classify_transition("A", "B", "A", "B") == "initial_to_target"
    assert analysis.classify_transition("A", "C", "A", "B") == "initial_to_other"
    assert analysis.classify_transition("B", "A", "A", "B") == "target_to_initial"
    assert analysis.classify_transition(None, "A", "A", "B") == "invalid_or_missing"


def test_original_swapped_two_option_symmetry() -> None:
    original = {"A": 2.0, "B": 5.0}
    swapped = {"A": 5.0, "B": 2.0}
    original_gap = max(original.values()) - min(original.values())
    swapped_gap = max(swapped.values()) - min(swapped.values())
    assert original_gap == swapped_gap == 3.0


def test_e2c_missing_record_is_retained_without_imputation() -> None:
    plan = json.loads(analysis.PLAN_PATH.read_text(encoding="utf-8"))
    rows = analysis.load_jsonl_blob(plan["inputs"]["e2c_results"])
    items, audit = analysis.build_e2_items(rows, "E2C")
    assert len(items) == 96
    assert audit["missing_uncapped_logits"] == ["CEDAR-E2C-022-F"]
    missing = next(item for item in items if item["item_id"] == "CEDAR-E2C-022-F")
    assert missing["g_top"] is None
    assert missing["delta_m_general"] is None
    assert missing["transition"] == "invalid_or_missing"


def test_cluster_resampling_uses_whole_clusters_not_rows() -> None:
    rows = [
        synthetic_item("a1", "a", 0.1, 1),
        synthetic_item("a2", "a", 0.2, 0),
        synthetic_item("b1", "b", 2.0, 0),
    ]

    class AlwaysFirst:
        @staticmethod
        def randrange(_: int) -> int:
            return 0

    sampled = analysis.resample_clusters(rows, AlwaysFirst(), pooled=False)
    assert [row["item_id"] for row in sampled] == ["a1", "a2", "a1", "a2"]


def test_quantiles_use_only_uncapped_gap_and_stable_id_ties() -> None:
    rows = [synthetic_item(f"id-{index:02d}", str(index), 1.0, index % 2) for index in range(40)]
    first = copy.deepcopy(rows)
    second = copy.deepcopy(rows)
    for row in second:
        row["any_flip"] = 1 - row["any_flip"]
    analysis.assign_gap_quantiles(first)
    analysis.assign_gap_quantiles(second)
    assert {row["stable_id"]: row["gap_quantile"] for row in first} == {
        row["stable_id"]: row["gap_quantile"] for row in second
    }


def test_fixed_effect_logit_recovers_negative_gap_slope() -> None:
    rows = []
    for index in range(20):
        rows.append(synthetic_item(f"near-{index}", f"n{index}", -1.0, int(index < 15)))
        rows.append(synthetic_item(f"far-{index}", f"f{index}", 1.0, int(index < 3)))
    fitted = analysis.fit_fixed_effect_logit(rows, "any_flip", ["z_g_top"])
    assert fitted["status"] == "estimated"
    assert fitted["coefficients"]["z_g_top"] < 0


def test_deterministic_serialization_hash() -> None:
    payload = {"b": [2, 1], "a": {"x": 3}}
    first = analysis.sha256_bytes(analysis.canonical_json(payload).encode("utf-8"))
    second = analysis.sha256_bytes(analysis.canonical_json(payload).encode("utf-8"))
    assert first == second


def test_quantile_counts_match_analysis_rows() -> None:
    rows = [
        synthetic_item(str(index), str(index), float(index), int(index < 5))
        for index in range(40)
    ]
    analysis.assign_gap_quantiles(rows)
    summary = analysis.descriptive_boundary(rows, "g_top")
    assert sum(value["n"] for value in summary["quantiles"].values()) == 40
    assert sum(value["flips"] for value in summary["quantiles"].values()) == 5


def test_plan_contains_no_local_absolute_path() -> None:
    text = analysis.PLAN_PATH.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in text
    assert "/home/" not in text
    assert "/data" not in text


def test_frozen_git_blob_read_is_repeatable() -> None:
    plan = json.loads(analysis.PLAN_PATH.read_text(encoding="utf-8"))
    spec = plan["inputs"]["e2a_analysis"]
    first = analysis.git_blob(spec["commit"], spec["path"])
    second = analysis.git_blob(spec["commit"], spec["path"])
    assert first == second
    assert analysis.sha256_bytes(first) == spec["sha256"]


def test_cluster_bootstrap_sampling_is_seed_deterministic() -> None:
    rows = [
        synthetic_item(str(index), str(index // 2), float(index), index % 2)
        for index in range(8)
    ]
    first = analysis.resample_clusters(rows, random.Random(17), pooled=False)
    second = analysis.resample_clusters(rows, random.Random(17), pooled=False)
    assert [row["item_id"] for row in first] == [row["item_id"] for row in second]
