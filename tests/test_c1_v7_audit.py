from __future__ import annotations

from scripts import audit_c1_v7_heldout as audit


def result(item_id: str, value: float) -> dict:
    return {
        "id": item_id,
        "actual_answer_logits": {
            "logits": {"A": value, "B": value + 1, "C": value + 2}
        },
        "scaffold_answer_logits": None,
    }


def test_max_logit_difference_is_paired_by_id() -> None:
    left = [result("b", 1.0), result("a", 2.0)]
    right = [result("a", 2.00001), result("b", 1.00002)]
    assert audit.max_logit_difference(left, right) < 1e-4


def test_comparable_configuration_ignores_only_phase_and_condition() -> None:
    left = {
        "kind": "x",
        "scope": "y",
        "phase": "formal-uncapped",
        "condition": "uncapped",
        "selection": "full",
        "seed": 0,
        "dataset": {"sha256": "a"},
        "model": {"revision": "r"},
        "generation": {"do_sample": False},
        "axis_telemetry": {"layers": [46]},
        "review_deviation": "single",
        "git": {"commit": "c"},
    }
    right = {**left, "phase": "formal-p25", "condition": "p25"}
    assert audit.comparable_configuration(left) == audit.comparable_configuration(right)


def test_p25_cap_stats_complete_reads_nested_by_layer_schema() -> None:
    by_layer = {
        str(layer): {
            "total_positions": 10,
            "triggered_positions": 5,
            "projection_correction_sum": 1.0,
        }
        for layer in range(46, 54)
    }
    row = {
        "cap_stats": {
            "by_layer": by_layer,
            "pooled": {
                "total_positions": 80,
                "triggered_positions": 40,
                "projection_correction_sum": 8.0,
            },
            "definition": "synthetic nested cap stats",
        }
    }
    assert audit.p25_cap_stats_complete(row) is True


def test_p25_cap_stats_complete_rejects_missing_layer() -> None:
    by_layer = {
        str(layer): {"total_positions": 10, "projection_correction_sum": 1.0}
        for layer in range(46, 53)
    }
    assert audit.p25_cap_stats_complete({"cap_stats": {"by_layer": by_layer}}) is False
