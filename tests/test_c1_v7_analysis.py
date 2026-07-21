from __future__ import annotations

from scripts import analyze_c1_v7_heldout as analysis


def synthetic_pairs() -> list[dict]:
    pairs = []
    for family_index in range(6):
        family = f"F{family_index}"
        for variant in range(2):
            base = f"{family}-{variant}"
            for arm in analysis.ARMS:
                uncapped = "target" if arm == "V" else "initial"
                p25 = uncapped
                if arm == "S" and family_index < 4 and variant == 0:
                    p25 = "target"
                pairs.append(
                    {
                        "id": f"{base}-{arm}",
                        "base_id": base,
                        "family_id": family,
                        "arm": arm,
                        "strict_uncapped": uncapped,
                        "strict_p25": p25,
                        "lenient_uncapped": uncapped,
                        "lenient_p25": p25,
                    }
                )
    return pairs


def test_estimands_separate_social_from_ao_and_preserve_vn() -> None:
    values = analysis.estimands(synthetic_pairs(), "strict")
    assert values["delta_S"] == 4 / 12
    assert values["delta_AO"] == 0
    assert values["delta_social"] == 4 / 12
    assert values["delta_V"] == 0
    assert values["delta_N"] == 0


def test_cluster_bootstrap_and_gate_use_family_units() -> None:
    pairs = synthetic_pairs()
    bootstrap = analysis.family_cluster_bootstrap(
        pairs, scoring="strict", samples=500, seed=7
    )
    diagnostics = analysis.gate_diagnostics(
        pairs, bootstrap, analysis.estimands(pairs, "lenient")
    )
    assert bootstrap["delta_S"]["estimate"] == 4 / 12
    assert diagnostics["harmful_family_count"] == 4
    assert diagnostics["gate_a_checks"]["delta_S_ge_10pp"] is True


def test_transition_table_retains_other_wrong_and_invalid() -> None:
    pairs = synthetic_pairs()
    pairs[0]["strict_p25"] = "other_wrong"
    pairs[4]["strict_p25"] = "invalid_or_truncated"
    table = analysis.transitions(pairs, "N", "strict")
    assert table["initial->other_wrong"] == 1
    assert table["initial->invalid_or_truncated"] == 1


def test_telemetry_summary_reads_nested_cap_stats_by_layer() -> None:
    pairs = [
        {
            "arm": "S",
            "p25": {
                "cap_stats": {
                    "by_layer": {
                        "46": {
                            "total_positions": 10,
                            "triggered_positions": 4,
                            "projection_correction_sum": 2.5,
                        },
                        "47": {
                            "total_positions": 10,
                            "triggered_positions": 5,
                            "projection_correction_sum": 3.5,
                        },
                    },
                    "pooled": {"total_positions": 20},
                    "definition": "synthetic nested cap stats",
                }
            },
            "uncapped": {
                "axis_telemetry": {
                    "phases": {
                        "prefill": {
                            "reference_cap_exposure": {
                                "p25": {
                                    "46": {
                                        "count": 3,
                                        "triggered": 2,
                                        "correction_sum": 1.25,
                                    }
                                }
                            }
                        }
                    }
                }
            },
        }
    ]
    report = analysis.telemetry_summaries(pairs)
    assert report["actual_p25"]["S"] == {
        "total_positions": 20,
        "triggered_positions": 9,
        "trigger_rate": 0.45,
        "correction_sum": 6.0,
    }
    assert report["uncapped_counterfactual_p25"]["S"]["prefill"] == {
        "total_positions": 3,
        "triggered_positions": 2,
        "trigger_rate": 2 / 3,
        "correction_sum": 1.25,
    }
