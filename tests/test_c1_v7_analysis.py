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
