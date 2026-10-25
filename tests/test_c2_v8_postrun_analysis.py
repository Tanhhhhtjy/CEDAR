from __future__ import annotations

import math

from scripts import analyze_c2_v8_natural_confirmation as frozen
from scripts import analyze_c2_v8_natural_confirmation_postrun as corrected


def direct_record(margin: float) -> dict:
    return {"answer_logits": {"target_minus_initial": margin}}


def test_corrected_hierarchical_model_uses_identifiable_nested_effects() -> None:
    pairs = []
    labels = ("A", "B", "C", "D")
    for index in range(48):
        base = f"HB{index:02d}"
        common = {
            "base_id": base,
            "domain": f"D{index % 4}",
            "difficulty": ("easy", "medium", "hard")[index % 3],
            "template_id": (index // 4) % 4,
            "initial_answer": labels[index % 4],
            "target_answer": labels[(index + 1) % 4],
        }
        for arm in frozen.ARMS:
            baseline = -1.5 + 0.07 * (index % 9) + (0.03 if arm == "S" else 0)
            cap_delta = 0.45 + 0.02 * (index % 5)
            if arm == "S":
                cap_delta -= 0.70 + 0.01 * (index % 7)
            pairs.append(
                {
                    "id": f"{base}-{arm}",
                    **common,
                    "arm": arm,
                    "uncapped": direct_record(baseline),
                    "p25": direct_record(baseline + cap_delta),
                }
            )

    fitted = corrected.fit_hierarchical_model(pairs)
    assert fitted["status"] == "fitted"
    assert fitted["converged"] is True
    assert fitted["random_effects"] == ["(1 | domain)", "(1 | base)"]
    assert math.isfinite(fitted["interaction_estimate"])
    assert fitted["interaction_estimate"] < 0
