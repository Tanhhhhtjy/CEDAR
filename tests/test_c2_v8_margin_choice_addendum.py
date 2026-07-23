from __future__ import annotations

from scripts import analyze_c2_v8_margin_choice_addendum as addendum


def test_percentile_interpolates_endpoints_and_midpoint() -> None:
    values = [0.0, 1.0, 2.0, 3.0]
    assert addendum.percentile(values, 0.0) == 0.0
    assert addendum.percentile(values, 1.0) == 3.0
    assert addendum.percentile(values, 0.5) == 1.5


def test_stratified_bootstrap_is_deterministic() -> None:
    rows = [
        {"domain": "d1", "difficulty": "easy", "value": 0.0},
        {"domain": "d1", "difficulty": "easy", "value": 1.0},
        {"domain": "d2", "difficulty": "hard", "value": 2.0},
        {"domain": "d2", "difficulty": "hard", "value": 3.0},
    ]
    first = addendum.stratified_bootstrap(
        rows, lambda row: row["value"], samples=200, seed=17
    )
    second = addendum.stratified_bootstrap(
        rows, lambda row: row["value"], samples=200, seed=17
    )
    assert first == second
    assert first["estimate"] == 1.5


def test_category_distinguishes_initial_target_and_other() -> None:
    pair = {
        "id": "x",
        "initial_answer": "A",
        "target_answer": "B",
        "uncapped": {"answer_logits": {"option_argmax": "A"}},
        "p25": {"answer_logits": {"option_argmax": "B"}},
    }
    assert addendum.category(pair, "uncapped", "answer_logits") == "initial"
    assert addendum.category(pair, "p25", "answer_logits") == "target"
    pair["p25"]["answer_logits"]["option_argmax"] = "C"
    assert addendum.category(pair, "p25", "answer_logits") == "other_wrong"


def test_cluster_bootstrap_preserves_point_estimate() -> None:
    rows = [
        {"cluster": "a", "value": -1.0},
        {"cluster": "a", "value": -1.0},
        {"cluster": "b", "value": 1.0},
    ]
    result = addendum.cluster_bootstrap(
        rows,
        lambda row: row["value"],
        "cluster",
        samples=200,
        seed=19,
    )
    assert result["estimate"] == -1 / 3
    assert result["clusters"] == 2
