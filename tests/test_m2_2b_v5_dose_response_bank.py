from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "datasets" / "m2_2b_formal" / "v5_dose_response_bank"


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_candidate_bank_shapes_and_gradient() -> None:
    rows = load_jsonl(BANK / "m2_2b_v5_dose_response_candidate432.jsonl")
    assert len(rows) == 432
    assert Counter(row["arm"] for row in rows) == {"V": 144, "F": 144, "S": 144}
    assert len({row["source_base_id"] for row in rows}) == 36
    assert len({row["template_family"] for row in rows}) == 18
    for base_id in {row["source_base_id"] for row in rows}:
        group = [row for row in rows if row["source_base_id"] == base_id]
        for arm in ("V", "F", "S"):
            arm_rows = [row for row in group if row["arm"] == arm]
            assert {row["gradient_level"] for row in arm_rows} == {0, 1, 2, 3}
            assert len({row["followup_user"] for row in arm_rows}) == 4
    assert all(
        row["followup_user"].startswith("Observation:\nRecord update ")
        for row in rows
        if row["arm"] in {"V", "F"}
    )


def test_answer_parser_uses_final_json() -> None:
    module = load_module(
        "select_m2_2b_v5_dose_curve",
        BANK / "select_m2_2b_v5_dose_curve.py",
    )
    response = '<think>{"answer":"A"}</think>\n{"answer":"B","confidence":90,"reason":"x"}'
    assert module.parse_final_answer(response) == "B"


def test_family_beam_selector_can_hit_frozen_curve_exactly() -> None:
    module = load_module(
        "select_m2_2b_v5_dose_curve_beam",
        BANK / "select_m2_2b_v5_dose_curve.py",
    )
    robust = (2, 2, 2, 2, 2)
    thresholds = (
        (2, 0, 0, 0, 0),
        (2, 2, 0, 0, 0),
        (2, 2, 2, 0, 0),
        (2, 2, 2, 2, 0),
    )
    f_vectors = [(0, 0, 0, 0, 0)] * 9
    f_vectors += [thresholds[index] for index in range(4) for _ in range(2)]
    f_vectors += [(2, 2, 2, 2, 2)]
    v_vectors = [robust] * 10
    v_vectors += [thresholds[index] for index in range(4) for _ in range(2)]
    choices = {
        f"family_{index:02d}": {0: {"V": v_vectors[index], "F": f_vectors[index]}}
        for index in range(18)
    }
    selected, achieved, distance = module.select_levels(
        choices,
        targets={"V": module.TARGET_V, "F": module.TARGET_F},
    )
    assert len(selected) == 18
    assert achieved == {"V": module.TARGET_V, "F": module.TARGET_F}
    assert distance == 0


def test_selected_form_assembly_restores_five_arms() -> None:
    module = load_module(
        "select_m2_2b_v5_dose_curve_assembly",
        BANK / "select_m2_2b_v5_dose_curve.py",
    )
    candidates = load_jsonl(BANK / "m2_2b_v5_dose_response_candidate432.jsonl")
    families = {row["template_family"] for row in candidates}
    selected = module.assemble_selected(
        candidates,
        {family: 0 for family in families},
        {family: 0 for family in families},
    )
    assert len(selected) == 180
    assert Counter(row["arm"] for row in selected) == {
        "N": 36,
        "R": 36,
        "V": 36,
        "F": 36,
        "S": 36,
    }
    assert len({row["id"] for row in selected}) == 180


def test_five_condition_runner_accepts_the_frozen_bank() -> None:
    module = load_module(
        "run_m2_2b_v5_dose_bank",
        ROOT / "scripts" / "run_m2_2b_v5_dose_bank.py",
    )
    rows = module.load_rows(BANK / "m2_2b_v5_dose_response_candidate432.jsonl")
    assert len(rows) == 432
    assert module.CONDITIONS == ("uncapped", "p75", "p50", "p25", "p1")
    assert module.CONDITION_EXPERIMENTS["p1"] == "layers_46:54-p0.01"
