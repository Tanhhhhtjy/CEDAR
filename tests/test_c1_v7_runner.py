from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_c1_v7_heldout.py"


def load_module():
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("c1_v7_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["c1_v7_runner"] = module
    spec.loader.exec_module(module)
    return module


def test_inputs_and_frozen_phase_selections() -> None:
    runner = load_module()
    rows = runner.validate_inputs()
    assert len(rows) == 192
    assert len(runner.select_rows(rows, runner.PHASES["smoke-nohook"])) == 8
    assert len(runner.select_rows(rows, runner.PHASES["smoke-monitor"])) == 8
    assert len(runner.select_rows(rows, runner.PHASES["smoke-p25"])) == 4
    assert len(runner.select_rows(rows, runner.PHASES["formal-uncapped"])) == 192
    assert len(runner.select_rows(rows, runner.PHASES["formal-p25"])) == 192


def test_answer_classification_and_lenient_parser() -> None:
    runner = load_module()
    row = {"initial_answer": "A", "target_answer": "B"}
    assert runner.classify_answer(row, "A", False) == "initial"
    assert runner.classify_answer(row, "B", False) == "target"
    assert runner.classify_answer(row, "C", False) == "other_wrong"
    assert runner.classify_answer(row, None, False) == "invalid_or_truncated"
    assert runner.lenient_final_answer('{"answer":"B","reason":"x"}') == ("B", None)
