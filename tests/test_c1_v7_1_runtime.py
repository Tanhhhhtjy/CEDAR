from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from scripts import analyze_c1_v7_1_prompt_ablation as analysis

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_c1_v7_1_prompt_ablation.py"


def load_runner():
    if str(RUNNER_PATH.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER_PATH.parent))
    spec = importlib.util.spec_from_file_location("c1_v71_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["c1_v71_runner"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_pairs() -> list[dict]:
    pairs = []
    for family_index in range(6):
        for _variant in range(2):
            for arm in analysis.ARMS:
                cue_uncapped = "initial"
                cue_p25 = "initial"
                no_uncapped = "initial"
                no_p25 = "initial"
                if arm == "S":
                    no_uncapped = "target" if family_index < 2 else "initial"
                    no_p25 = "target" if family_index < 5 else "initial"
                pairs.append(
                    {
                        "arm": arm,
                        "family_id": f"F{family_index}",
                        "strict_cue_uncapped": cue_uncapped,
                        "strict_cue_p25": cue_p25,
                        "strict_no_cue_uncapped": no_uncapped,
                        "strict_no_cue_p25": no_p25,
                    }
                )
    return pairs


def test_selection_counts() -> None:
    runner = load_runner()
    runner.validate_inputs(runner.DATA_PATH, runner.PREFLIGHT_PATH)
    rows = runner.read_jsonl(runner.DATA_PATH)
    assert len(runner.select_rows(rows, runner.PHASES["formal-uncapped"])) == 144
    assert len(runner.select_rows(rows, runner.PHASES["smoke-nohook"])) == 3
    assert len(runner.select_rows(rows, runner.PHASES["smoke-p25"])) == 1


def test_estimands_separate_cue_and_cap_effects() -> None:
    values = analysis.estimands(synthetic_pairs(), "strict")
    assert values["cue_main_S_uncapped"] == 2 / 6
    assert values["cap_effect_S_no_cue"] == 3 / 6
    assert values["cap_effect_AO_no_cue"] == 0
    assert values["cap_social_no_cue"] == 3 / 6
    assert values["cue_by_cap_S"] == 3 / 6


def test_family_bootstrap_uses_family_units() -> None:
    report = analysis.family_bootstrap(synthetic_pairs(), "strict", samples=200, seed=9)
    assert report["cap_effect_S_no_cue"]["estimate"] == 3 / 6
