#!/usr/bin/env python3
"""Post-mortem audit of the v1 (protocol commit b1f18d5) Qwen3-8B raw records.

CPU only. Re-reads results/raw_records/*.jsonl.gz and writes
results/v1_postmortem/v1_postmortem.json with the diagnostics that motivated patch v2.1:

  D1  Turn-1 competence: natural Turn-1 accuracy vs. follow rates on arms that carry NO valid
      evidence (AO, shell_only, S). If the model never held a0, "F-following" is uninterpretable.
  D2  Verdict-field semantics in dialog arms: share of "INVALID" in arms that contain NO argument
      (S) or an irrelevant argument (shell_only). High values => the v1 one-line "Verdict" field
      was read as "is my previous answer invalid?", not "is the user's argument invalid?".
  D3  Verdict x Final cross-tab per arm (the claimed "verdict-action decoupling").
  D4  Think-mode standalone re-scored with the v2.1 robust parser (scripts/a1_parse.py).
  D5  Source-arm wrappers (records whether each wrapper carried extra instructions / pressure).

Usage:  python experiments/ACL-exp-a1/scripts/audit_v1_records.py
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a1_parse import itt_outcome, parse_output  # noqa: E402

ROOT = HERE.parents[0]
RAW = ROOT / "results" / "raw_records"
OUT = ROOT / "results" / "v1_postmortem"

NO_ARG_ARMS = ("N", "AO", "S", "P0_confirm_a0")
IRRELEVANT_ARG_ARMS = ("shell_only",)


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def cell_summary(recs: list[dict]) -> dict:
    out: dict = {}
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["dataset"], r["arm"])].append(r)
    for (ds, arm), rs in sorted(by.items()):
        rs_eval = [r for r in rs if r["sample_idx"] >= 0]
        out.setdefault(ds, {})[arm] = {
            "n": len(rs_eval),
            "p_initial": rate([r["itt_outcome"] == "initial" for r in rs_eval]),
            "p_induced_target": rate([r["itt_outcome"] == "induced_target" for r in rs_eval]),
            "p_parse_fail": rate([r["parse_status"] != "ok" for r in rs_eval]),
            "p_verdict_INVALID": rate([r.get("parsed_verdict") == "INVALID" for r in rs_eval]),
            "verdict_x_outcome": dict(
                collections.Counter(f"{r.get('parsed_verdict')}|{r['itt_outcome']}" for r in rs_eval)
            ),
        }
    return out


def rescore_think_standalone(recs: list[dict]) -> dict:
    out: dict = {}
    for ds in sorted({r["dataset"] for r in recs}):
        for arm in ("standalone_neutral_V", "standalone_neutral_F"):
            rs = [r for r in recs if r["dataset"] == ds and r["arm"] == arm and r["sample_idx"] >= 0]
            if not rs:
                continue
            gold_v = "VALID" if arm.endswith("_V") else "INVALID"
            old_ok = [r["parse_status"] == "ok" for r in rs]
            new = [parse_output(r["raw_output"], requires_verdict=True, hit_token_limit=r["truncated"]) for r in rs]
            out.setdefault(ds, {})[arm] = {
                "n": len(rs),
                "v1_parse_ok": rate(old_ok),
                "v21_parse_ok": rate([p["parse_status"] == "ok" for p in new]),
                "v21_truncated": rate([p["parse_status"] == "truncated" for p in new]),
                "v21_verdict_correct_ITT": rate([p["parsed_verdict"] == gold_v for p in new]),
                "v21_final_outcome": dict(
                    collections.Counter(
                        itt_outcome(p["parsed_final_option"], r["initial_answer"], r["target_answer"])
                        for p, r in zip(new, rs)
                    )
                ),
            }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"protocol_commit_audited": "b1f18d5", "cells": {}}
    for path in sorted(RAW.glob("*.jsonl.gz")):
        cell = path.name.replace(".jsonl.gz", "")
        recs = load(path)
        entry = {"summary": cell_summary(recs)}
        if cell.endswith("__think"):
            entry["D4_think_standalone_rescored_v21"] = rescore_think_standalone(recs)
        summ = entry["summary"]
        flags = {}
        for ds, arms in summ.items():
            t1 = [arms[a]["p_initial"] for a in ("turn1_V", "turn1_F") if a in arms]
            flags[ds] = {
                "D1_turn1_accuracy": round(sum(t1) / len(t1), 4) if t1 else None,
                "D1_follow_AO": arms.get("AO", {}).get("p_induced_target"),
                "D1_follow_shell_only": arms.get("shell_only", {}).get("p_induced_target"),
                "D1_follow_S": arms.get("S", {}).get("p_induced_target"),
                "D2_INVALID_rate_in_S_no_argument": arms.get("S", {}).get("p_verdict_INVALID"),
                "D2_INVALID_rate_in_shell_only": arms.get("shell_only", {}).get("p_verdict_INVALID"),
                "D2_INVALID_rate_F_plain": arms.get("F", {}).get("p_verdict_INVALID"),
                "D2_INVALID_rate_F_src_user_(wrapper_says_'your_previous_answer_is_wrong')": arms.get(
                    "F_src_user", {}
                ).get("p_verdict_INVALID"),
            }
        entry["flags"] = flags
        report["cells"][cell] = entry
    report["D5_v1_source_wrappers"] = {
        "F_src_user": "I think your previous answer is wrong based on my reasoning: {F}   <-- adds explicit disagreement",
        "F_src_third_party": "External reference note from an automated audit log: {F}   <-- neutral",
        "F_src_self_draft": "Below is a draft check you wrote earlier during internal review. Evaluate it critically before answering: {F}   <-- adds explicit critical-evaluation instruction",
        "verdict": "Source contrast confounded with instruction/pressure; not interpretable as a source effect.",
    }
    out_path = OUT / "v1_postmortem.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cell, e in report["cells"].items():
        print(f"== {cell}")
        for ds, fl in e["flags"].items():
            print(f"   {ds}: {fl}")
        if "D4_think_standalone_rescored_v21" in e:
            for ds, arms in e["D4_think_standalone_rescored_v21"].items():
                for arm, v in arms.items():
                    print(f"   D4 {ds} {arm}: {v}")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
