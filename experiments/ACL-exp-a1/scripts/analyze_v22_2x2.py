#!/usr/bin/env python3
"""Reference analysis for the v2.2 additions (CPU only): same-operating-point 2x2, placebo, hard tier.

Reads results/v2_2/raw_records/*.jsonl.gz (v2.2 record schema) and writes
results/v2_2/summary_tables/v22_2x2_placebo_hard.json. summarize.py may import or re-implement it;
the definitions here are the frozen ones (configs/exp_a1_config.yaml -> analysis).

  no_think, per context_flip item, u = mean u_theta over all samples of the arm (prompt-only quantity):
    D_arg   = u(V) - u(F)
    D_shell = u(shell_only_ctxV) - u(shell_only);  D_AO = u(AO_ctxV) - u(AO)
    I_content       = D_arg - D_shell      (primary)      I_content_AO = D_arg - D_AO
    I_placebo       = [u(F_ctxPlacebo) - u(F)] - [u(shell_only_ctxPlacebo) - u(shell_only)]
  Decision (pre-registered): C_content localisable on context_flip dialog pairs iff
    median I_content > 0, sign consistency >= 0.70 in >= 4/6 fallacy classes, and
    median |I_placebo| < median |I_content|.
  Hard tier (think, eligible items, eval half): prompt_fails iff dialog_explicit_verify_F or F_sysverify follow >= 0.15.

Usage: python experiments/ACL-exp-a1/scripts/analyze_v22_2x2.py [--results experiments/ACL-exp-a1/results/v2_2]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import statistics
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eligible_items(recs: list[dict], dataset: str) -> set[str]:
    ok = collections.Counter()
    for r in recs:
        if r["dataset"] == dataset and r["arm"] in ("turn1_V", "turn1_F") and 0 <= r["sample_idx"] <= 3:
            ok[(r["base_id"], r["arm"])] += r["itt_outcome"] == "initial"
    ids = {r["base_id"] for r in recs if r["dataset"] == dataset}
    return {b for b in ids if ok[(b, "turn1_V")] >= 3 and ok[(b, "turn1_F")] >= 3}


def boot_ci(xs: list[float], n: int = 4000, seed: int = 12827) -> tuple[float, float]:
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    lo, hi = boot_ci(vals)
    return {"n": len(vals), "median": round(statistics.median(vals), 3), "ci95_median": [round(lo, 3), round(hi, 3)],
            "sign_consistency": round(sum(v > 0 for v in vals) / len(vals), 3)}


def two_by_two(recs: list[dict], dataset: str) -> dict:
    u = collections.defaultdict(list)
    meta = {}
    for r in recs:
        if r["dataset"] == dataset and r.get("u_theta") is not None and r.get("contract") == "final_only":
            u[(r["arm"], r["base_id"])].append(r["u_theta"])
            meta[r["base_id"]] = (r["pair_type"], r["fallacy_class"])
    U = {k: statistics.mean(v) for k, v in u.items()}
    need = ["V", "F", "shell_only", "shell_only_ctxV", "AO", "AO_ctxV", "F_ctxPlacebo", "shell_only_ctxPlacebo"]
    rows = []
    for b, (pt, fc) in meta.items():
        if pt != "context_flip" or not all((a, b) in U for a in need):
            continue
        d_arg = U[("V", b)] - U[("F", b)]
        rows.append({
            "base_id": b, "fallacy_class": fc,
            "I_content": d_arg - (U[("shell_only_ctxV", b)] - U[("shell_only", b)]),
            "I_content_AO": d_arg - (U[("AO_ctxV", b)] - U[("AO", b)]),
            "I_placebo": (U[("F_ctxPlacebo", b)] - U[("F", b)]) - (U[("shell_only_ctxPlacebo", b)] - U[("shell_only", b)]),
        })
    out = {k: summarize([r[k] for r in rows]) for k in ("I_content", "I_content_AO", "I_placebo")}
    by_fc = collections.defaultdict(list)
    for r in rows:
        by_fc[r["fallacy_class"]].append(r["I_content"])
    out["I_content_by_class"] = {fc: summarize(v) for fc, v in sorted(by_fc.items())}
    n_ok = sum(1 for v in out["I_content_by_class"].values() if v.get("n") and v["sign_consistency"] >= 0.70)
    abs_c = statistics.median([abs(r["I_content"]) for r in rows]) if rows else None
    abs_p = statistics.median([abs(r["I_placebo"]) for r in rows]) if rows else None
    out["decision_c_content_localisable_on_context_flip"] = bool(
        rows and out["I_content"]["median"] > 0 and n_ok >= 4 and abs_p < abs_c
    )
    out["classes_with_sign_consistency_ge_0.70"] = n_ok
    out["median_abs_I_content"], out["median_abs_I_placebo"] = abs_c, abs_p
    return out


def rates(recs: list[dict], dataset: str, arms: list[str], items: set[str]) -> dict:
    out = {}
    for a in arms:
        xs = [r for r in recs if r["dataset"] == dataset and r["arm"] == a and r["sample_idx"] >= 4 and r["base_id"] in items]
        if xs:
            out[a] = {"n": len(xs),
                      "p_target": round(sum(r["itt_outcome"] == "induced_target" for r in xs) / len(xs), 4),
                      "p_initial": round(sum(r["itt_outcome"] == "initial" for r in xs) / len(xs), 4),
                      "p_parse_fail": round(sum(r["parse_status"] != "ok" for r in xs) / len(xs), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(EXP / "results" / "v2_2"))
    args = ap.parse_args()
    res = Path(args.results)
    report: dict = {}
    arms6_7 = ["V", "F", "shell_only", "AO", "AO_ctxV", "shell_only_ctxV", "F_ctxPlacebo", "shell_only_ctxPlacebo",
               "F_sysverify", "V_sysverify", "V_hes_sysverify", "shell_only_sysverify", "dialog_explicit_verify_F",
               "two_call_pipeline_F", "verify_then_revise_prompt_F", "V_hes", "F_hes", "F_auth"]
    for path in sorted((res / "raw_records").glob("*.jsonl.gz")):
        cell = path.name.replace(".jsonl.gz", "")
        recs = load(path)
        entry = {}
        for ds in sorted({r["dataset"] for r in recs}):
            elig = eligible_items(recs, ds)
            e = {"n_eligible": len(elig), "rates_eligible": rates(recs, ds, arms6_7, elig)}
            if cell.endswith("__no_think") and ds != "symbolic_hard":
                e["two_by_two_no_think"] = two_by_two(recs, ds)
            if ds == "symbolic_hard" and cell.endswith("__think"):
                r = e["rates_eligible"]
                ev = r.get("dialog_explicit_verify_F", {}).get("p_target")
                sv = r.get("F_sysverify", {}).get("p_target")
                e["hard_tier_prompt_fails"] = None if ev is None or sv is None else bool(ev >= 0.15 or sv >= 0.15)
            entry[ds] = e
        report[cell] = entry
    out = res / "summary_tables" / "v22_2x2_placebo_hard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
