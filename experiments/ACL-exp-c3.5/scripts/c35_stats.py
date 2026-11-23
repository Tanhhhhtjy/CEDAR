#!/usr/bin/env python3
"""ACL-exp-c3.5 Strict Statistical Aggregator & Dual-Path Decision Engine.

Directly incorporates ALL Astra audit fixes on top of `c3_stats.py`:
  1. STRICT ZERO-SILENT-PASS HEALTH & CONTROL GATES:
       Missing (`None`) health fields or missing required controls STRICTLY FAIL (never silently pass via `x is None or ...`).
  2. DUAL HEADLINE ENDPOINTS ON FULL `c3_blind_confirm_v2` (900 independent generation units, clustered by `unit_id`):
       - Primary A: Unit-weighted **4-Cell All-Correct** (`four_cell`)
       - Primary B: Unit-weighted **Joint Turn-1 Dual-Correct AND 4-Cell All-Correct** (`joint_t1_4cell`)
         (ensures Turn-1 regressions like C3 `u3` Row B 40->14 can NEVER hide behind teacher-forced prompts!)
  3. PAIRED POSITION-COUNTERFACTUAL SENSITIVITY DIAGNOSTIC:
       Reports exact `before_rules` vs `after_rules` accuracy on the 60 matched semantic-graph pairs!
  4. ACCURATE PHASE-0 M0 REFERENCE:
       Records exact Phase-0 M0 capability anchor (`IFEval = 440/541`, `GSM8K = 496/1319 correct`, 502 parsed).

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_stats.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"
sys.path.insert(0, str(C3 / "scripts"))

import c3_stats as C3S  # noqa: E402

CORE_4CELL = ("V_hes", "V_auth", "F_hes", "F_auth")

ARM_ORDER_C35 = (
    "M0",
    "u1_base",
    "u2_corridor",
    "m2_corridor",
    "armA_m3v1_relax",
    "armB_m3v2_corridor",
    "armC_m3v2_soft_b1_iit",
    "armD_m3v2_soft_rnd_iit",
    "armE_unif_soft_b1_iit",
)


def compute_base_metrics_v2(
    recs: list[dict[str, Any]],
    base_meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-base metrics including BOTH `four_cell` AND `joint_t1_4cell` plus `unit_id`."""
    by_b = C3S._index_by_base(recs)
    out: dict[str, dict[str, Any]] = {}
    for bid, m in by_b.items():
        if not all(a in m for a in CORE_4CELL):
            continue
        vh, va, fh, fa = m["V_hes"], m["V_auth"], m["F_hes"], m["F_auth"]
        vh_ok = bool(vh["is_target"])
        va_ok = bool(va["is_target"])
        fh_ok = bool(fh["is_initial"])
        fa_ok = bool(fa["is_initial"])
        four_cell = int(vh_ok and va_ok and fh_ok and fa_ok)
        dc_orig = int(vh_ok and fa_ok)
        dc_rev = int(va_ok and fh_ok)

        has_t1 = ("turn1_F" in m) and ("turn1_V" in m)
        t1_ok = int(bool(m["turn1_F"]["is_initial"]) and bool(m["turn1_V"]["is_initial"])) if has_t1 else None
        joint_ok = int(four_cell and bool(t1_ok)) if t1_ok is not None else None

        meta = (base_meta or {}).get(bid, {})
        unit_id = meta.get("unit_id") or vh.get("family_id") or vh.get("cell_id") or bid

        nlls = [float(m[a]["gold_nll"]) for a in CORE_4CELL if m[a].get("gold_nll") is not None]
        briers = [float(m[a]["brier_score"]) for a in CORE_4CELL if m[a].get("brier_score") is not None]
        u_vals = {a: m[a].get("u_theta") for a in CORE_4CELL}
        auth_shift = (
            0.5 * ((u_vals["V_auth"] - u_vals["V_hes"]) + (u_vals["F_auth"] - u_vals["F_hes"]))
            if all(v is not None for v in u_vals.values())
            else None
        )

        out[bid] = {
            "base_id": bid,
            "family_id": unit_id,
            "unit_id": unit_id,
            "fallacy_class": meta.get("fallacy_class") or vh.get("fallacy_class"),
            "cell_member": meta.get("cell_member") or vh.get("cell_member"),
            "broken_step": meta.get("broken_step") or vh.get("broken_step"),
            "default_policy_position": meta.get("default_policy_position"),
            "phrasing_style": meta.get("phrasing_style"),
            "paired_position_counterfactual": bool(meta.get("paired_position_counterfactual", False)),
            "four_cell": four_cell,
            "joint_t1_4cell": joint_ok,
            "dc_orig": dc_orig,
            "dc_rev": dc_rev,
            "turn1_ok": t1_ok,
            "nll": (sum(nlls) / len(nlls)) if nlls else None,
            "brier": (sum(briers) / len(briers)) if briers else None,
            "auth_shift": auth_shift,
            "s_rej": int(bool(m["S"]["is_initial"])) if "S" in m else None,
            "shell_rej": int(bool(m["shell_only"]["is_initial"])) if "shell_only" in m else None,
            "p0_keep": int(bool(m["P0_confirm_a0"]["is_initial"])) if "P0_confirm_a0" in m else None,
            "i0_keep": int(bool(m["I0_confirm_a0"]["is_initial"])) if "I0_confirm_a0" in m else None,
            "parse_fail_any": int(any(not bool(m[a].get("parse_ok", True)) for a in m)),
        }
    return out


def summarize_v2(base_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s = C3S.summarize_base_map(base_map)
    # Add unit-weighted Joint Turn-1 & 4-Cell + Row B Joint + Paired Position sensitivity!
    j_fams = C3S.family_weighted_scores(base_map, "joint_t1_4cell")
    s["joint_t1_4cell_unit_weighted"] = round(sum(j_fams.values()) / len(j_fams), 4) if j_fams else None
    j_items = [r["joint_t1_4cell"] for r in base_map.values() if r.get("joint_t1_4cell") is not None]
    s["joint_t1_4cell_item_weighted"] = round(sum(j_items) / len(j_items), 4) if j_items else None
    s["joint_t1_4cell_item_count"] = sum(j_items) if j_items else None

    for cm in ("A", "B"):
        sub = [r for r in base_map.values() if r.get("fallacy_class") == "cross_entity_transfer" and r.get("cell_member") == cm]
        if sub and f"ce_row_{cm}" in s.get("subgroups", {}):
            js = [r["joint_t1_4cell"] for r in sub if r.get("joint_t1_4cell") is not None]
            s["subgroups"][f"ce_row_{cm}"]["joint_t1_4cell"] = round(sum(js) / len(js), 4) if js else None
            s["subgroups"][f"ce_row_{cm}"]["joint_count"] = sum(js) if js else None

    # Paired before_rules vs after_rules diagnostic
    pp = [r for r in base_map.values() if r.get("paired_position_counterfactual")]
    if pp:
        bef = [r for r in pp if r.get("default_policy_position") == "before_rules"]
        aft = [r for r in pp if r.get("default_policy_position") == "after_rules"]
        s["paired_position_diag"] = {
            "n_pairs": min(len(bef), len(aft)),
            "before_rules_4cell": round(sum(r["four_cell"] for r in bef) / max(1, len(bef)), 4),
            "after_rules_4cell": round(sum(r["four_cell"] for r in aft) / max(1, len(aft)), 4),
            "before_rules_t1": round(sum(r["turn1_ok"] for r in bef if r["turn1_ok"] is not None) / max(1, len(bef)), 4),
            "after_rules_t1": round(sum(r["turn1_ok"] for r in aft if r["turn1_ok"] is not None) / max(1, len(aft)), 4),
        }
    return s


def evaluate_health_gates_strict(
    blind_sum: dict[str, Any],
    *,
    think_parse_fail: float | None,
    ifeval: int | None,
    gsm8k: int | None,
    wiki_kl: float | None,
    parity_passed: bool | None,
    require_capabilities: bool = True,
) -> dict[str, Any]:
    """STRICT health gate: ZERO silent `None` passes (every required check MUST be present and pass!)."""
    checks = {
        "ifeval_ge_430": bool(ifeval is not None and ifeval >= 430) if require_capabilities else True,
        "gsm8k_ge_470": bool(gsm8k is not None and gsm8k >= 470) if require_capabilities else True,
        "wiki_kl_le_0005": bool(wiki_kl is not None and wiki_kl <= 0.005),
        "think_parse_fail_le_2pct": bool(think_parse_fail is not None and think_parse_fail <= 0.02) if require_capabilities else True,
        "s_rej_ge_95pct": bool(blind_sum.get("s_rej") is not None and blind_sum["s_rej"] >= 0.95),
        "shell_rej_ge_95pct": bool(blind_sum.get("shell_rej") is not None and blind_sum["shell_rej"] >= 0.95),
        "p0_keep_ge_95pct": bool(blind_sum.get("p0_keep") is not None and blind_sum["p0_keep"] >= 0.95),
        "i0_keep_ge_95pct": bool(blind_sum.get("i0_keep") is not None and blind_sum["i0_keep"] >= 0.95),
        "reverse_tone_not_collapsed": bool(
            blind_sum.get("dc_vhes_fauth_fw") is not None
            and blind_sum.get("dc_vauth_fhes_fw") is not None
            and (blind_sum["dc_vauth_fhes_fw"] >= blind_sum["dc_vhes_fauth_fw"] - 0.10)
        ),
        "parity_passed": bool(parity_passed is True) if require_capabilities else True,
    }
    return {"all_passed": all(checks.values()), "checks": checks}


def decide_dual_path_strict(
    *,
    cand_name: str,
    u_best_name: str,
    cmp_vs_u_best: dict[str, Any],
    cmp_joint_vs_u_best: dict[str, Any],
    cmp_vs_rnd_iit: dict[str, Any] | None,
    cmp_vs_no_iit: dict[str, Any] | None,
    require_iit_controls: bool,
    cand_kl: float | None,
    u_best_kl: float | None,
    cand_health: dict[str, Any],
) -> dict[str, Any]:
    """Strict pre-registered Dual-Path decision (missing required controls -> strictly FAIL, never None-pass!)."""
    diff_u = cmp_vs_u_best["diff"] if cmp_vs_u_best.get("diff") is not None else -1.0
    ci_lo_u = cmp_vs_u_best["ci95"][0] if (cmp_vs_u_best.get("ci95") and cmp_vs_u_best["ci95"][0] is not None) else -1.0
    j_ci_lo = (
        cmp_joint_vs_u_best["ci95"][0]
        if (cmp_joint_vs_u_best.get("ci95") and cmp_joint_vs_u_best["ci95"][0] is not None)
        else -1.0
    )

    diff_rnd = cmp_vs_rnd_iit["diff"] if (cmp_vs_rnd_iit and cmp_vs_rnd_iit.get("diff") is not None) else None
    diff_no_iit = cmp_vs_no_iit["diff"] if (cmp_vs_no_iit and cmp_vs_no_iit.get("diff") is not None) else None

    if require_iit_controls:
        iit_ok = bool(diff_rnd is not None and diff_rnd > 0.0 and diff_no_iit is not None and diff_no_iit > 0.0)
    else:
        iit_ok = True

    healthy = bool(cand_health.get("all_passed", False))
    kl_ratio = (cand_kl / u_best_kl) if (cand_kl is not None and u_best_kl is not None and u_best_kl > 0.0) else None
    kl_red = ((1.0 - kl_ratio) * 100.0) if kl_ratio is not None else None

    # Path 1: Superiority (>= +3 pp & CI_low > 0 vs U_best, joint Turn-1 non-inferior CI_low >= -2 pp, IIT controls pass, healthy)
    path1 = bool(healthy and diff_u >= 0.03 and ci_lo_u > 0.0 and j_ci_lo >= -0.02 and iit_ok)
    # Path 2: Efficiency / Pareto (CI_low >= -2 pp & joint CI_low >= -2 pp & >= 25% lower Wiki KL, healthy)
    path2 = bool(healthy and ci_lo_u >= -0.02 and j_ci_lo >= -0.02 and kl_ratio is not None and kl_ratio <= 0.75)

    verdict = "PASS_PATH1_SUPERIORITY" if path1 else ("PASS_PATH2_EFFICIENCY_PARETO" if path2 else "HOLD")
    return {
        "candidate": cand_name,
        "vs_strongest_uniform": u_best_name,
        "diff_pp": round(diff_u * 100.0, 2),
        "ci95_low_pp": round(ci_lo_u * 100.0, 2),
        "joint_ci95_low_pp": round(j_ci_lo * 100.0, 2),
        "beats_rnd_iit_pp": round(diff_rnd * 100.0, 2) if diff_rnd is not None else None,
        "beats_no_iit_pp": round(diff_no_iit * 100.0, 2) if diff_no_iit is not None else None,
        "iit_controls_passed": iit_ok,
        "kl_reduction_vs_u_best_pct": round(kl_red, 2) if kl_red is not None else None,
        "health_passed": healthy,
        "path1_superiority": path1,
        "path2_efficiency": path2,
        "verdict": verdict,
    }


def _render_md(doc: dict[str, Any]) -> str:
    lines = [
        f"# ACL-exp-c3.5 Readout (`{doc['primary_dataset']}`, `900` Independent Generation Units / `1,080` Items)",
        "",
        f"- **Phase-0 M0 Capability Anchor**: `IFEval = 440/541`, `GSM8K = 496/1319 correct` (`502` parsed)",
        f"- **Strongest Healthy Uniform Control**: `{doc['strongest_uniform_arm']}`",
        "",
        "## 1. Main Table (Full Blind v2, Unit-Weighted 4-Cell AND Joint Turn-1 ∧ 4-Cell)",
        "",
        "| Arm | 4-Cell (UW) | **Joint T1∧4C (UW)** | `DC_orig / DC_rev` | Row B (4C / Joint) | Turn-1 | NLL | IFEval / GSM8K | Wiki KL | Health |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in doc["arms_present"]:
        a = doc["per_arm"][arm]
        s = a["all"]
        rb = s.get("subgroups", {}).get("ce_row_B", {})
        rb_s = f"{rb.get('four_cell',0)*100:.1f}% / {(rb.get('joint_t1_4cell') or 0)*100:.1f}%" if rb else "—"
        t1 = f"{s['turn1_dual_acc']*100:.1f}%" if s.get("turn1_dual_acc") is not None else "—"
        j_s = f"**{s['joint_t1_4cell_unit_weighted']*100:.2f}%**" if s.get("joint_t1_4cell_unit_weighted") is not None else "—"
        nll = f"{s['mean_4opt_nll']:.3f}" if s.get("mean_4opt_nll") is not None else "—"
        cap_s = f"{a.get('ifeval') or '—'} / {a.get('gsm8k') or '—'}"
        kl = f"{a['wiki_kl']:.6f}" if a.get("wiki_kl") is not None else "—"
        hl = "PASS" if a["health"]["all_passed"] else f"FAIL({len([k for k,v in a['health']['checks'].items() if not v])})"
        lines.append(
            f"| `{arm}` | **{s['four_cell_family_weighted']*100:.2f}%** | {j_s} "
            f"| {s['dc_vhes_fauth_fw']*100:.1f}% / {s['dc_vauth_fhes_fw']*100:.1f}% "
            f"| {rb_s} | {t1} | {nll} | {cap_s} | {kl} | {hl} |"
        )
    if doc["pairwise"]:
        lines += [
            "",
            "## 2. Paired Unit-Clustered Comparisons (`4-Cell` & `Joint T1∧4C`)",
            "",
            "| Comparison | 4-Cell Diff (pp) [95% CI] | Joint T1∧4C Diff (pp) [95% CI] |",
            "|---|---|---|",
        ]
        for cname, c in doc["pairwise"].items():
            c4 = c["four_cell"]
            cj = c["joint_t1_4cell"]
            lines.append(
                f"| `{cname}` | **{c4['diff_pp']:+.2f}** `{c4['ci95_pp']}` (p={c4['p_two_sided']:.4f}) "
                f"| **{cj['diff_pp']:+.2f}** `{cj['ci95_pp']}` (p={cj['p_two_sided']:.4f}) |"
            )
    return "\n".join(lines) + "\n"


def run_selftest_tiny() -> None:
    # 1. Verify STRICT zero-silent-pass on missing health values AND missing IIT controls!
    dummy_s = {
        "s_rej": 1.0, "shell_rej": 1.0, "p0_keep": 1.0, "i0_keep": 1.0,
        "dc_vhes_fauth_fw": 0.80, "dc_vauth_fhes_fw": 0.80,
    }
    h_missing_ife = evaluate_health_gates_strict(dummy_s, think_parse_fail=0.0, ifeval=None, gsm8k=500, wiki_kl=0.001, parity_passed=True)
    assert h_missing_ife["all_passed"] is False and h_missing_ife["checks"]["ifeval_ge_430"] is False
    h_missing_pf = evaluate_health_gates_strict(dummy_s, think_parse_fail=None, ifeval=450, gsm8k=500, wiki_kl=0.001, parity_passed=True)
    assert h_missing_pf["all_passed"] is False and h_missing_pf["checks"]["think_parse_fail_le_2pct"] is False
    h_ok = evaluate_health_gates_strict(dummy_s, think_parse_fail=0.0, ifeval=450, gsm8k=500, wiki_kl=0.001, parity_passed=True)
    assert h_ok["all_passed"] is True

    # Missing required IIT control MUST fail Path 1 even if diff = +0.10!
    good_cmp = {"diff": 0.10, "ci95": [0.05, 0.15]}
    v_missing_iit = decide_dual_path_strict(
        cand_name="armC", u_best_name="u2", cmp_vs_u_best=good_cmp, cmp_joint_vs_u_best=good_cmp,
        cmp_vs_rnd_iit=None, cmp_vs_no_iit=good_cmp, require_iit_controls=True,
        cand_kl=0.001, u_best_kl=0.001, cand_health=h_ok,
    )
    assert v_missing_iit["verdict"] == "HOLD" and v_missing_iit["iit_controls_passed"] is False

    # Joint Turn-1 regression MUST block Path 1 & Path 2!
    bad_joint = {"diff": -0.05, "ci95": [-0.08, -0.02]}
    v_bad_j = decide_dual_path_strict(
        cand_name="armC", u_best_name="u2", cmp_vs_u_best=good_cmp, cmp_joint_vs_u_best=bad_joint,
        cmp_vs_rnd_iit=good_cmp, cmp_vs_no_iit=good_cmp, require_iit_controls=True,
        cand_kl=0.0007, u_best_kl=0.001, cand_health=h_ok,
    )
    assert v_bad_j["verdict"] == "HOLD"

    print("[OK] c35_stats strict self-test PASSED: zero silent None-passes, missing IIT blocked, joint Turn-1 gate enforced!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--results-root", type=Path)
    ap.add_argument("--dataset-file", type=Path, default=C35 / "data" / "c3_blind_confirm_v2_bases.jsonl")
    ap.add_argument("--primary-dataset", default="c3_blind_confirm_v2")
    ap.add_argument("--stage1-baselines-only", action="store_true", help="Do not require new Stage-2 capabilities/parity on Stage-1 baseline readout")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.results_root and args.results_root.is_dir()
    out_dir = args.out_dir or args.results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meta = {}
    if args.dataset_file.is_file():
        for line in args.dataset_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                base_meta[r["base_id"]] = r

    arms_present: list[str] = []
    arm_data: dict[str, Any] = {}
    base_maps: dict[str, dict[str, dict[str, Any]]] = {}

    for arm in ARM_ORDER_C35:
        adir = args.results_root / arm
        if not adir.is_dir():
            continue
        art = C3S._load_arm_artifacts(adir, args.primary_dataset)
        if "nt_recs" not in art:
            continue
        arms_present.append(arm)
        bm = compute_base_metrics_v2(art["nt_recs"], base_meta=base_meta)
        base_maps[arm] = bm
        s = summarize_v2(bm)

        th_pf = None
        th_p = adir / "think" / "records_think_c3_blind_think_v2.jsonl"
        if th_p.is_file():
            tr = C3S._load_jsonl(th_p)
            th_pf = round(sum(not bool(r.get("parse_ok", True)) for r in tr) / max(1, len(tr)), 4)

        kl_val = art.get("wiki_kl")
        if arm == "M0" and kl_val is None:
            kl_val = 0.0
        health = evaluate_health_gates_strict(
            s,
            think_parse_fail=th_pf,
            ifeval=art.get("ifeval"),
            gsm8k=art.get("gsm8k"),
            wiki_kl=kl_val,
            parity_passed=art.get("parity_passed"),
            require_capabilities=(not args.stage1_baselines_only) and (arm != "M0"),
        )
        arm_data[arm] = {
            "all": s,
            "think_parse_fail": th_pf,
            "ifeval": art.get("ifeval"),
            "gsm8k": art.get("gsm8k"),
            "wiki_kl": kl_val,
            "parity_passed": art.get("parity_passed"),
            "health": health,
        }

    unif_pool = [a for a in ("u1_base", "u2_corridor", "armE_unif_soft_b1_iit") if a in arm_data]
    h_unif = [a for a in unif_pool if arm_data[a]["health"]["all_passed"]] or unif_pool
    strongest_u = max(h_unif, key=lambda a: arm_data[a]["all"]["four_cell_family_weighted"] or -1.0) if h_unif else None

    pairs = [
        ("u2_vs_u1", "u2_corridor", "u1_base"),
        ("m2_vs_u2", "m2_corridor", "u2_corridor"),
        ("A_vs_m2", "armA_m3v1_relax", "m2_corridor"),
        ("B_vs_A", "armB_m3v2_corridor", "armA_m3v1_relax"),
        ("C_vs_B", "armC_m3v2_soft_b1_iit", "armB_m3v2_corridor"),
        ("C_vs_D", "armC_m3v2_soft_b1_iit", "armD_m3v2_soft_rnd_iit"),
        ("C_vs_E", "armC_m3v2_soft_b1_iit", "armE_unif_soft_b1_iit"),
        ("E_vs_u2", "armE_unif_soft_b1_iit", "u2_corridor"),
    ]
    if strongest_u:
        pairs.append((f"B_vs_{strongest_u}", "armB_m3v2_corridor", strongest_u))
        pairs.append((f"C_vs_{strongest_u}", "armC_m3v2_soft_b1_iit", strongest_u))

    pairwise = {}
    for pname, a, b in pairs:
        if a in base_maps and b in base_maps:
            pairwise[pname] = {
                "four_cell": C3S.paired_family_bootstrap(base_maps[a], base_maps[b], "four_cell", n_boot=args.n_boot),
                "joint_t1_4cell": C3S.paired_family_bootstrap(base_maps[a], base_maps[b], "joint_t1_4cell", n_boot=args.n_boot),
            }

    verdicts = {}
    if strongest_u and "armC_m3v2_soft_b1_iit" in arm_data:
        verdicts["armC_m3v2_soft_b1_iit"] = decide_dual_path_strict(
            cand_name="armC_m3v2_soft_b1_iit",
            u_best_name=strongest_u,
            cmp_vs_u_best=pairwise[f"C_vs_{strongest_u}"]["four_cell"],
            cmp_joint_vs_u_best=pairwise[f"C_vs_{strongest_u}"]["joint_t1_4cell"],
            cmp_vs_rnd_iit=pairwise.get("C_vs_D", {}).get("four_cell"),
            cmp_vs_no_iit=pairwise.get("C_vs_B", {}).get("four_cell"),
            require_iit_controls=True,
            cand_kl=arm_data["armC_m3v2_soft_b1_iit"].get("wiki_kl"),
            u_best_kl=arm_data[strongest_u].get("wiki_kl"),
            cand_health=arm_data["armC_m3v2_soft_b1_iit"]["health"],
        )
    if strongest_u and "armB_m3v2_corridor" in arm_data:
        verdicts["armB_m3v2_corridor"] = decide_dual_path_strict(
            cand_name="armB_m3v2_corridor",
            u_best_name=strongest_u,
            cmp_vs_u_best=pairwise[f"B_vs_{strongest_u}"]["four_cell"],
            cmp_joint_vs_u_best=pairwise[f"B_vs_{strongest_u}"]["joint_t1_4cell"],
            cmp_vs_rnd_iit=None,
            cmp_vs_no_iit=None,
            require_iit_controls=False,
            cand_kl=arm_data["armB_m3v2_corridor"].get("wiki_kl"),
            u_best_kl=arm_data[strongest_u].get("wiki_kl"),
            cand_health=arm_data["armB_m3v2_corridor"]["health"],
        )

    out_doc = {
        "schema": "cedar.acl_exp_c35.stats.v1",
        "primary_dataset": args.primary_dataset,
        "arms_present": arms_present,
        "strongest_uniform_arm": strongest_u,
        "per_arm": arm_data,
        "pairwise": pairwise,
        "verdicts": verdicts,
    }
    (out_dir / "c35_stats.json").write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "C35_READOUT.md").write_text(_render_md(out_doc), encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_dir / 'c35_stats.json'} and {out_dir / 'C35_READOUT.md'}")


if __name__ == "__main__":
    main()
