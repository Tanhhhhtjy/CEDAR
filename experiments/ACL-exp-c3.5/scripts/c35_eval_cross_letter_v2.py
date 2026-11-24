#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 5 Upgraded Cross-Letter & Cross-Outcome Causal Interchange Suite (8B & 14B Shared Probe 2 v2).

Directly implements EVERY methodological upgrade requested by Astra's Stage 4 Audit:
  1. BOTH DISJOINT LETTER PERMUTATIONS (`perm_0: a0<->o1, at<->o2` AND `perm_1: a0<->o2, at<->o1`):
       Eliminates lexicographic ordering artifacts in the remapped receptor option table!
  2. BIDIRECTIONAL SWAPS ACROSS BOTH `auth` AND `hes` TONES (+ `plain`):
       All 6 validity-flip pairs: `V<->F`, `V_auth<->F_auth`, `V_hes<->F_hes`.
  3. INCLUDES 3-HOP CE ROW B EXPLICITLY (`select_balanced_probe_v2_rows`, default n=96):
       24 CF (4 classes x 3 steps x 2), 24 AF (4 classes x 3 steps x 2),
       12 CE Row A (depth 2) + 12 CE Row A (depth 3), 12 CE Row B (depth 2) + 12 CE Row B (depth 3).
  4. CROSS-ITEM DIFFERENT-CONCLUSION-NAME PANEL (`cross_item_diff_outcome`):
       Pairs source base `i` with a receptor base `j` from a STRICTLY DIFFERENT domain where BOTH
       option letters `{src_a0, src_at} ∩ {tgt_a0, tgt_at} == ∅` AND semantic conclusion names
       `{src_outcomes} ∩ {tgt_outcomes} == ∅`! Tests whether B1 transfers abstract validity/correction
       decisions when NEITHER letter NOR semantic outcome string is shared!
  5. SAME-VALIDITY STAY CONTROL (`same_validity_stay_cross_letter` & `same_validity_stay_cross_item`):
       Patches `V->V`, `F->F`, `V_auth->V_auth`, `F_auth->F_auth` across disjoint-letter / cross-item
       receptors, expecting receptor to PRESERVE its own gold answer (rules out indiscriminate flipping)!
  6. COMMON-DENOMINATOR AGGREGATOR (`--aggregate-common`):
       Reports ALL swaps, per-model both-orig-ok swaps, AND exact shared-eligible common-denominator swaps
       across models (e.g. `u2_corridor`, `M3-v4-PureMid-MLP+`, `M3-v4-PureMid-MLP+-Random-Mask`)!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_v2.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-c1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_eval as C1EV  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c3_eval_interchange as IC3  # noqa: E402
import c35_modules as M35  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

OPTION_LETTERS = ("A", "B", "C", "D")

FLIP_PAIRS = (
    ("V", "F", +1.0, "plain"),
    ("F", "V", -1.0, "plain"),
    ("V_auth", "F_auth", +1.0, "auth"),
    ("F_auth", "V_auth", -1.0, "auth"),
    ("V_hes", "F_hes", +1.0, "hes"),
    ("F_hes", "V_hes", -1.0, "hes"),
)

STAY_PAIRS = (
    ("V", "V", +1.0, "plain"),
    ("F", "F", -1.0, "plain"),
    ("V_auth", "V_auth", +1.0, "auth"),
    ("F_auth", "F_auth", -1.0, "auth"),
)


def build_disjoint_remapped_row(row: dict[str, Any], perm_order: int = 0) -> tuple[dict[str, Any], dict[str, str]]:
    """Return a semantically identical copy of `row` with 100% disjoint `{a0, at}` letters under BOTH valid orders!

    `perm_order == 0`: `(a0 <-> o1, at <-> o2)` where `o1 < o2`
    `perm_order == 1`: `(a0 <-> o2, at <-> o1)` where `o1 < o2`
    """
    assert perm_order in (0, 1)
    a0, at = row["initial_answer"], row["target_answer"]
    others = [k for k in OPTION_LETTERS if k not in (a0, at)]
    assert len(others) == 2
    o1, o2 = (others[0], others[1]) if perm_order == 0 else (others[1], others[0])
    perm = {a0: o1, at: o2, o1: a0, o2: at}
    assert {perm[a0], perm[at]}.isdisjoint({a0, at})

    old_l2o = row["letter_to_outcome"]
    new_l2o = {perm[k]: v for k, v in old_l2o.items()}

    def _remap_ctx_options(ctx: str) -> str:
        out_lines = []
        for line in ctx.splitlines():
            m_dot = re.match(r"^([A-D])\.\s+.+$", line)
            m_par = re.match(r"^([A-D])\)\s+.+$", line)
            if m_dot:
                k = m_dot.group(1)
                out_lines.append(f"{k}. {new_l2o[k]}")
            elif m_par:
                k = m_par.group(1)
                out_lines.append(f"{k}) {new_l2o[k]}")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    def _remap_arm_letters(txt: str) -> str:
        return re.sub(r"\bOption ([A-D])\b", lambda m: f"Option {perm[m.group(1)]}", txt)

    new_row = copy.deepcopy(row)
    new_row["initial_answer"] = perm[a0]
    new_row["target_answer"] = perm[at]
    new_row["letter_to_outcome"] = new_l2o
    for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
        if k in new_row:
            new_row[k] = _remap_ctx_options(new_row[k])
    for arm_name, txt in list(new_row["arms"].items()):
        new_row["arms"][arm_name] = _remap_arm_letters(txt)

    assert new_row["letter_to_outcome"][new_row["initial_answer"]] == old_l2o[a0]
    assert new_row["letter_to_outcome"][new_row["target_answer"]] == old_l2o[at]
    assert {new_row["initial_answer"], new_row["target_answer"]}.isdisjoint({a0, at})
    return new_row, perm


def build_disjoint_target_for_source(tgt_row: dict[str, Any], src_row: dict[str, Any], perm_order: int = 0) -> dict[str, Any]:
    """Remap `tgt_row` so its `{a0, at}` is 100% DISJOINT from `src_row`'s `{a0, at}` under BOTH letter orders!"""
    assert perm_order in (0, 1)
    src_pair = {src_row["initial_answer"], src_row["target_answer"]}
    allowed = [k for k in OPTION_LETTERS if k not in src_pair]
    assert len(allowed) == 2
    t_a0, t_at = tgt_row["initial_answer"], tgt_row["target_answer"]
    des_a0, des_at = (allowed[0], allowed[1]) if perm_order == 0 else (allowed[1], allowed[0])
    # Construct bijection mapping t_a0 -> des_a0, t_at -> des_at, remaining 2 -> src_pair
    rem_src = [k for k in OPTION_LETTERS if k not in (t_a0, t_at)]
    rem_dst = sorted(src_pair)
    perm = {t_a0: des_a0, t_at: des_at, rem_src[0]: rem_dst[0], rem_src[1]: rem_dst[1]}
    assert len(set(perm.values())) == 4
    assert {perm[t_a0], perm[t_at]}.isdisjoint(src_pair)

    old_l2o = tgt_row["letter_to_outcome"]
    new_l2o = {perm[k]: v for k, v in old_l2o.items()}

    def _remap_ctx(ctx: str) -> str:
        out_lines = []
        for line in ctx.splitlines():
            m_dot = re.match(r"^([A-D])\.\s+.+$", line)
            m_par = re.match(r"^([A-D])\)\s+.+$", line)
            if m_dot:
                k = m_dot.group(1)
                out_lines.append(f"{k}. {new_l2o[k]}")
            elif m_par:
                k = m_par.group(1)
                out_lines.append(f"{k}) {new_l2o[k]}")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    new_row = copy.deepcopy(tgt_row)
    new_row["initial_answer"] = perm[t_a0]
    new_row["target_answer"] = perm[t_at]
    new_row["letter_to_outcome"] = new_l2o
    for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
        if k in new_row:
            new_row[k] = _remap_ctx(new_row[k])
    for arm_name, txt in list(new_row["arms"].items()):
        new_row["arms"][arm_name] = re.sub(r"\bOption ([A-D])\b", lambda m: f"Option {perm[m.group(1)]}", txt)

    # Hard assert BOTH disjoint option letters AND disjoint semantic outcome strings!
    assert {new_row["initial_answer"], new_row["target_answer"]}.isdisjoint(src_pair)
    src_outs = set(src_row["letter_to_outcome"].values())
    tgt_outs = set(new_row["letter_to_outcome"].values())
    assert src_outs.isdisjoint(tgt_outs), (src_row["domain"], tgt_row["domain"], src_outs & tgt_outs)
    return new_row


def build_7way_specs(m_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    b1 = m_info["iit_targets"]["b1"]
    rh = m_info["iit_targets"]["rnd_head_1"]
    rf = m_info["iit_targets"]["rnd_full_1"]
    return {
        "b1_full": {"ver_heads": b1["ver_only_heads"], "ver_mlps": b1["ver_mlps"], "form_heads": b1["form_only_heads"]},
        "b1_mlp_only": {"ver_heads": [], "ver_mlps": b1["ver_mlps"], "form_heads": []},
        "b1_heads_only": {"ver_heads": b1["ver_only_heads"], "ver_mlps": [], "form_heads": b1["form_only_heads"]},
        "b1_heads_rnd_mlp": {"ver_heads": b1["ver_only_heads"], "ver_mlps": rf["ver_mlps"], "form_heads": b1["form_only_heads"]},
        "rnd_heads_b1_mlp": {"ver_heads": rh["ver_only_heads"], "ver_mlps": b1["ver_mlps"], "form_heads": rh["form_only_heads"]},
        "rnd_full": {"ver_heads": rf["ver_only_heads"], "ver_mlps": rf["ver_mlps"], "form_heads": rf["form_only_heads"]},
    }


def _subgroup_tag(row: dict[str, Any]) -> str:
    if row["fallacy_class"] == "cross_entity_transfer":
        mem = row.get("cell_member", "A")
        dep = int(row.get("difficulty_depth", 2))
        return f"ce_row_{mem}_d{dep}"
    return str(row["pair_type"])


def select_balanced_probe_v2_rows(
    all_rows: list[dict[str, Any]],
    n_cf: int = 24,
    n_af: int = 24,
    n_ce_a_d2: int = 12,
    n_ce_a_d3: int = 12,
    n_ce_b_d2: int = 12,
    n_ce_b_d3: int = 12,
) -> list[dict[str, Any]]:
    """Stratified selection explicitly covering CF, AF, CE Row A (d2 & d3), AND 3-hop CE Row B (d2 & d3)!"""
    cf_pool = [r for r in all_rows if r["pair_type"] == "context_flip" and r["fallacy_class"] != "cross_entity_transfer" and not r.get("paired_position_counterfactual")]
    af_pool = [r for r in all_rows if r["pair_type"] == "argument_flip" and r["fallacy_class"] != "cross_entity_transfer" and not r.get("paired_position_counterfactual")]
    ce_a_d2 = [r for r in all_rows if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "A" and int(r.get("difficulty_depth", 2)) == 2]
    ce_a_d3 = [r for r in all_rows if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "A" and int(r.get("difficulty_depth", 2)) == 3]
    ce_b_d2 = [r for r in all_rows if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "B" and int(r.get("difficulty_depth", 2)) == 2]
    ce_b_d3 = [r for r in all_rows if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "B" and int(r.get("difficulty_depth", 2)) == 3]

    def _round_robin(pool: list[dict[str, Any]], take: int) -> list[dict[str, Any]]:
        by_k: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in pool:
            by_k[(r["fallacy_class"], r.get("broken_step"), r["domain"])].append(r)
        out: list[dict[str, Any]] = []
        ks = sorted(by_k)
        ptr = 0
        while len(out) < min(take, len(pool)):
            prog = False
            for k in ks:
                if ptr < len(by_k[k]) and len(out) < take:
                    out.append(by_k[k][ptr])
                    prog = True
            if not prog:
                break
            ptr += 1
        return out

    return (
        _round_robin(cf_pool, n_cf)
        + _round_robin(af_pool, n_af)
        + ce_a_d2[:n_ce_a_d2]
        + ce_a_d3[:n_ce_a_d3]
        + ce_b_d2[:n_ce_b_d2]
        + ce_b_d3[:n_ce_b_d3]
    )


def select_cross_item_pairs(rows: list[dict[str, Any]], n_pairs: int = 24) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deterministically pair source `i` with receptor `j` from a STRICTLY DIFFERENT domain (disjoint outcome names!)."""
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    # Prefer balanced non-CE + CE pairs across distinct domains
    for i, src in enumerate(rows):
        if len(out) >= n_pairs:
            break
        src_outs = set(src["letter_to_outcome"].values())
        for off in range(1, len(rows)):
            cand = rows[(i + off * 7) % len(rows)]
            if cand["domain"] != src["domain"] and src_outs.isdisjoint(set(cand["letter_to_outcome"].values())):
                out.append((src, cand))
                break
    assert len(out) == min(n_pairs, len(rows)), f"Could only build {len(out)} cross-item pairs"
    return out


def _summarize_records(records: list[dict[str, Any]], expect_flip: bool) -> dict[str, Any]:
    """Summarize records with ALL swaps, BOTH-ORIG-OK swaps, and subgroup/tone/perm_order breakdowns."""
    tot = len(records)
    both_ok = [r for r in records if r["both_orig_ok"]]
    n_ok = len(both_ok)
    key = "abstract_validity_flip" if expect_flip else "stay_own_gold"

    def _agg(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {"n": 0, "rate": None, "literal_copy_rate": None, "mean_signed_du": None}
        return {
            "n": n,
            "rate": round(sum(int(x[key]) for x in sub) / n, 4),
            "literal_copy_rate": round(sum(int(x["literal_letter_copy"]) for x in sub) / n, 4),
            "stay_rate": round(sum(int(x["stay_own_gold"]) for x in sub) / n, 4),
            "mean_signed_du": round(sum(float(x["signed_du"]) for x in sub) / n, 4),
        }

    by_sub = defaultdict(list)
    by_tone = defaultdict(list)
    by_perm = defaultdict(list)
    for r in both_ok:
        by_sub[r["subgroup"]].append(r)
        by_tone[r["tone"]].append(r)
        by_perm[f"perm_{r['perm_order']}"].append(r)

    return {
        "n_all": tot,
        "n_both_orig_ok": n_ok,
        "all_swaps": _agg(records),
        "both_orig_ok": _agg(both_ok),
        "by_subgroup_ok": {k: _agg(v) for k, v in sorted(by_sub.items())},
        "by_tone_ok": {k: _agg(v) for k, v in sorted(by_tone.items())},
        "by_perm_order_ok": {k: _agg(v) for k, v in sorted(by_perm.items())},
    }


def evaluate_cross_letter_v2_suite(
    model,
    tok,
    a1_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    specs_6: dict[str, dict[str, Any]],
    head_dim: int,
    opt_tok_ids: dict[str, int],
    *,
    n_cross_item: int = 24,
    k_max: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run full Probe 2 v2 suite across ALL 4 regimes:
      1. `cross_letter_flip` (same item, BOTH disjoint permutations `perm_0` & `perm_1`, 6 bidirectional flip pairs)
      2. `cross_letter_stay` (same item, BOTH disjoint permutations, 4 same-validity stay controls)
      3. `cross_item_diff_outcome_flip` (different domain + disjoint outcome names + BOTH disjoint letter perms, 6 flip pairs)
      4. `cross_item_diff_outcome_stay` (different domain + disjoint outcome names + BOTH disjoint letter perms, 4 stay controls)
    """
    arms_needed = ("V", "F", "V_auth", "V_hes", "F_auth", "F_hes")
    enc_orig: dict[str, dict[str, torch.Tensor]] = {}
    eval_orig: dict[str, dict[str, dict[str, Any]]] = {}
    enc_rm: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    eval_rm: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    rm_rows: dict[tuple[str, int], dict[str, Any]] = {}

    for row in rows:
        bid = row["base_id"]
        enc_orig[bid], eval_orig[bid] = {}, {}
        for arm in arms_needed:
            ids_o = B1L.encode_prompt(tok, build_messages(a1_cfg, row, arm, "no_think"), "Final:", enable_thinking=False)
            enc_orig[bid][arm] = ids_o
            eval_orig[bid][arm] = IC3._eval_4opt_logits(model, ids_o, opt_tok_ids, row["initial_answer"], row["target_answer"])
        for po in (0, 1):
            rm_r, _ = build_disjoint_remapped_row(row, perm_order=po)
            rm_rows[(bid, po)] = rm_r
            enc_rm[(bid, po)], eval_rm[(bid, po)] = {}, {}
            for arm in arms_needed:
                ids_r = B1L.encode_prompt(tok, build_messages(a1_cfg, rm_r, arm, "no_think"), "Final:", enable_thinking=False)
                enc_rm[(bid, po)][arm] = ids_r
                eval_rm[(bid, po)][arm] = IC3._eval_4opt_logits(model, ids_r, opt_tok_ids, rm_r["initial_answer"], rm_r["target_answer"])

    # Pre-encode cross-item different-outcome disjoint receptors (for BOTH perm orders!)
    ci_pairs = select_cross_item_pairs(rows, n_pairs=n_cross_item)
    ci_rows: dict[tuple[int, int], dict[str, Any]] = {}
    enc_ci: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    eval_ci: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for pi, (srow, trow) in enumerate(ci_pairs):
        for po in (0, 1):
            rem = build_disjoint_target_for_source(trow, srow, perm_order=po)
            ci_rows[(pi, po)] = rem
            enc_ci[(pi, po)], eval_ci[(pi, po)] = {}, {}
            for arm in arms_needed:
                ids_c = B1L.encode_prompt(tok, build_messages(a1_cfg, rem, arm, "no_think"), "Final:", enable_thinking=False)
                enc_ci[(pi, po)][arm] = ids_c
                eval_ci[(pi, po)][arm] = IC3._eval_4opt_logits(model, ids_c, opt_tok_ids, rem["initial_answer"], rem["target_answer"])

    # Identity control on b1_full (MUST be 100% exact!)
    b1_vh = [tuple(x) for x in specs_6["b1_full"]["ver_heads"]]
    b1_vm = [int(x) for x in specs_6["b1_full"]["ver_mlps"]]
    id_n = id_ok = 0
    id_max_du = 0.0
    for row in rows[: min(16, len(rows))]:
        bid = row["base_id"]
        for arm in ("V", "F"):
            res = IC3._interchange_4opt_logits(
                model, enc_orig[bid][arm], enc_orig[bid][arm],
                b1_vh, b1_vm, head_dim, opt_tok_ids, row["initial_answer"], row["target_answer"], k_max=k_max,
            )
            to = eval_orig[bid][arm]
            id_n += 1
            id_ok += int(res["pred"] == to["pred"])
            id_max_du = max(id_max_du, abs(res["u"] - to["u"]))
    assert id_ok == id_n and id_max_du < 1e-2, (id_ok, id_n, id_max_du)

    detailed: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {
        "identity_control": {"n_checks": id_n, "exact_match_rate": round(id_ok / max(1, id_n), 4), "max_abs_du": round(id_max_du, 6)},
        "cross_letter_flip": {},
        "cross_letter_stay": {},
        "cross_item_diff_outcome_flip": {},
        "cross_item_diff_outcome_stay": {},
    }

    for sname, sp in specs_6.items():
        vh = [tuple(x) for x in sp["ver_heads"]]
        vm = [int(x) for x in sp["ver_mlps"]]
        mode_recs: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Regimes 1 & 2: same-item disjoint cross-letter (BOTH perm orders!)
        for row in rows:
            bid = row["base_id"]
            src_a0, src_at = row["initial_answer"], row["target_answer"]
            sub_tag = _subgroup_tag(row)
            for po in (0, 1):
                trow = rm_rows[(bid, po)]
                tgt_a0, tgt_at = trow["initial_answer"], trow["target_answer"]
                assert {src_a0, src_at}.isdisjoint({tgt_a0, tgt_at})

                for mode, pairs_list, expect_flip in (
                    ("cross_letter_flip", FLIP_PAIRS, True),
                    ("cross_letter_stay", STAY_PAIRS, False),
                ):
                    for src_arm, tgt_arm, y_src, tone in pairs_list:
                        src_lit_gold = src_at if y_src > 0 else src_a0
                        if expect_flip:
                            tgt_orig_gold = tgt_a0 if y_src > 0 else tgt_at
                            tgt_cf_gold = tgt_at if y_src > 0 else tgt_a0
                        else:
                            tgt_orig_gold = tgt_at if y_src > 0 else tgt_a0
                            tgt_cf_gold = tgt_orig_gold

                        so = eval_orig[bid][src_arm]
                        to = eval_rm[(bid, po)][tgt_arm]
                        res = IC3._interchange_4opt_logits(
                            model, enc_orig[bid][src_arm], enc_rm[(bid, po)][tgt_arm],
                            vh, vm, head_dim, opt_tok_ids, tgt_a0, tgt_at, k_max=k_max,
                        )
                        src_ok = so["pred"] == src_lit_gold
                        tgt_ok = to["pred"] == tgt_orig_gold
                        rec = {
                            "swap_uid": f"{mode}|{bid}|{bid}|p{po}|{src_arm}->{tgt_arm}",
                            "mode": mode,
                            "spec": sname,
                            "src_base_id": bid,
                            "tgt_base_id": bid,
                            "subgroup": sub_tag,
                            "perm_order": po,
                            "tone": tone,
                            "src_arm": src_arm,
                            "tgt_arm": tgt_arm,
                            "src_lit_gold": src_lit_gold,
                            "tgt_orig_gold": tgt_orig_gold,
                            "tgt_cf_gold": tgt_cf_gold,
                            "patch_pred": res["pred"],
                            "both_orig_ok": bool(src_ok and tgt_ok),
                            "abstract_validity_flip": bool(expect_flip and res["pred"] == tgt_cf_gold),
                            "stay_own_gold": bool(res["pred"] == tgt_orig_gold),
                            "literal_letter_copy": bool(res["pred"] == src_lit_gold),
                            "signed_du": round(float(y_src * (res["u"] - to["u"])), 4),
                        }
                        mode_recs[mode].append(rec)
                        detailed.append(rec)

        # Regimes 3 & 4: cross-item DIFFERENT domain + DIFFERENT semantic outcome names + disjoint letters (BOTH perms!)
        for pi, (srow, trow_raw) in enumerate(ci_pairs):
            sbid = srow["base_id"]
            tbid = trow_raw["base_id"]
            src_a0, src_at = srow["initial_answer"], srow["target_answer"]
            sub_tag = f"{_subgroup_tag(srow)}->{_subgroup_tag(trow_raw)}"
            for po in (0, 1):
                trow = ci_rows[(pi, po)]
                tgt_a0, tgt_at = trow["initial_answer"], trow["target_answer"]
                assert {src_a0, src_at}.isdisjoint({tgt_a0, tgt_at})

                for mode, pairs_list, expect_flip in (
                    ("cross_item_diff_outcome_flip", FLIP_PAIRS, True),
                    ("cross_item_diff_outcome_stay", STAY_PAIRS, False),
                ):
                    for src_arm, tgt_arm, y_src, tone in pairs_list:
                        src_lit_gold = src_at if y_src > 0 else src_a0
                        if expect_flip:
                            tgt_orig_gold = tgt_a0 if y_src > 0 else tgt_at
                            tgt_cf_gold = tgt_at if y_src > 0 else tgt_a0
                        else:
                            tgt_orig_gold = tgt_at if y_src > 0 else tgt_a0
                            tgt_cf_gold = tgt_orig_gold

                        so = eval_orig[sbid][src_arm]
                        to = eval_ci[(pi, po)][tgt_arm]
                        res = IC3._interchange_4opt_logits(
                            model, enc_orig[sbid][src_arm], enc_ci[(pi, po)][tgt_arm],
                            vh, vm, head_dim, opt_tok_ids, tgt_a0, tgt_at, k_max=k_max,
                        )
                        src_ok = so["pred"] == src_lit_gold
                        tgt_ok = to["pred"] == tgt_orig_gold
                        rec = {
                            "swap_uid": f"{mode}|{sbid}|{tbid}|p{po}|{src_arm}->{tgt_arm}",
                            "mode": mode,
                            "spec": sname,
                            "src_base_id": sbid,
                            "tgt_base_id": tbid,
                            "subgroup": sub_tag,
                            "perm_order": po,
                            "tone": tone,
                            "src_arm": src_arm,
                            "tgt_arm": tgt_arm,
                            "src_lit_gold": src_lit_gold,
                            "tgt_orig_gold": tgt_orig_gold,
                            "tgt_cf_gold": tgt_cf_gold,
                            "patch_pred": res["pred"],
                            "both_orig_ok": bool(src_ok and tgt_ok),
                            "abstract_validity_flip": bool(expect_flip and res["pred"] == tgt_cf_gold),
                            "stay_own_gold": bool(res["pred"] == tgt_orig_gold),
                            "literal_letter_copy": bool(res["pred"] == src_lit_gold),
                            "signed_du": round(float(y_src * (res["u"] - to["u"])), 4),
                        }
                        mode_recs[mode].append(rec)
                        detailed.append(rec)

        for mode, expect_flip in (
            ("cross_letter_flip", True),
            ("cross_letter_stay", False),
            ("cross_item_diff_outcome_flip", True),
            ("cross_item_diff_outcome_stay", False),
        ):
            summaries[mode][sname] = _summarize_records(mode_recs[mode], expect_flip=expect_flip)

    return summaries, detailed


def aggregate_common_denominator(record_files: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    """Compute exact ALL-swaps, PER-MODEL-eligible, AND COMMON-DENOMINATOR-eligible comparisons across models."""
    loaded: dict[str, dict[tuple[str, str], dict[str, dict[str, Any]]]] = {}
    for alias, p in record_files.items():
        by_ms: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_ms[(r["mode"], r["spec"])][r["swap_uid"]] = r
        loaded[alias] = by_ms

    aliases = list(record_files.keys())
    all_keys = sorted(set().union(*(d.keys() for d in loaded.values())))
    report: dict[str, Any] = {"aliases": aliases, "modes": defaultdict(dict)}

    for mode, spec in all_keys:
        expect_flip = mode.endswith("_flip")
        metric_k = "abstract_validity_flip" if expect_flip else "stay_own_gold"
        # Common eligible swap_uids where EVERY model in `aliases` has `both_orig_ok == True`
        elig_sets = [
            {uid for uid, r in loaded[a].get((mode, spec), {}).items() if r["both_orig_ok"]}
            for a in aliases
        ]
        common_elig = sorted(set.intersection(*elig_sets)) if elig_sets else []
        per_model = {}
        for a in aliases:
            all_r = list(loaded[a].get((mode, spec), {}).values())
            ok_r = [r for r in all_r if r["both_orig_ok"]]
            cm_r = [loaded[a][(mode, spec)][uid] for uid in common_elig]
            per_model[a] = {
                "all_n": len(all_r),
                "all_rate": round(sum(int(r[metric_k]) for r in all_r) / max(1, len(all_r)), 4),
                "own_ok_n": len(ok_r),
                "own_ok_rate": round(sum(int(r[metric_k]) for r in ok_r) / max(1, len(ok_r)), 4) if ok_r else None,
                "own_ok_lit_copy": round(sum(int(r["literal_letter_copy"]) for r in ok_r) / max(1, len(ok_r)), 4) if ok_r else None,
                "common_ok_n": len(cm_r),
                "common_ok_rate": round(sum(int(r[metric_k]) for r in cm_r) / max(1, len(cm_r)), 4) if cm_r else None,
                "common_ok_lit_copy": round(sum(int(r["literal_letter_copy"]) for r in cm_r) / max(1, len(cm_r)), 4) if cm_r else None,
            }
        report["modes"][mode][spec] = {"n_common_eligible": len(common_elig), "per_model": per_model}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe2_v2_common_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def run_selftest_tiny() -> None:
    import tempfile
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_train as C1T

    rows_all = [json.loads(l) for l in (C35 / "data" / "c3_blind_confirm_v3_bases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = select_balanced_probe_v2_rows(rows_all, n_cf=4, n_af=4, n_ce_a_d2=2, n_ce_a_d3=2, n_ce_b_d2=2, n_ce_b_d3=2)
    assert len(picked) == 16
    assert any(r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "B" and r["difficulty_depth"] == 3 for r in picked)

    # Verify BOTH disjoint letter permutations produce DISTINCT, valid disjoint mappings!
    for r in picked:
        rm0, p0 = build_disjoint_remapped_row(r, perm_order=0)
        rm1, p1 = build_disjoint_remapped_row(r, perm_order=1)
        assert rm0["initial_answer"] != rm1["initial_answer"]
        assert rm0["target_answer"] != rm1["target_answer"]
        assert {rm0["initial_answer"], rm0["target_answer"]} == {rm1["initial_answer"], rm1["target_answer"]}
        assert {r["initial_answer"], r["target_answer"]}.isdisjoint({rm0["initial_answer"], rm0["target_answer"]})

    ci = select_cross_item_pairs(picked, n_pairs=4)
    for sr, tr in ci:
        t0 = build_disjoint_target_for_source(tr, sr, perm_order=0)
        t1 = build_disjoint_target_for_source(tr, sr, perm_order=1)
        assert t0["initial_answer"] != t1["initial_answer"]
        assert set(sr["letter_to_outcome"].values()).isdisjoint(set(t0["letter_to_outcome"].values()))

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    tok = C1T._TinyTok()
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in OPTION_LETTERS}
    model = Qwen3ForCausalLM(cfg).float().eval()
    tiny_specs = {
        "b1_full": {"ver_heads": [(1, 0), (2, 1)], "ver_mlps": [1, 2], "form_heads": [(2, 3)]},
        "b1_mlp_only": {"ver_heads": [], "ver_mlps": [1, 2], "form_heads": []},
        "b1_heads_only": {"ver_heads": [(1, 0), (2, 1)], "ver_mlps": [], "form_heads": [(2, 3)]},
        "b1_heads_rnd_mlp": {"ver_heads": [(1, 0), (2, 1)], "ver_mlps": [0, 3], "form_heads": [(2, 3)]},
        "rnd_heads_b1_mlp": {"ver_heads": [(0, 2), (3, 1)], "ver_mlps": [1, 2], "form_heads": [(0, 3)]},
        "rnd_full": {"ver_heads": [(0, 2), (3, 1)], "ver_mlps": [0, 3], "form_heads": [(0, 3)]},
    }
    orig_enc = B1L.encode_prompt

    def _tiny_enc(t, msgs, pref, enable_thinking=False):
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
        return torch.tensor(t.encode(txt), dtype=torch.long)

    B1L.encode_prompt = _tiny_enc
    try:
        summ, det = evaluate_cross_letter_v2_suite(
            model, tok, a1_cfg, picked[:4], tiny_specs, head_dim=16, opt_tok_ids=opt_ids, n_cross_item=2, k_max=4
        )
    finally:
        B1L.encode_prompt = orig_enc

    assert summ["identity_control"]["exact_match_rate"] == 1.0
    for mode in ("cross_letter_flip", "cross_letter_stay", "cross_item_diff_outcome_flip", "cross_item_diff_outcome_stay"):
        assert len(summ[mode]) == 6, mode
    # Verify common-denominator aggregator cleanly
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.jsonl"
        p2 = Path(td) / "b.jsonl"
        txt = "\n".join(json.dumps(x) for x in det) + "\n"
        p1.write_text(txt, encoding="utf-8")
        p2.write_text(txt, encoding="utf-8")
        rep = aggregate_common_denominator({"m1": p1, "m2": p2}, Path(td) / "out")
        assert "cross_letter_flip" in rep["modes"] and "cross_item_diff_outcome_flip" in rep["modes"]
    print(f"[OK] c35_eval_cross_letter_v2 self-test PASSED (id=100%, 4 regimes x 6 specs, {len(det)} records, common-denom verified!)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--aggregate-common", nargs="*", help="List of alias=path_to_records.jsonl pairs to aggregate")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    ap.add_argument("--components-file", type=Path, default=C35 / "data" / "c35_components_frozen.json")
    ap.add_argument("--arch", type=str, default=None)
    ap.add_argument("--adapter-state", type=Path, default=None)
    ap.add_argument("--dataset-file", type=Path, default=C35 / "data" / "c3_blind_confirm_v3_bases.jsonl")
    ap.add_argument("--n-cf", type=int, default=24)
    ap.add_argument("--n-af", type=int, default=24)
    ap.add_argument("--n-ce-a-d2", type=int, default=12)
    ap.add_argument("--n-ce-a-d3", type=int, default=12)
    ap.add_argument("--n-ce-b-d2", type=int, default=12)
    ap.add_argument("--n-ce-b-d3", type=int, default=12)
    ap.add_argument("--n-cross-item", type=int, default=24)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--alias", type=str, default="arm")
    ap.add_argument("--out-dir", type=Path, default=C35 / "results" / "stage5" / "probe2_v2")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    if args.aggregate_common:
        rmap = {}
        for item in args.aggregate_common:
            al, pp = item.split("=", 1)
            rmap[al] = Path(pp)
        rep = aggregate_common_denominator(rmap, args.out_dir)
        print(f"[OK] Wrote common-denominator summary -> {args.out_dir / 'probe2_v2_common_summary.json'}")
        print(json.dumps({m: {s: rep["modes"][m][s]["n_common_eligible"] for s in rep["modes"][m]} for m in rep["modes"]}, indent=2))
        return

    assert args.model, "--model required"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    m_info = comp["models"][args.model_key]
    head_dim = int(m_info["model_spec"]["head_dim"])
    specs_6 = build_7way_specs(m_info)

    all_rows = [json.loads(l) for l in args.dataset_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = select_balanced_probe_v2_rows(
        all_rows,
        n_cf=args.n_cf, n_af=args.n_af,
        n_ce_a_d2=args.n_ce_a_d2, n_ce_a_d3=args.n_ce_a_d3,
        n_ce_b_d2=args.n_ce_b_d2, n_ce_b_d3=args.n_ce_b_d3,
    )

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    opt_tok_ids = C1EV.get_cached_option_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    if args.adapter_state is not None:
        assert args.arch, "--arch required with --adapter-state"
        arch_spec = m_info["architectures"][args.arch]
        scaling = C3EV._read_scaling(args.adapter_state)
        M35.apply_c35_adapters(model, arch_spec, **scaling)
        M35.load_c3_adapter_state(model, args.adapter_state)
        model.eval()

    summaries, detailed = evaluate_cross_letter_v2_suite(
        model, tok, a1_cfg, rows, specs_6, head_dim=head_dim, opt_tok_ids=opt_tok_ids,
        n_cross_item=args.n_cross_item, k_max=args.k_max,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec_p = args.out_dir / f"probe2_v2_records_{args.alias}.jsonl"
    rec_p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in detailed) + "\n", encoding="utf-8", newline="\n")
    out_doc = {
        "schema": "cedar.acl_exp_c35.cross_letter_interchange.v2",
        "alias": args.alias,
        "model_key": args.model_key,
        "arch": args.arch,
        "dataset_file": str(args.dataset_file),
        "n_bases": len(rows),
        "n_cross_item_pairs": args.n_cross_item,
        "composition_by_subgroup": dict(Counter(_subgroup_tag(r) for r in rows)),
        "elapsed_seconds": round(time.time() - t0, 1),
        "summaries": summaries,
    }
    sum_p = args.out_dir / f"probe2_v2_summary_{args.alias}.json"
    sum_p.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] Probe 2 v2 finished for {args.alias} ({args.model_key}) in {out_doc['elapsed_seconds']}s -> {sum_p}")


if __name__ == "__main__":
    main()
