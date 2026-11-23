#!/usr/bin/env python3
"""Build fixed, solver-certified `c3_blind_confirm_v2` & `c3_blind_think_v2` for ACL-exp-c3.5 (CPU only).

Fixes BOTH dataset bugs caught by Astra's audit of ACL-exp-c3:
  1. BUG 1 FIX (CE decoy-clause omission on `before_rules`):
       Matches BOTH `Question:` AND `Based only` when inserting the competing-branch decoy clause in Style-C/D CE
       contexts, and HARD ASSERTS `inserted_decoy == True` on ALL 240/240 CE prompts (120 before_rules + 120 after_rules)!
  2. BUG 2 FIX (Redundant `s1 AND s2` synthesis where `s2` already entailed `s1`):
       Replaces linear `obs -> s1 -> s2` with a **TRUE INDEPENDENT DUAL-PREMISE DAG**:
         - Branch A: `obsA(e1)` --(Rule 1)--> `bA(e1)`
         - Branch B: `obsB(e1)` --(Rule 2)--> `bB(e1)`   (100% independent of Branch A!)
         - Synthesis: `bA(e1) AND bB(e1)` --(Rule 3)--> `trig(e1)`
         - Override : `trig(e1)` --(Rule 4)--> `at_name`
       HARD SOLVER ASSERTION on EVERY single one of the 840 non-CE bases:
         * Full V (`obsA + obsB`) -> `at_name`
         * Drop `obsA` alone      -> `a0_name` (flips gold!)
         * Drop `obsB` alone      -> `a0_name` (flips gold!)
         * Full F                 -> `a0_name` with exact `broken_step in {1, 2, 3}` certified!
  3. PAIRED POSITION / STYLE COUNTERFACTUAL SLICE + FULL HIGH-POWER BLIND SET:
       - 720 independent dual-premise non-CE bases (4 classes x 3 broken_steps x 2 pair_types x 30 reps; balanced
         independently across domain, `before_rules / after_rules`, and two disjoint blind styles `style_c_dossier`
         & `style_d_memo`)
       - 120 paired `(before_rules vs after_rules)` counterfactual bases (60 identical semantic graphs rendered in
         BOTH positions -> tests position invariance cleanly without confounding semantic graphs!)
       - 240 compound CE bases = 120 complete 2x2 cells (`120 Row A` + `120 Row B`, 100% decoy-clause verified!)
       - Total = `1,080` bases / `900` independent generation units (`unit_id` clusters CE A/B pairs & paired pos-pairs)!
       - Zero peek at `c1_test_id / c1_test_ood`, zero `converse_error` / `depth-4`, zero `Only if`, 0 skeleton overlap
         with `c3_train`, G0 TF-IDF GroupKFold AUC ~ 0.50!

Usage:
  python experiments/ACL-exp-c3.5/scripts/c35_build_blind_v2.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
A1_SCRIPTS = REPO / "experiments" / "ACL-exp-a1" / "scripts"
C1_SCRIPTS = REPO / "experiments" / "ACL-exp-c1" / "scripts"
C1_DATA = REPO / "experiments" / "ACL-exp-c1" / "data"
C2_SCRIPTS = REPO / "experiments" / "ACL-exp-c2" / "scripts"
C2_DATA = REPO / "experiments" / "ACL-exp-c2" / "data"
C3_SCRIPTS = REPO / "experiments" / "ACL-exp-c3" / "scripts"
C3_DATA = REPO / "experiments" / "ACL-exp-c3" / "data"
C35_DATA = C35 / "data"

sys.path.insert(0, str(A1_SCRIPTS))
sys.path.insert(0, str(C1_SCRIPTS))
sys.path.insert(0, str(C2_SCRIPTS))
sys.path.insert(0, str(C3_SCRIPTS))

import build_symbolic_pilot_v2 as A1_SYM  # noqa: E402
import c1_build_datasets as C1B  # noqa: E402
import c1_build_proxy_template_ood as PT1  # noqa: E402
import c2_build_datasets as C2B  # noqa: E402
import c3_build_datasets as C3B  # noqa: E402
from a1_arm_wrappers import ARM_ORDER, build_arms  # noqa: E402

C35_BLIND_SEED = 20280119
OPTION_LETTERS = ("A", "B", "C", "D")

# Independent second observation predicate for Branch B on each of the 6 C3_BLIND_DOMAINS
DOMAIN_OBS_B: dict[str, str] = {
    "observatory_dome": "confirmed thermal equilibrium",
    "substation_grid": "recorded frequency lock",
    "canal_lock": "verified chamber equalization",
    "cellar_vault": "logged cellar humidity clearance",
    "seismic_array": "confirmed borehole telemetry sync",
    "telemetry_buoy": "reported buoy transponder heartbeat",
}


def _render_blind_context(
    dom: dict[str, Any],
    facts: list[str],
    rules: list[str],
    target_entity: str,
    letter_to_outcome: dict[str, str],
    default_policy_text: str,
    pos: str,
    style: str,
) -> str:
    """Render in one of two disjoint blind discourse styles (`style_c_dossier` or `style_d_memo`)."""
    assert style in ("style_c_dossier", "style_d_memo")
    opts = "\n".join(f"{k}. {letter_to_outcome[k]}" for k in OPTION_LETTERS)
    if style == "style_c_dossier":
        fact_block = "\n".join(f"Entry {i + 1}: {f}" for i, f in enumerate(facts))
        clause_lines = []
        for i, r in enumerate(rules):
            r2 = re.sub(r"^If (.+?), then ", r"When \1, it follows that ", r)
            r2 = re.sub(
                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                r"When \1, supersede the baseline assignment and route \2 to \3.",
                r2,
            )
            clause_lines.append(f"Clause {i + 1} — {r2}")
        rule_block = "\n".join(clause_lines)
        dp = default_policy_text.replace("Default policy:", "Baseline Rule:", 1)
        body = f"{dp}\n\nDecision Provisions:\n{rule_block}" if pos == "before_rules" else f"Decision Provisions:\n{rule_block}\n\n{dp}"
        return (
            f"[Case Dossier]\n{dom['intro']}\n"
            f"Closed-dossier note: an entry's status holds only when listed under Observed Statuses, derived via Decision Provisions, or reported in a verified telemetry bulletin.\n\n"
            f"Observed Statuses:\n{fact_block}\n\n"
            f"{body}\n\n"
            f"Question: Based strictly on the listed dossier entries and provisions, what is the current classification of {target_entity}?\n"
            f"Options:\n{opts}"
        )
    else:
        fact_block = "\n".join(f"Record #{i + 1}: {f}" for i, f in enumerate(facts))
        clause_lines = []
        for i, r in enumerate(rules):
            r2 = re.sub(r"^If (.+?), then ", r"In cases where \1, establish that ", r)
            r2 = re.sub(
                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                r"In cases where \1, replace the standing assignment and direct \2 to \3.",
                r2,
            )
            clause_lines.append(f"Directive {i + 1}: {r2}")
        rule_block = "\n".join(clause_lines)
        dp = default_policy_text.replace("Default policy:", "Standing Rule:", 1)
        body = f"{dp}\n\nOperating Directives:\n{rule_block}" if pos == "before_rules" else f"Operating Directives:\n{rule_block}\n\n{dp}"
        return (
            f"[Operations Memo]\n{dom['intro']}\n"
            f"Closed-memo note: a status holds only if stated under Current Records, derived from Operating Directives, or supplied in a verified field update.\n\n"
            f"Current Records:\n{fact_block}\n\n"
            f"{body}\n\n"
            f"Question: Using only the current records and operating directives above, what is the assigned classification of {target_entity}?\n"
            f"Options:\n{opts}"
        )


def _rewrite_arg_blind_style(txt: str, style: str) -> str:
    if style == "style_c_dossier":
        t = re.sub(r"^New verified observation:\s*", "Verified telemetry bulletin: ", txt)
        t = re.sub(r"\bStep (\d+):\s*", r"Phase \1: ", t)
        t = re.sub(r"\bRule (\d+)\b", r"Clause \1", t)
        t = re.sub(r"\bConclusion:\s*", "Final Determination: ", t)
        return t
    else:
        t = re.sub(r"^New verified observation:\s*", "Verified field update: ", txt)
        t = re.sub(r"\bStep (\d+):\s*", r"Check \1 — ", t)
        t = re.sub(r"\bRule (\d+)\b", r"Directive \1", t)
        t = re.sub(r"\bConclusion:\s*", "Resulting Classification: ", t)
        return t


def _verify_dual_premise_dag(
    base_facts: set[str],
    obs_set: set[str],
    rules: list[dict[str, Any]],
    bA: str,
    bB: str,
    trig: str,
    *,
    expected_broken_step: int | None,
) -> None:
    """Strictly verify independent dual-premise DAG derivations & exact broken step in {None, 1, 2, 3}."""
    known = set(base_facts) | set(obs_set)
    depth_of: dict[str, int] = {a: 0 for a in known}
    for _ in range(len(rules) + 2):
        changed = False
        for r in rules:
            if all(c in known for c in r["conditions"]):
                d_new = 1 + max((depth_of[c] for c in r["conditions"]), default=0)
                concl = r["conclusion"]
                if concl not in known or d_new < depth_of[concl]:
                    known.add(concl)
                    depth_of[concl] = d_new
                    changed = True
        if not changed:
            break
    if expected_broken_step is None:
        assert depth_of.get(bA) == 1, (bA, depth_of.get(bA))
        assert depth_of.get(bB) == 1, (bB, depth_of.get(bB))
        assert depth_of.get(trig) == 2, (trig, depth_of.get(trig))
    elif expected_broken_step == 1:
        assert bA not in known and depth_of.get(bB) == 1 and trig not in known, (bA in known, depth_of.get(bB), trig in known)
    elif expected_broken_step == 2:
        assert depth_of.get(bA) == 1 and bB not in known and trig not in known, (depth_of.get(bA), bB in known, trig in known)
    elif expected_broken_step == 3:
        assert depth_of.get(bA) == 1 and depth_of.get(bB) == 1 and trig not in known, (depth_of.get(bA), depth_of.get(bB), trig in known)
    else:
        raise ValueError(expected_broken_step)


def _build_independent_dual_premise_base(
    *,
    domains: tuple[dict[str, Any], ...],
    seed_base: int,
    idx: int,
    dom_idx: int,
    fallacy_class: str,
    broken_step: int,
    pair_type: str,
    pos: str,
    style: str,
    trans: tuple[str, str],
    bid: str,
    unit_id: str,
    split_name: str,
) -> dict[str, Any]:
    """Build a solver-certified INDEPENDENT dual-premise base (removing obsA OR obsB independently flips gold!)."""
    assert fallacy_class in ("scope_swap", "missing_conjunct", "negation_drop", "threshold_violation")
    assert broken_step in (1, 2, 3)
    assert pair_type in ("context_flip", "argument_flip")

    rng = random.Random(seed_base + idx * 10009 + broken_step * 101)
    dom = domains[dom_idx]
    p = dict(dom["props"])
    a0, at = trans

    outcomes = list(dom["outcomes"])
    rng.shuffle(outcomes)
    letter_to_outcome = dict(zip(OPTION_LETTERS, outcomes))
    a0_name, at_name = letter_to_outcome[a0], letter_to_outcome[at]
    decoy_letters = [k for k in OPTION_LETTERS if k not in (a0, at)]
    a_decoy = decoy_letters[rng.randrange(len(decoy_letters))]
    a_decoy_name = letter_to_outcome[a_decoy]

    ep = dom["entity_prefix"]
    ents = list(dom["entities"])
    rng.shuffle(ents)
    e1, e2, e3, e4 = [f"{ep} {x}" for x in ents]
    zones = list(dom["zones"])
    rng.shuffle(zones)
    z1, z2 = zones[0], zones[1]

    p_obsA = p["p_obs"]
    p_obsB = DOMAIN_OBS_B[dom["name"]]
    p_bA = p["p_mid1"]
    p_bB = p["p_mid3"]
    p_trig = p["p_mid2"]
    p_aux = p["p_aux"]
    p_alt = p["p_alt_mid"]
    metric = dom["metric_name"]

    t_low = rng.randint(18, 86)
    d1, d2 = rng.randint(1, 3), rng.randint(1, 3)
    t_v = t_low
    obs_val_v = t_low + d1
    t_f = obs_val_v + d2
    obs_val_f = t_v - d2

    override_trigger = f"trig({e1})"
    # TRUE INDEPENDENT DUAL-PREMISE TOPOLOGY:
    # Step 1 (Branch A): obsA(e1) -> bA(e1)
    # Step 2 (Branch B): obsB(e1) -> bB(e1)  (completely independent of Step 1!)
    # Step 3 (Joint Synthesis): bA(e1) AND bB(e1) -> trig(e1)
    step_in_pred = {
        1: [f"obsA({e1})"],
        2: [f"obsB({e1})"],
        3: [f"bA({e1})", f"bB({e1})"],
    }
    step_in_text = {
        1: f"{e1} {p_obsA}",
        2: f"{e1} {p_obsB}",
        3: f"{e1} {p_bA} and {e1} {p_bB}",
    }
    step_out_pred = {1: f"bA({e1})", 2: f"bB({e1})", 3: override_trigger}
    step_out_text = {1: f"{e1} {p_bA}", 2: f"{e1} {p_bB}", 3: f"{e1} {p_trig}"}

    obs_v_set: set[str] = {f"obsA({e1})", f"obsB({e1})"}
    obs_f_set: set[str] = {f"obsA({e1})", f"obsB({e1})"}
    obs_v_text = f"{e1} {p_obsA}, and independently {e1} {p_obsB}."
    obs_f_text = obs_v_text

    if fallacy_class == "scope_swap":
        if pair_type == "context_flip":
            facts_v = [
                f"{e1} is registered under active default status.",
                f"{e1} belongs to {z1}.",
                f"{e2} belongs to {z2}.",
                f"For {e1}, {dom['guard_inactive_text']}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            facts_f = list(facts_v)
            bf_v = {f"default({e1})", f"in_zone({e1},{z1})", f"in_zone({e2},{z2})", f"guard_inactive({e1})", f"n1({e3})", f"n2({e4})"}
            bf_f = set(bf_v)
            gate_v_txt = f"For entries in {z1}, if {step_in_text[broken_step]}, then {step_out_text[broken_step]}."
            gate_f_txt = f"For entries in {z2}, if {step_in_text[broken_step]}, then {step_out_text[broken_step]}."
            gate_v_cond = [f"in_zone({e1},{z1})"] + step_in_pred[broken_step]
            gate_f_cond = [f"in_zone({e1},{z2})"] + step_in_pred[broken_step]
            sp_v = f"Applying Rule {broken_step} to the verified status, {step_out_text[broken_step]}."
            sp_f = sp_v
        else:
            facts_v = [
                f"{e1} is registered under active default status.",
                f"For {e1}, {dom['guard_inactive_text']}.",
                f"{e2} belongs to {z2}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            facts_f = list(facts_v)
            bf_v = {f"default({e1})", f"guard_inactive({e1})", f"in_zone({e2},{z2})", f"n1({e3})", f"n2({e4})"}
            bf_f = set(bf_v)
            gate_v_txt = f"For entries in {z1}, if {step_in_text[broken_step]}, then {step_out_text[broken_step]}."
            gate_f_txt = gate_v_txt
            gate_v_cond = [f"in_zone({e1},{z1})"] + step_in_pred[broken_step]
            gate_f_cond = list(gate_v_cond)
            obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} is assigned to {z1}."
            obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} is assigned to {z2}."
            obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"in_zone({e1},{z1})"}
            obs_f_set = {f"obsA({e1})", f"obsB({e1})", f"in_zone({e1},{z2})"}
            sp_v = f"Applying Rule {broken_step} under that zone assignment, {step_out_text[broken_step]}."
            sp_f = sp_v

    elif fallacy_class == "missing_conjunct":
        if pair_type == "context_flip":
            facts_v = [
                f"{e1} is registered under active default status.",
                f"{e1} belongs to {z1}.",
                f"{e1} {p_aux}.",
                f"For {e1}, {dom['guard_inactive_text']}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            facts_f = [
                f"{e1} is registered under active default status.",
                f"{e1} belongs to {z1}.",
                f"{e2} {p_aux}.",
                f"For {e1}, {dom['guard_inactive_text']}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            bf_v = {f"default({e1})", f"in_zone({e1},{z1})", f"aux({e1})", f"guard_inactive({e1})", f"n1({e3})", f"n2({e4})"}
            bf_f = {f"default({e1})", f"in_zone({e1},{z1})", f"aux({e2})", f"guard_inactive({e1})", f"n1({e3})", f"n2({e4})"}
            gate_v_txt = f"If {step_in_text[broken_step]} and {e1} {p_aux}, then {step_out_text[broken_step]}."
            gate_f_txt = gate_v_txt
            gate_v_cond = step_in_pred[broken_step] + [f"aux({e1})"]
            gate_f_cond = list(gate_v_cond)
            sp_v = f"Applying Rule {broken_step} to the verified status, {step_out_text[broken_step]}."
            sp_f = sp_v
        else:
            facts_v = [
                f"{e1} is registered under active default status.",
                f"{e1} belongs to {z1}.",
                f"{e2} belongs to {z2}.",
                f"For {e1}, {dom['guard_inactive_text']}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            facts_f = list(facts_v)
            bf_v = {f"default({e1})", f"in_zone({e1},{z1})", f"in_zone({e2},{z2})", f"guard_inactive({e1})", f"n1({e3})", f"n2({e4})"}
            bf_f = set(bf_v)
            gate_v_txt = f"If {step_in_text[broken_step]} and {e1} {p_aux}, then {step_out_text[broken_step]}."
            gate_f_txt = gate_v_txt
            gate_v_cond = step_in_pred[broken_step] + [f"aux({e1})"]
            gate_f_cond = list(gate_v_cond)
            obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} {p_aux}."
            obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e2} {p_aux}."
            obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e1})"}
            obs_f_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e2})"}
            sp_v = f"Applying Rule {broken_step} with that reported credential, {step_out_text[broken_step]}."
            sp_f = sp_v

    elif fallacy_class == "negation_drop":
        if pair_type == "context_flip":
            facts_v = [
                f"{e1} is registered under active default status.",
                f"{e1} belongs to {z1}.",
                f"For {e1}, {dom['guard_inactive_text']}.",
                f"For {e2}, {dom['guard_active_text']}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            facts_f = [
                f"{e1} is registered under active default status.",
                f"{e1} belongs to {z1}.",
                f"For {e1}, {dom['guard_active_text']}.",
                f"For {e2}, {dom['guard_inactive_text']}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            bf_v = {f"default({e1})", f"in_zone({e1},{z1})", f"guard_inactive({e1})", f"guard_active({e2})", f"n1({e3})", f"n2({e4})"}
            bf_f = {f"default({e1})", f"in_zone({e1},{z1})", f"guard_active({e1})", f"guard_inactive({e2})", f"n1({e3})", f"n2({e4})"}
            gate_v_txt = f"If {step_in_text[broken_step]} and {dom['guard_inactive_text']} for {e1}, then {step_out_text[broken_step]}."
            gate_f_txt = gate_v_txt
            gate_v_cond = step_in_pred[broken_step] + [f"guard_inactive({e1})"]
            gate_f_cond = list(gate_v_cond)
            sp_v = f"Applying Rule {broken_step} to the verified status, {step_out_text[broken_step]}."
            sp_f = sp_v
        else:
            require_inactive = (idx % 2 == 0)
            req_txt = dom["guard_inactive_text"] if require_inactive else dom["guard_active_text"]
            opp_txt = dom["guard_active_text"] if require_inactive else dom["guard_inactive_text"]
            req_pred = f"guard_ok({e1})"
            opp_pred = f"guard_bad({e1})"
            facts_v = [
                f"{e1} is registered under active default status.",
                f"{e1} belongs to {z1}.",
                f"{e2} belongs to {z2}.",
                f"{e3} {p['p_noise1']}.",
                f"{e4} {p['p_noise2']}.",
            ]
            facts_f = list(facts_v)
            bf_v = {f"default({e1})", f"in_zone({e1},{z1})", f"in_zone({e2},{z2})", f"n1({e3})", f"n2({e4})"}
            bf_f = set(bf_v)
            gate_v_txt = f"If {step_in_text[broken_step]} and {req_txt} for {e1}, then {step_out_text[broken_step]}."
            gate_f_txt = gate_v_txt
            gate_v_cond = step_in_pred[broken_step] + [req_pred]
            gate_f_cond = list(gate_v_cond)
            obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and for {e1}, {req_txt}."
            obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and for {e1}, {opp_txt}."
            obs_v_set = {f"obsA({e1})", f"obsB({e1})", req_pred}
            obs_f_set = {f"obsA({e1})", f"obsB({e1})", opp_pred}
            sp_v = f"Applying Rule {broken_step} under that guard status, {step_out_text[broken_step]}."
            sp_f = sp_v

    else:  # threshold_violation
        facts_v = [
            f"{e1} is registered under active default status.",
            f"{e1} belongs to {z1}.",
            f"For {e1}, {dom['guard_inactive_text']}.",
            f"{e3} {p['p_noise1']}.",
            f"{e4} {p['p_noise2']}.",
        ]
        facts_f = list(facts_v)
        bf_v = {f"default({e1})", f"in_zone({e1},{z1})", f"guard_inactive({e1})", f"n1({e3})", f"n2({e4})"}
        bf_f = set(bf_v)
        if pair_type == "context_flip":
            gate_v_txt = f"If {step_in_text[broken_step]} and the {metric} is at least {t_v}, then {step_out_text[broken_step]}."
            gate_f_txt = f"If {step_in_text[broken_step]} and the {metric} is at least {t_f}, then {step_out_text[broken_step]}."
            gate_v_cond = step_in_pred[broken_step] + ["thresh_v"]
            gate_f_cond = step_in_pred[broken_step] + ["thresh_f"]
            obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, with a {metric} of {obs_val_v}."
            obs_f_text = obs_v_text
            obs_v_set = {f"obsA({e1})", f"obsB({e1})", "thresh_v"}
            obs_f_set = {f"obsA({e1})", f"obsB({e1})", "thresh_v"}
            sp_v = f"Applying Rule {broken_step} using the verified {metric} reading, {step_out_text[broken_step]}."
            sp_f = sp_v
        else:
            gate_v_txt = f"If {step_in_text[broken_step]} and the {metric} is at least {t_v}, then {step_out_text[broken_step]}."
            gate_f_txt = gate_v_txt
            gate_v_cond = step_in_pred[broken_step] + ["thresh_ok"]
            gate_f_cond = list(gate_v_cond)
            obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, with a {metric} of {obs_val_v}."
            obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, with a {metric} of {obs_val_f}."
            obs_v_set = {f"obsA({e1})", f"obsB({e1})", "thresh_ok"}
            obs_f_set = {f"obsA({e1})", f"obsB({e1})"}
            sp_v = f"Applying Rule {broken_step} using the verified {metric} reading, {step_out_text[broken_step]}."
            sp_f = sp_v

    default_policy_text = (
        f"Default policy: {e1} is under active default status, so {e1} is assigned to {a0_name} "
        f"unless the rules establish that {e1} {p_trig}."
    )

    def _assemble_rules(gate_txt: str, gate_cond: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
        r1_t = gate_txt if broken_step == 1 else f"If {step_in_text[1]}, then {step_out_text[1]}."
        r2_t = gate_txt if broken_step == 2 else f"If {step_in_text[2]}, then {step_out_text[2]}."
        r3_t = gate_txt if broken_step == 3 else f"If {step_in_text[3]}, then {step_out_text[3]}."
        r1_c = gate_cond if broken_step == 1 else list(step_in_pred[1])
        r2_c = gate_cond if broken_step == 2 else list(step_in_pred[2])
        r3_c = gate_cond if broken_step == 3 else list(step_in_pred[3])

        rlines = [
            r1_t,
            r2_t,
            r3_t,
            f"If {e1} {p_trig}, override the default classification and assign {e1} to {at_name}.",
            # Competing-branch active distractor rules (non-triggering because emergency waiver is absent!)
            f"If {e1} {p_obsA} and {e1} {p_bA}, then {e1} {p_alt}.",
            f"If {e1} {p_alt} and {e1} carries an emergency board waiver, override the default classification and assign {e1} to {a_decoy_name}.",
            f"If {e3} {p['p_noise1']} and {e4} {p['p_noise2']}, record a routine audit note for {e3}.",
            f"If {e2} {p_obsA}, then {e2} is placed on a standby list.",
        ]
        rdicts = [
            {"id": "Rule 1", "conditions": r1_c, "conclusion": step_out_pred[1]},
            {"id": "Rule 2", "conditions": r2_c, "conclusion": step_out_pred[2]},
            {"id": "Rule 3", "conditions": r3_c, "conclusion": step_out_pred[3]},
            {"id": "Rule 5", "conditions": [f"obsA({e1})", f"bA({e1})"], "conclusion": f"alt({e1})"},
            {"id": "Rule 6", "conditions": [f"alt({e1})", f"waiver({e1})"], "conclusion": f"decoy_trig({e1})"},
            {"id": "Rule 7", "conditions": [f"n1({e3})", f"n2({e4})"], "conclusion": f"audit({e3})"},
            {"id": "Rule 8", "conditions": [f"obsA({e2})"], "conclusion": f"standby({e2})"},
        ]
        return rlines, rdicts

    rlines_v, rules_v = _assemble_rules(gate_v_txt, gate_v_cond)
    rlines_f, rules_f = _assemble_rules(gate_f_txt, gate_f_cond)

    # STRICT SOLVER & INDEPENDENT DUAL-PREMISE NECESSITY CERTIFICATION!
    _, t1v = A1_SYM.evaluate_world(bf_v, {}, rules_v, a0_name, override_trigger, at_name)
    _, t1f = A1_SYM.evaluate_world(bf_f, {}, rules_f, a0_name, override_trigger, at_name)
    assert t1v == a0_name and t1f == a0_name
    _, outv = A1_SYM.evaluate_world(bf_v | obs_v_set, {}, rules_v, a0_name, override_trigger, at_name)
    _, outf = A1_SYM.evaluate_world(bf_f | obs_f_set, {}, rules_f, a0_name, override_trigger, at_name)
    assert outv == at_name and outf == a0_name, (bid, outv, outf)

    # Astra's exact independent-premise necessity test: dropping obsA OR obsB independently flips gold back to a0!
    _, drop_a = A1_SYM.evaluate_world(bf_v | (obs_v_set - {f"obsA({e1})"}), {}, rules_v, a0_name, override_trigger, at_name)
    _, drop_b = A1_SYM.evaluate_world(bf_v | (obs_v_set - {f"obsB({e1})"}), {}, rules_v, a0_name, override_trigger, at_name)
    assert drop_a == a0_name and drop_b == a0_name, f"Dual-premise independence failed on {bid}: dropA={drop_a}, dropB={drop_b}"

    _verify_dual_premise_dag(bf_v, obs_v_set, rules_v, f"bA({e1})", f"bB({e1})", override_trigger, expected_broken_step=None)
    _verify_dual_premise_dag(bf_f, obs_f_set, rules_f, f"bA({e1})", f"bB({e1})", override_trigger, expected_broken_step=broken_step)

    ctx_v = _render_blind_context(dom, facts_v, rlines_v, e1, letter_to_outcome, default_policy_text, pos, style)
    ctx_f = _render_blind_context(dom, facts_f, rlines_f, e1, letter_to_outcome, default_policy_text, pos, style)
    if pair_type == "context_flip":
        assert ctx_v != ctx_f
    else:
        assert ctx_v == ctx_f

    def _build_core(obs_t: str, sp_p: str) -> str:
        s1_t = sp_p if broken_step == 1 else f"Since {e1} {p_obsA}, Rule 1 establishes that {e1} {p_bA}."
        s2_t = sp_p if broken_step == 2 else f"Independently, since {e1} {p_obsB}, Rule 2 establishes that {e1} {p_bB}."
        s3_t = sp_p if broken_step == 3 else f"Combining both independent statuses ({e1} {p_bA} and {e1} {p_bB}) under Rule 3, {e1} {p_trig}."
        return (
            f"New verified observation: {obs_t} "
            f"Step 1: {s1_t} "
            f"Step 2: {s2_t} "
            f"Step 3: {s3_t} "
            f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {at_name} (Option {at})."
        )

    core_v = _build_core(obs_v_text, sp_v)
    core_f = _build_core(obs_f_text, sp_f)
    if pair_type == "context_flip":
        assert core_v == core_f

    shell_only = (
        f"New administrative log: routine archive headers were checked. "
        f"Step 1: Since those clerical timestamps appear in the log, the standard display checklist applies. "
        f"Step 2: Applying display formatting conventions to archived entries adds the revised presentation tag. "
        f"Conclusion: Accordingly, {e1} should be listed under {at_name} (Option {at})."
    )
    i0_text = (
        f"Verification note: {e1} remains governed by the default policy with no valid override triggered, "
        f"so Option {a0} remains valid. Please confirm your final answer."
    )
    arms = build_arms(core_v=core_v, core_f=core_f, shell_only=shell_only, a0=a0, at=at, i0_text=i0_text, final_noun="answer")
    for an in list(arms.keys()):
        if an.startswith(("V", "F")) or an == "shell_only":
            arms[an] = _rewrite_arg_blind_style(arms[an], style)
    assert tuple(arms) == ARM_ORDER

    return {
        "schema": "cedar.acl_exp_c35.blind_dual_premise.v2",
        "split": split_name,
        "base_id": bid,
        "cell_id": unit_id,
        "family_id": unit_id,
        "unit_id": unit_id,
        "domain": dom["name"],
        "pair_type": pair_type,
        "difficulty_depth": 3,
        "fallacy_class": fallacy_class,
        "broken_step": broken_step,
        "independent_dual_premise": True,
        "phrasing_style": style,
        "default_policy_position": pos,
        "initial_answer": a0,
        "target_answer": at,
        "letter_to_outcome": letter_to_outcome,
        "context_prompt_v": ctx_v,
        "context_prompt_f": ctx_f,
        "context_prompt_f_placebo": ctx_f,
        "context_prompt_shared_for_non_v": ctx_f,
        "arms": arms,
    }


def _build_fixed_compound_ce_cell(
    *,
    domains: tuple[dict[str, Any], ...],
    seed_base: int,
    cell_idx: int,
    dom_idx: int,
    depth: int,
    pos: str,
    style: str,
    trans: tuple[str, str],
    cell_id: str,
    bid_a: str,
    bid_b: str,
    split_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fixed CE compound 2x2 cell: HARD ASSERTS competing-branch decoy clause is inserted on ALL 240/240 prompts!"""
    ra, rb = C2B._build_single_chain_2x2_ce_cell(
        domains=domains,
        seed_base=seed_base,
        cell_idx=cell_idx,
        dom_idx=dom_idx,
        depth=depth,
        pos=pos,
        trans=trans,
        cell_id=cell_id,
        bid_a=bid_a,
        bid_b=bid_b,
        split_name=split_name,
        schema_name="cedar.acl_exp_c35.blind_compound_ce_2x2.v2",
    )
    dom = domains[dom_idx]
    p = dict(dom["props"])
    rng = random.Random(seed_base + cell_idx * 4229)
    a0, at = trans
    decoy_letters = [k for k in OPTION_LETTERS if k not in (a0, at)]
    a_decoy = decoy_letters[rng.randrange(len(decoy_letters))]

    for row in (ra, rb):
        a_decoy_name = row["letter_to_outcome"][a_decoy]

        def _render_ce_ctx(ctx: str) -> str:
            lines = []
            rule_idx = 0
            in_rules = False
            inserted_decoy = False

            def _decoy_line(next_idx: int) -> str:
                if style == "style_c_dossier":
                    return (
                        f"Clause {next_idx} — When an entry {p['p_alt_mid']} and carries an emergency board waiver, "
                        f"supersede the baseline assignment and route it to {a_decoy_name}."
                    )
                else:
                    return (
                        f"Directive {next_idx}: In cases where an entry {p['p_alt_mid']} and carries an emergency board waiver, "
                        f"replace the standing assignment and direct it to {a_decoy_name}."
                    )

            for line in ctx.splitlines():
                s = line.strip()
                if s == dom["intro"]:
                    lines.append("[Case Dossier]" if style == "style_c_dossier" else "[Operations Memo]")
                    lines.append(s)
                elif s.startswith("Closed-world note:"):
                    if style == "style_c_dossier":
                        lines.append(
                            "Closed-dossier note: an entry's status holds only when listed under Observed Statuses, derived via Decision Provisions, or reported in a verified telemetry bulletin."
                        )
                    else:
                        lines.append(
                            "Closed-memo note: a status holds only if stated under Current Records, derived from Operating Directives, or supplied in a verified field update."
                        )
                elif s == "Facts:":
                    lines.append("Observed Statuses:" if style == "style_c_dossier" else "Current Records:")
                elif s.startswith("Fact "):
                    m = re.match(r"^Fact (\d+)\.\s*(.+)$", s)
                    if m:
                        lines.append(f"Entry {m.group(1)}: {m.group(2)}" if style == "style_c_dossier" else f"Record #{m.group(1)}: {m.group(2)}")
                    else:
                        lines.append(line)
                elif s == "Rules:":
                    in_rules = True
                    lines.append("Decision Provisions:" if style == "style_c_dossier" else "Operating Directives:")
                elif s.startswith("Rule "):
                    m = re.match(r"^Rule (\d+)\.\s*(.+)$", s)
                    if m:
                        rule_idx = int(m.group(1))
                        body = m.group(2)
                        if style == "style_c_dossier":
                            body2 = re.sub(r"^If (.+?), then ", r"When \1, it follows that ", body)
                            body2 = re.sub(
                                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                                r"When \1, supersede the baseline assignment and route \2 to \3.",
                                body2,
                            )
                            lines.append(f"Clause {rule_idx} — {body2}")
                        else:
                            body2 = re.sub(r"^If (.+?), then ", r"In cases where \1, establish that ", body)
                            body2 = re.sub(
                                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                                r"In cases where \1, replace the standing assignment and direct \2 to \3.",
                                body2,
                            )
                            lines.append(f"Directive {rule_idx}: {body2}")
                    else:
                        lines.append(line)
                elif s.startswith("Default policy:"):
                    if in_rules and not inserted_decoy:
                        lines.append(_decoy_line(rule_idx + 1))
                        inserted_decoy = True
                        in_rules = False
                    lines.append(
                        s.replace("Default policy:", "Baseline Rule:" if style == "style_c_dossier" else "Standing Rule:", 1)
                    )
                elif s.startswith(("Question:", "Based only")):
                    # FIX FOR ASTRA BUG 1: catches BOTH `Question:` AND `Based only ...`!
                    if in_rules and not inserted_decoy:
                        lines.append(_decoy_line(rule_idx + 1))
                        inserted_decoy = True
                        in_rules = False
                    if style == "style_c_dossier":
                        lines.append(s.replace("listed facts and rules", "listed dossier entries and provisions"))
                    else:
                        lines.append(s.replace("listed facts and rules", "current records and operating directives"))
                else:
                    lines.append(line)
            assert inserted_decoy, f"Decoy clause was NOT inserted on {row['base_id']} pos={pos}!"
            return "\n".join(lines)

        for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
            row[k] = _render_ce_ctx(row[k])
        for an in list(row["arms"].keys()):
            if an.startswith(("V", "F")) or an == "shell_only":
                row["arms"][an] = _rewrite_arg_blind_style(row["arms"][an], style)
        row["family_id"] = cell_id
        row["unit_id"] = cell_id
        row["compound_blind"] = True
        row["phrasing_style"] = style
        row["broken_step"] = 1
    return ra, rb


def main() -> None:
    C3B._verify_all_domain_pools_disjoint()
    C35_DATA.mkdir(parents=True, exist_ok=True)

    # Deduplicate against ALL open A1 / C1 / C2 / C3 splits (NEVER touch c1_test_id / c1_test_ood!)
    dedup = C1B.SplitDeduplicator()
    C1B.load_a1_rows_into_dedup(dedup)
    open_files = [
        C1_DATA / "c1_train_bases.jsonl",
        C1_DATA / "c1_dev_bases.jsonl",
        C1_DATA / "c1_proxy_ood_bases.jsonl",
        C1_DATA / "c1_proxy_template_ood_bases.jsonl",
        C1_DATA / "c1_proxy_template_v2_bases.jsonl",
        C1_DATA / "c1_step2b_confirm_ce_bases.jsonl",
        C2_DATA / "c2_train_bases.jsonl",
        C2_DATA / "c2_confirm_bidir_bases.jsonl",
        C2_DATA / "c2_stealth_stress_bases.jsonl",
        C3_DATA / "c3_train_bases.jsonl",
        C3_DATA / "c3_dev_bases.jsonl",
    ]
    c3_train_rows: list[dict[str, Any]] = []
    for p in open_files:
        assert "c1_test_id" not in p.name and "c1_test_ood" not in p.name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            dedup.register_external(r)
            if p.name == "c3_train_bases.jsonl":
                c3_train_rows.append(r)

    non_ce_classes = ("scope_swap", "missing_conjunct", "negation_drop", "threshold_violation")
    blind_v2_bases: list[dict[str, Any]] = []

    # 1. 720 independent dual-premise non-CE bases (4 classes x 3 steps x 2 pair_types x 30 reps)
    specs_indep = []
    for fc in non_ce_classes:
        for bstep in (1, 2, 3):
            for pt in ("context_flip", "argument_flip"):
                for _ in range(30):
                    specs_indep.append((fc, bstep, pt))
    assert len(specs_indep) == 720
    fac_indep = C3B._deconfounded_factors(720, len(C3B.C3_BLIND_DOMAINS), seed=C35_BLIND_SEED + 101, include_style=True)

    for idx, ((fc, bstep, pt), fac) in enumerate(zip(specs_indep, fac_indep), start=1):
        dom_idx, pos = fac["dom_idx"], fac["pos"]
        style = "style_c_dossier" if fac["style"] == "style_a" else "style_d_memo"
        trans = dedup.next_balanced_transition(dom_idx, salt=211)
        bid = f"C35-BLD-{len(blind_v2_bases) + 1:04d}"
        attempt = 0
        while True:
            cur_idx = 900000 + idx + attempt * 6 * 19991
            row = _build_independent_dual_premise_base(
                domains=C3B.C3_BLIND_DOMAINS,
                seed_base=C35_BLIND_SEED + 100000,
                idx=cur_idx,
                dom_idx=dom_idx,
                fallacy_class=fc,
                broken_step=bstep,
                pair_type=pt,
                pos=pos,
                style=style,
                trans=trans,
                bid=bid,
                unit_id=bid,
                split_name="c3_blind_confirm_v2",
            )
            if not dedup.has_collision(row):
                dedup.commit(row)
                blind_v2_bases.append(row)
                break
            attempt += 1
            assert attempt < 80

    # 2. 60 PAIRED `(before_rules vs after_rules)` counterfactual semantic graphs = 120 bases (60 units)
    specs_pair = []
    for fc in non_ce_classes:
        for bstep in (1, 2, 3):
            for _ in range(5):
                specs_pair.append((fc, bstep, "context_flip" if len(specs_pair) % 2 == 0 else "argument_flip"))
    assert len(specs_pair) == 60
    fac_pair = C3B._deconfounded_factors(60, len(C3B.C3_BLIND_DOMAINS), seed=C35_BLIND_SEED + 202, include_style=True)
    for pidx, ((fc, bstep, pt), fac) in enumerate(zip(specs_pair, fac_pair), start=1):
        dom_idx = fac["dom_idx"]
        style_a = "style_c_dossier" if fac["style"] == "style_a" else "style_d_memo"
        style_b = "style_d_memo" if style_a == "style_c_dossier" else "style_c_dossier"
        trans = dedup.next_balanced_transition(dom_idx, salt=223)
        uid = f"C35-POSPAIR-{pidx:03d}"
        cur_idx = 950000 + pidx * 13
        for pos, sty in (("before_rules", style_a), ("after_rules", style_b)):
            bid = f"C35-BLD-{len(blind_v2_bases) + 1:04d}"
            row = _build_independent_dual_premise_base(
                domains=C3B.C3_BLIND_DOMAINS,
                seed_base=C35_BLIND_SEED + 300000,
                idx=cur_idx,  # EXACT SAME semantic graph & numbers!
                dom_idx=dom_idx,
                fallacy_class=fc,
                broken_step=bstep,
                pair_type=pt,
                pos=pos,
                style=sty,
                trans=trans,
                bid=bid,
                unit_id=uid,
                split_name="c3_blind_confirm_v2",
            )
            row["paired_position_counterfactual"] = True
            blind_v2_bases.append(row)

    # 3. 120 Complete 2x2 Fixed Compound CE Cells = 240 bases (ALL 240/240 decoy-clause asserted!)
    fac_ce = C3B._deconfounded_factors(120, len(C3B.C3_BLIND_DOMAINS), seed=C35_BLIND_SEED + 303, include_style=True)
    for cell_i, fac in enumerate(fac_ce):
        dom_idx, pos = fac["dom_idx"], fac["pos"]
        style = "style_c_dossier" if fac["style"] == "style_a" else "style_d_memo"
        depth = 2 if cell_i < 60 else 3
        trans = dedup.next_balanced_transition(dom_idx, salt=229)
        cell_id = f"C35-BLD-CECELL-{cell_i + 1:03d}"
        bid_a = f"C35-BLD-{len(blind_v2_bases) + 1:04d}"
        bid_b = f"C35-BLD-{len(blind_v2_bases) + 2:04d}"
        attempt = 0
        while True:
            ra, rb = _build_fixed_compound_ce_cell(
                domains=C3B.C3_BLIND_DOMAINS,
                seed_base=C35_BLIND_SEED + 600000 + attempt * 8209,
                cell_idx=cell_i,
                dom_idx=dom_idx,
                depth=depth,
                pos=pos,
                style=style,
                trans=trans,
                cell_id=cell_id,
                bid_a=bid_a,
                bid_b=bid_b,
                split_name="c3_blind_confirm_v2",
            )
            if (not dedup.has_collision(ra)) and (not dedup.has_collision(rb)):
                dedup.commit(ra)
                dedup.commit(rb)
                blind_v2_bases.extend([ra, rb])
                break
            attempt += 1
            assert attempt < 50

    assert len(blind_v2_bases) == 1080
    n_units = len({r["unit_id"] for r in blind_v2_bases})
    assert n_units == 900, n_units

    # Stratified 240-base think slice
    by_s: dict[tuple, list[dict]] = defaultdict(list)
    ce_map: dict[str, list[dict]] = defaultdict(list)
    for r in blind_v2_bases:
        if r["fallacy_class"] == "cross_entity_transfer":
            ce_map[r["cell_id"]].append(r)
        elif not r.get("paired_position_counterfactual"):
            by_s[(r["fallacy_class"], r["broken_step"], r["pair_type"])].append(r)
    rng_th = random.Random(C35_BLIND_SEED + 777)
    p_non_ce: list[dict] = []
    for s_idx, sk in enumerate(sorted(by_s)):
        take = 7 if s_idx < 16 else 6
        pool = list(by_s[sk])
        rng_th.shuffle(pool)
        p_non_ce.extend(pool[:take])
    ce_keys = sorted(ce_map)
    rng_th.shuffle(ce_keys)
    p_ce = [r for cid in ce_keys[:40] for r in ce_map[cid]]
    blind_think_v2 = p_non_ce + p_ce
    assert len(blind_think_v2) == 240

    # EXHAUSTIVE AUDITS (ALL HARD ASSERTED!)
    C3B._audit_no_only_if(blind_v2_bases, "c3_blind_confirm_v2")
    for r in blind_v2_bases:
        assert r["fallacy_class"] != "converse_error"
        assert r["difficulty_depth"] <= 3
        assert "emergency board waiver" in r["context_prompt_v"], f"Missing decoy clause on {r['base_id']}!"
        assert "emergency board waiver" in r["context_prompt_f"], f"Missing decoy clause on {r['base_id']}!"

    ce_rows = [r for r in blind_v2_bases if r["fallacy_class"] == "cross_entity_transfer"]
    assert len(ce_rows) == 240
    assert Counter(r["default_policy_position"] for r in ce_rows) == {"before_rules": 120, "after_rules": 120}
    assert Counter(r["cell_member"] for r in ce_rows) == {"A": 120, "B": 120}

    non_ce_indep = [r for r in blind_v2_bases if r["fallacy_class"] != "cross_entity_transfer" and not r.get("paired_position_counterfactual")]
    indep_chi2 = C3B._audit_factor_independence(non_ce_indep, "c3_blind_confirm_v2_indep", check_style=True)

    # Skeleton 0-overlap against c3_train
    train_v_skels = {PT1.extract_placeholder_skeleton(r["arms"]["V"], C2B.C2_TRAIN_DOMAINS) for r in c3_train_rows}
    train_ctx_skels = {PT1.extract_placeholder_skeleton(r["context_prompt_v"], C2B.C2_TRAIN_DOMAINS) for r in c3_train_rows}
    ov_v = sum(PT1.extract_placeholder_skeleton(r["arms"]["V"], C3B.C3_BLIND_DOMAINS) in train_v_skels for r in blind_v2_bases)
    ov_ctx = sum(PT1.extract_placeholder_skeleton(r["context_prompt_v"], C3B.C3_BLIND_DOMAINS) in train_ctx_skels for r in blind_v2_bases)
    assert ov_v == 0 and ov_ctx == 0, (ov_v, ov_ctx)

    g0_v2 = C3B._run_g0_tfidf_auc(blind_v2_bases)
    assert 0.42 <= g0_v2["overall_auc"] <= 0.58 and g0_v2["cf_byte_match"] == 1.0, g0_v2

    out_files = {
        "c3_blind_confirm_v2_bases.jsonl": blind_v2_bases,
        "c3_blind_think_v2_bases.jsonl": blind_think_v2,
    }
    manifest_files = {}
    for fn, rows in out_files.items():
        p = C35_DATA / fn
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8", newline="\n")
        manifest_files[fn] = {
            "count": len(rows),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }

    manifest = {
        "schema": "cedar.acl_exp_c35.datasets_manifest.v1",
        "sealed_test_touched": False,
        "only_if_bug_occurrences": 0,
        "ce_decoy_clause_present_count": f"{sum('emergency board waiver' in r['context_prompt_v'] for r in ce_rows)}/{len(ce_rows)}",
        "independent_dual_premise_verified_count": sum(bool(r.get("independent_dual_premise")) for r in blind_v2_bases),
        "total_bases": len(blind_v2_bases),
        "independent_generation_units": n_units,
        "paired_position_counterfactual_pairs": 60,
        "styles": dict(Counter(r["phrasing_style"] for r in blind_v2_bases)),
        "positions": dict(Counter(r["default_policy_position"] for r in blind_v2_bases)),
        "factor_independence_chi2": indep_chi2,
        "skeleton_overlap_vs_c3_train": {"V": ov_v, "ctx": ov_ctx},
        "g0_tfidf_auc": g0_v2,
        "files": manifest_files,
    }
    mpath = C35_DATA / "c35_datasets_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] Built & audited ACL-exp-c3.5 blind_v2 -> {mpath}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
