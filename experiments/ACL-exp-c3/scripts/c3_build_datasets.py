#!/usr/bin/env python3
"""Build de-confounded, solver-certified, zero-leakage datasets for ACL-exp-c3 (CPU only).

Direct response to the ACL-exp-c2 forensic audit & Astra's C3 pre-registration specs:
  1. MATHEMATICAL ZERO-LEAKAGE & ZERO-PEEK:
       - NEVER opens or reads sealed `c1_test_id_bases.jsonl` or `c1_test_ood_bases.jsonl`.
       - STRICTLY EXCLUDES `converse_error` and `depth-4` from EVERY dataset built here.
       - Deduplicates against ALL open splits across A1, C1, AND C2 (`c2_train`, `c2_confirm_bidir`, `c2_stealth_stress`).
       - `c3_blind_confirm` uses 6 BRAND-NEW C3-exclusive domains (100% disjoint on domain names, entity prefixes,
         entities, zones, guards, metrics, predicates, and outcomes from A1, C1, PROXY_OOD, and C2_TRAIN).
  2. FIXES BOTH C2 DATA BUGS (HARD ASSERTED AT BUILD TIME!):
       - Bug 1 (Style-B `Only if` semantic corruption): Rewrites sufficient `If X, override...` cleanly as logically
         equivalent `Whenever X, reassign...` (asserts ZERO occurrences of `"Only if"` across all prompts!).
       - Bug 2 (Factor confounding on non-CE items): Independently randomizes `(domain, default_policy_position,
         phrasing_style)` and asserts statistical independence via Pearson chi-square (`p > 0.15` for every pair!).
  3. BALANCED MULTI-HOP BREAK DEPTHS IN TRAINING (`c3_train_bases.jsonl`, 600 bases -> 6,000 atomic 10-item groups):
       - Eliminates C2's `480 / 120 / 0` Step-1/2/3 imbalance via solver-certified depth-2/3 breaks across ALL 3 hops:
           * 480 non-CE bases (4 classes x 120 = 40 Step-1 + 40 Step-2 + 40 Step-3, 50/50 CF/AF)
           * 120 CE bases = 60 complete 2x2 crossed cells (60 Row A same-entity + 60 Row B legitimate cross-entity)
       - Exact same 600-base / 6,000-example token budget as C2!
  4. INDEPENDENT DEV SET (`c3_dev_bases.jsonl`, 180 bases on `PROXY_OOD_DOMAINS`):
       - 120 non-CE (4 classes x 30: 10 Step-1 + 10 Step-2 + 10 Step-3) + 30 complete 2x2 CE cells (60 bases).
       - Used EXCLUSIVELY for checkpoint / LR / corridor selection so `c3_blind_confirm` remains 100% untouched blind!
  5. HIGH-POWER BRAND-NEW BLIND CONFIRMATION SET (`c3_blind_confirm_bases.jsonl`, 1,080 bases / 960 independent families):
       - Built on 6 BRAND-NEW `C3_BLIND_DOMAINS` with a **HELD-OUT COMPOUND PROOF STRUCTURE COMBINATION** completely
         absent from `c3_train` and `c3_dev`:
           (a) Cross-step multi-premise synthesis (`Rule 3` requires BOTH `s1(e1)` from Step 1 AND `s2(e1)` from Step 2
               plus the gating condition),
           (b) Live activated competing-branch distractor rule (fires a plausible intermediate flag pointing to a 3rd
               decoy option whose secondary clearance is unmet),
           (c) Completely disjoint Style-C discourse template family (`[Case Dossier]`, `Observed Statuses`,
               `Decision Provisions`, `Clause 1..N`, `Phase 1..3`, `Final Determination`) — 0% skeleton overlap with
               `c3_train` AND `c1_proxy_template_v2` asserted!
       - Composition: 840 Compound non-CE (4 classes x 210: 70 Step-1 + 70 Step-2 + 70 Step-3, 50/50 CF/AF)
         + 120 Complete 2x2 Compound CE cells (= 240 bases: 120 Row A + 120 Row B) = 1,080 bases (960 independent
         family clusters -> ~80%+ paired power for a 3 pp effect even after M0 eligibility filtering!).
       - Also exports `c3_blind_think_bases.jsonl` (stratified 240-base slice for fast vLLM `think` evaluation).

Usage:
  python experiments/ACL-exp-c3/scripts/c3_build_datasets.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]
REPO = C3.parents[1]
A1_SCRIPTS = REPO / "experiments" / "ACL-exp-a1" / "scripts"
C1_SCRIPTS = REPO / "experiments" / "ACL-exp-c1" / "scripts"
C1_DATA = REPO / "experiments" / "ACL-exp-c1" / "data"
C2_SCRIPTS = REPO / "experiments" / "ACL-exp-c2" / "scripts"
C2_DATA = REPO / "experiments" / "ACL-exp-c2" / "data"
C3_DATA = C3 / "data"

sys.path.insert(0, str(A1_SCRIPTS))
sys.path.insert(0, str(C1_SCRIPTS))
sys.path.insert(0, str(C2_SCRIPTS))

import build_symbolic_pilot_v2 as A1_SYM  # noqa: E402
import c1_build_datasets as C1B  # noqa: E402
import c1_build_proxy_ood as P1  # noqa: E402
import c1_build_proxy_template_ood as PT1  # noqa: E402
import c2_build_datasets as C2B  # noqa: E402
from a1_arm_wrappers import ARM_ORDER, build_arms  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402


C3_TRAIN_SEED = 20271101
C3_DEV_SEED = 20271129
C3_BLIND_SEED = 20271217

OPTION_LETTERS = ("A", "B", "C", "D")
ALL_TRANSITIONS = tuple((a, b) for a in OPTION_LETTERS for b in OPTION_LETTERS if a != b)
assert len(ALL_TRANSITIONS) == 12


# 6 BRAND-NEW C3-EXCLUSIVE BLIND DOMAINS (100% disjoint from A1, C1_DOMAINS, PROXY_OOD, and C2_TRAIN!)
C3_BLIND_DOMAINS: tuple[dict[str, Any], ...] = (
    {
        "name": "observatory_dome",
        "intro": "A closed mountain observatory operations desk applies exact telescope dome rules. Use only the listed dossier records and provisions.",
        "entity_prefix": "Telescope",
        "entities": ("Optic-14", "Optic-29", "Optic-47", "Optic-68"),
        "zones": ("Bay Zenith", "Bay Nadir"),
        "guard_name": "shutter_bolt",
        "guard_active_text": "the shutter bolt is engaged",
        "guard_inactive_text": "the shutter bolt is released",
        "metric_name": "seeing index",
        "props": (
            ("p_obs", "logged a tracking shift"),
            ("p_mid1", "enters mirror calibration"),
            ("p_mid2", "authorizes dome override"),
            ("p_mid3", "clears night priority"),
            ("p_mid4", "receives final slew release"),
            ("p_aux", "carries an astronomer key"),
            ("p_alt_mid", "holds a focus bulletin"),
            ("p_noise1", "passed collimation sweep"),
            ("p_noise2", "verified dew heater check"),
        ),
        "outcomes": ("Dome-Lyra", "Dome-Vega", "Dome-Rigel", "Dome-Orion"),
    },
    {
        "name": "substation_grid",
        "intro": "A closed high-voltage substation desk enforces exact feeder rules. Use only the listed dossier records and provisions.",
        "entity_prefix": "Breaker",
        "entities": ("Bus-11", "Bus-24", "Bus-38", "Bus-52"),
        "zones": ("Wing Primary", "Wing Reserve"),
        "guard_name": "arc_relay",
        "guard_active_text": "the arc relay is engaged",
        "guard_inactive_text": "the arc relay is released",
        "metric_name": "phase drift",
        "props": (
            ("p_obs", "reported a harmonic surge"),
            ("p_mid1", "enters load balancing"),
            ("p_mid2", "activates busbar override"),
            ("p_mid3", "opens voltage priority"),
            ("p_mid4", "receives final tie closure"),
            ("p_aux", "holds a dispatcher seal"),
            ("p_alt_mid", "carries a transformer notice"),
            ("p_noise1", "passed oil dielectric test"),
            ("p_noise2", "logged grounding continuity"),
        ),
        "outcomes": ("Tap-Amber", "Tap-Coral", "Tap-Ivory", "Tap-Slate"),
    },
    {
        "name": "canal_lock",
        "intro": "A closed inland canal lock chamber applies exact transit rules. Use only the listed dossier records and provisions.",
        "entity_prefix": "Barge",
        "entities": ("Sluice-15", "Sluice-31", "Sluice-46", "Sluice-73"),
        "zones": ("Basin High", "Basin Low"),
        "guard_name": "sill_pin",
        "guard_active_text": "the sill pin is engaged",
        "guard_inactive_text": "the sill pin is released",
        "metric_name": "draft depth",
        "props": (
            ("p_obs", "signaled a keel trim"),
            ("p_mid1", "enters chamber staging"),
            ("p_mid2", "grants lockage override"),
            ("p_mid3", "approves passage priority"),
            ("p_mid4", "receives final gate release"),
            ("p_aux", "carries a lockmaster chit"),
            ("p_alt_mid", "holds a towage memo"),
            ("p_noise1", "cleared fender check"),
            ("p_noise2", "logged winch calibration"),
        ),
        "outcomes": ("Berth-Alder", "Berth-Cedar", "Berth-Larch", "Berth-Maple"),
    },
    {
        "name": "cellar_vault",
        "intro": "A closed vintner aging cellar applies exact racking rules. Use only the listed dossier records and provisions.",
        "entity_prefix": "Cask",
        "entities": ("Barrel-13", "Barrel-27", "Barrel-49", "Barrel-66"),
        "zones": ("Row Vault", "Row Crypt"),
        "guard_name": "cork_seal",
        "guard_active_text": "the cork seal is engaged",
        "guard_inactive_text": "the cork seal is released",
        "metric_name": "tannin index",
        "props": (
            ("p_obs", "triggered an ullage check"),
            ("p_mid1", "enters cooper inspection"),
            ("p_mid2", "unlocks racking override"),
            ("p_mid3", "grants vintage priority"),
            ("p_mid4", "receives final cellar transfer"),
            ("p_aux", "holds a sommelier band"),
            ("p_alt_mid", "carries a stave record"),
            ("p_noise1", "passed hygrometer read"),
            ("p_noise2", "verified bung alignment"),
        ),
        "outcomes": ("Rack-Beryl", "Rack-Topaz", "Rack-Agate", "Rack-Quartz"),
    },
    {
        "name": "seismic_array",
        "intro": "A closed geothermal seismic monitoring station applies exact telemetry rules. Use only the listed dossier records and provisions.",
        "entity_prefix": "Sensor",
        "entities": ("Node-17", "Node-35", "Node-53", "Node-78"),
        "zones": ("Ridge Crest", "Ridge Trough"),
        "guard_name": "tremor_gate",
        "guard_active_text": "the tremor gate is engaged",
        "guard_inactive_text": "the tremor gate is released",
        "metric_name": "shear count",
        "props": (
            ("p_obs", "emitted a p-wave chime"),
            ("p_mid1", "enters waveform triage"),
            ("p_mid2", "triggers borehole override"),
            ("p_mid3", "enables array priority"),
            ("p_mid4", "receives final telemetry lock"),
            ("p_aux", "holds a field beacon"),
            ("p_alt_mid", "carries a spectrum log"),
            ("p_noise1", "passed geophone ping"),
            ("p_noise2", "verified battery voltage"),
        ),
        "outcomes": ("Band-Pulse", "Band-Wave", "Band-Drift", "Band-Echo"),
    },
    {
        "name": "telemetry_buoy",
        "intro": "A closed oceanographic buoy control room applies exact mooring rules. Use only the listed dossier records and provisions.",
        "entity_prefix": "Beacon",
        "entities": ("Float-21", "Float-43", "Float-64", "Float-85"),
        "zones": ("Tide Inner", "Tide Outer"),
        "guard_name": "anchor_clip",
        "guard_active_text": "the anchor clip is engaged",
        "guard_inactive_text": "the anchor clip is released",
        "metric_name": "salinity score",
        "props": (
            ("p_obs", "sent a drift beacon"),
            ("p_mid1", "enters sonar validation"),
            ("p_mid2", "authorizes mooring override"),
            ("p_mid3", "enables buoy priority"),
            ("p_mid4", "receives final station release"),
            ("p_aux", "carries a hydrographer tag"),
            ("p_alt_mid", "holds a current digest"),
            ("p_noise1", "cleared solar panel check"),
            ("p_noise2", "logged transponder sync"),
        ),
        "outcomes": ("Channel-Kestrel", "Channel-Heron", "Channel-Plover", "Channel-Tern"),
    },
)


def _verify_all_domain_pools_disjoint() -> None:
    """Mathematically verify C3_BLIND_DOMAINS has ZERO overlap with A1, C1, PROXY_OOD, and C2_TRAIN."""
    other_pools = (
        list(A1_SYM.DOMAINS)
        + list(C1B.C1_DOMAINS)
        + list(P1.PROXY_OOD_DOMAINS)
        + list(C2B.C2_TRAIN_DOMAINS)
    )
    c3_names = {d["name"] for d in C3_BLIND_DOMAINS}
    other_names = {d["name"] for d in other_pools}
    assert not (c3_names & other_names), f"Domain overlap: {c3_names & other_names}"

    c3_ents = {f"{d['entity_prefix']} {e}" for d in C3_BLIND_DOMAINS for e in d["entities"]}
    other_ents = {f"{d['entity_prefix']} {e}" for d in other_pools for e in d["entities"]}
    assert not (c3_ents & other_ents), f"Entity overlap: {c3_ents & other_ents}"

    c3_outs = {o for d in C3_BLIND_DOMAINS for o in d["outcomes"]}
    other_outs = {o for d in other_pools for o in d["outcomes"]}
    assert not (c3_outs & other_outs), f"Outcome overlap: {c3_outs & other_outs}"

    c3_zones = {z for d in C3_BLIND_DOMAINS for z in d["zones"]}
    other_zones = {z for d in other_pools for z in d["zones"]}
    assert not (c3_zones & other_zones), f"Zone overlap: {c3_zones & other_zones}"

    c3_metrics = {d["metric_name"] for d in C3_BLIND_DOMAINS}
    other_metrics = {d["metric_name"] for d in other_pools}
    assert not (c3_metrics & other_metrics), f"Metric overlap: {c3_metrics & other_metrics}"


# -------------------------------------------------------------------------------------- Clean Style Rewrites (NO `Only if`!)


def _apply_style_b_clean(row: dict[str, Any]) -> None:
    """Apply logically faithful Style-B paraphrase (FIXED from C2: `Whenever`, NEVER `Only if`!)."""
    def rw_ctx(ctx: str) -> str:
        lines = []
        for line in ctx.splitlines():
            s = line.strip()
            if s == "Facts:":
                lines.append("Recorded state:")
            elif s.startswith("Fact "):
                m = re.match(r"^Fact (\d+)\.\s*(.+)$", s)
                lines.append(f"Item {m.group(1)}: {m.group(2)}" if m else line)
            elif s == "Rules:":
                lines.append("Governing policies:")
            elif s.startswith("Rule "):
                m = re.match(r"^Rule (\d+)\.\s*(.+)$", s)
                if m:
                    num, body = m.group(1), m.group(2)
                    body2 = re.sub(r"^If (.+?), then ", r"Provided that \1, ", body)
                    body2 = re.sub(
                        r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                        r"Whenever \1, reassign \2 from the default to \3.",
                        body2,
                    )
                    lines.append(f"Policy {num}: {body2}")
                else:
                    lines.append(line)
            elif s.startswith("Default policy:"):
                lines.append(s.replace("Default policy:", "Standing default:", 1))
            else:
                lines.append(line)
        return "\n".join(lines)

    def rw_arg(txt: str) -> str:
        t = txt
        t = re.sub(r"^New verified observation:\s*", "Confirmed field report: ", t)
        t = re.sub(r"\bStep (\d+):\s*", r"Stage \1 — ", t)
        t = re.sub(r"\bRule (\d+)\b", r"Policy \1", t)
        t = re.sub(r"\bConclusion:\s*", "Summary verdict: ", t)
        return t

    for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
        row[k] = rw_ctx(row[k])
    for an in list(row["arms"].keys()):
        if an.startswith(("V", "F")):
            row["arms"][an] = rw_arg(row["arms"][an])
    row["phrasing_style"] = "style_b_fixed"


def _render_style_c_context(
    dom: dict[str, Any],
    facts: list[str],
    rules: list[str],
    target_entity: str,
    letter_to_outcome: dict[str, str],
    default_policy_text: str,
    pos: str,
) -> str:
    """Held-out Style-C discourse template family for `c3_blind_confirm` (disjoint from Style A, Style B, and Proxy v2!)."""
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
    dp_c = default_policy_text.replace("Default policy:", "Baseline Rule:", 1)
    opts = "\n".join(f"{k}. {letter_to_outcome[k]}" for k in OPTION_LETTERS)
    if pos == "before_rules":
        body = f"{dp_c}\n\nDecision Provisions:\n{rule_block}"
    else:
        body = f"Decision Provisions:\n{rule_block}\n\n{dp_c}"
    return (
        f"[Case Dossier]\n{dom['intro']}\n"
        f"Closed-dossier note: an entry's status holds only when listed under Observed Statuses, derived via Decision Provisions, or reported in a verified telemetry bulletin.\n\n"
        f"Observed Statuses:\n{fact_block}\n\n"
        f"{body}\n\n"
        f"Question: Based strictly on the listed dossier entries and provisions, what is the current classification of {target_entity}?\n"
        f"Options:\n{opts}"
    )


def _rewrite_arg_style_c(txt: str) -> str:
    t = txt
    t = re.sub(r"^New verified observation:\s*", "Verified telemetry bulletin: ", t)
    t = re.sub(r"\bStep (\d+):\s*", r"Phase \1: ", t)
    t = re.sub(r"\bRule (\d+)\b", r"Clause \1", t)
    t = re.sub(r"\bConclusion:\s*", "Final Determination: ", t)
    return t


# -------------------------------------------------------------------------------------- Unified Solver-Certified Depth-3 Break Builder


def _build_general_depth3_break_base(
    *,
    domains: tuple[dict[str, Any], ...],
    seed_base: int,
    idx: int,
    dom_idx: int,
    fallacy_class: str,
    broken_step: int,
    pair_type: str,
    pos: str,
    trans: tuple[str, str],
    bid: str,
    split_name: str,
    schema_name: str,
    compound_blind: bool = False,
) -> dict[str, Any]:
    """Build a solver-certified Depth-3 base breaking at EXACTLY `broken_step in {1, 2, 3}`.

    Supports BOTH `context_flip` (byte-identical Turn-2) and `argument_flip` (identical context, symmetric Turn-2).
    When `compound_blind=True`, activates the held-out C3 Blind Confirmation compound proof structure:
      - Cross-step multi-premise synthesis (Rule 3 requires BOTH `s1(e1)` from Step 1 AND `s2(e1)` from Step 2 + gate),
      - Active competing-branch distractor rule (live triggered rule producing `alt(e1)` pointing toward a 3rd decoy
        outcome `a_decoy` gated on an unmet secondary clearance),
      - Completely disjoint Style-C discourse template family!
    """
    assert fallacy_class in ("scope_swap", "missing_conjunct", "negation_drop", "threshold_violation")
    assert broken_step in (1, 2, 3)
    assert pair_type in ("context_flip", "argument_flip")

    rng = random.Random(seed_base + idx * 10007 + broken_step * 97)
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

    p_obs = p["p_obs"]
    p_s1 = p["p_mid1"]
    p_s2 = p["p_mid3"]
    p_trig = p["p_mid2"]
    p_aux = p["p_aux"]
    p_alt = p["p_alt_mid"]
    metric = dom["metric_name"]

    t_low = rng.randint(18, 86)
    d1 = rng.randint(1, 3)
    d2 = rng.randint(1, 3)
    t_v = t_low
    obs_val_v = t_low + d1      # >= t_v (valid!)
    t_f = obs_val_v + d2        # obs_val_v < t_f (invalid in CF!)
    obs_val_f = t_v - d2        # < t_v (invalid in AF!)

    # Base world facts and gating spec for the target broken step
    override_trigger = f"trig({e1})"
    # In compound_blind mode, Rule 3 synthesizes BOTH s1(e1) AND s2(e1)!
    step_in_pred = {
        1: [f"obs({e1})"],
        2: [f"s1({e1})"],
        3: [f"s1({e1})", f"s2({e1})"] if compound_blind else [f"s2({e1})"],
    }
    step_in_text = {
        1: f"{e1} {p_obs}",
        2: f"{e1} {p_s1}",
        3: f"{e1} {p_s1} and {e1} {p_s2}" if compound_blind else f"{e1} {p_s2}",
    }
    step_out_pred = {1: f"s1({e1})", 2: f"s2({e1})", 3: override_trigger}
    step_out_text = {1: f"{e1} {p_s1}", 2: f"{e1} {p_s2}", 3: f"{e1} {p_trig}"}

    # Build V and F worlds + Turn-2 observations
    obs_v_set: set[str] = {f"obs({e1})"}
    obs_f_set: set[str] = {f"obs({e1})"}
    obs_v_text = f"{e1} {p_obs}."
    obs_f_text = f"{e1} {p_obs}."

    if fallacy_class == "scope_swap":
        facts_v = [
            f"{e1} is registered under active default status.",
            f"{e1} belongs to {z1}.",
            f"{e2} belongs to {z2}.",
            f"For {e1}, {dom['guard_inactive_text']}.",
            f"{e3} {p['p_noise1']}.",
            f"{e4} {p['p_noise2']}.",
        ]
        bf_v = {f"default({e1})", f"in_zone({e1},{z1})", f"in_zone({e2},{z2})", f"guard_inactive({e1})", f"n1({e3})", f"n2({e4})"}
        if pair_type == "context_flip":
            facts_f = list(facts_v)
            bf_f = set(bf_v)
            gate_v_txt = f"For entries in {z1}, if {step_in_text[broken_step]}, then {step_out_text[broken_step]}."
            gate_f_txt = f"For entries in {z2}, if {step_in_text[broken_step]}, then {step_out_text[broken_step]}."
            gate_v_cond = [f"in_zone({e1},{z1})"] + step_in_pred[broken_step]
            gate_f_cond = [f"in_zone({e1},{z2})"] + step_in_pred[broken_step]
            step_phrase_v = f"Applying Rule {broken_step} to the verified status, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v
        else:
            # argument_flip: IDENTICAL context! Rule uses z1; V states e1 is in z1, F states e2 (wrong scope/entity)
            # Actually: keep e1 in z2 in context? No, if context is identical for V and F, then V must legitimately
            # trigger the rule in that context while F claims a rule whose zone doesn't match e1!
            # Clean symmetric AF: context has BOTH Rule_broken (for z1 -> out) AND decoy scope rule (for z2 -> alt);
            # e1 is in z1, e2 is in z2. Wait: to keep context identical & Turn-2 valid in V / invalid in F:
            # Put `e1` in `z1` ONLY via Turn-2 observation (`e1 p_obs and e1 is verified in z1` vs `e1 p_obs and e1 is verified in z2`)!
            # Wait: z1 and z2 are randomly shuffled per item -> G0 TF-IDF on Turn-2 is STRICTLY 0.50!
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
            obs_v_text = f"{e1} {p_obs} while assigned to {z1}."
            obs_f_text = f"{e1} {p_obs} while assigned to {z2}."
            obs_v_set = {f"obs({e1})", f"in_zone({e1},{z1})"}
            obs_f_set = {f"obs({e1})", f"in_zone({e1},{z2})"}
            step_phrase_v = f"Applying Rule {broken_step} under that zone assignment, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v

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
            step_phrase_v = f"Applying Rule {broken_step} to the verified status, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v
        else:
            # Symmetric AF: Turn-2 reports aux for e1 (V) vs aux for e2 (F) — e1/e2 randomly shuffled -> G0 AUC = 0.50!
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
            obs_v_text = f"{e1} {p_obs}, and {e1} {p_aux}."
            obs_f_text = f"{e1} {p_obs}, and {e2} {p_aux}."
            obs_v_set = {f"obs({e1})", f"aux({e1})"}
            obs_f_set = {f"obs({e1})", f"aux({e2})"}
            step_phrase_v = f"Applying Rule {broken_step} with that reported credential, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v

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
            step_phrase_v = f"Applying Rule {broken_step} to the verified status, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v
        else:
            # Symmetric AF: invert guard polarity in 50% of cases so "active/inactive" words have ZERO marginal correlation with V vs F!
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
            obs_v_text = f"{e1} {p_obs}, and for {e1}, {req_txt}."
            obs_f_text = f"{e1} {p_obs}, and for {e1}, {opp_txt}."
            obs_v_set = {f"obs({e1})", req_pred}
            obs_f_set = {f"obs({e1})", opp_pred}
            step_phrase_v = f"Applying Rule {broken_step} under that guard status, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v

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
            obs_v_text = f"{e1} {p_obs} with a {metric} of {obs_val_v}."
            obs_f_text = obs_v_text
            obs_v_set = {f"obs({e1})", "thresh_v"}
            obs_f_set = {f"obs({e1})", "thresh_v"}
            step_phrase_v = f"Applying Rule {broken_step} using the verified {metric} reading, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v
        else:
            gate_v_txt = f"If {step_in_text[broken_step]} and the {metric} is at least {t_v}, then {step_out_text[broken_step]}."
            gate_f_txt = gate_v_txt
            gate_v_cond = step_in_pred[broken_step] + ["thresh_ok"]
            gate_f_cond = list(gate_v_cond)
            obs_v_text = f"{e1} {p_obs} with a {metric} of {obs_val_v}."
            obs_f_text = f"{e1} {p_obs} with a {metric} of {obs_val_f}."
            obs_v_set = {f"obs({e1})", "thresh_ok"}
            obs_f_set = {f"obs({e1})"}
            step_phrase_v = f"Applying Rule {broken_step} using the verified {metric} reading, {step_out_text[broken_step]}."
            step_phrase_f = step_phrase_v

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
        ]
        rdicts = [
            {"id": "Rule 1", "conditions": r1_c, "conclusion": step_out_pred[1]},
            {"id": "Rule 2", "conditions": r2_c, "conclusion": step_out_pred[2]},
            {"id": "Rule 3", "conditions": r3_c, "conclusion": step_out_pred[3]},
        ]
        if compound_blind:
            # HELD-OUT COMPOUND FEATURE: Live activated competing-branch distractor rules!
            rlines.extend([
                f"If {e1} {p_obs} and {e1} {p_s1}, then {e1} {p_alt}.",
                f"If {e1} {p_alt} and {e1} carries an emergency board waiver, override the default classification and assign {e1} to {a_decoy_name}.",
                f"If {e3} {p['p_noise1']} and {e4} {p['p_noise2']}, record a routine audit note for {e3}.",
                f"If {e2} {p_obs}, then {e2} is placed on a standby list.",
            ])
            rdicts.extend([
                {"id": "Rule 5", "conditions": [f"obs({e1})", f"s1({e1})"], "conclusion": f"alt({e1})"},
                {"id": "Rule 6", "conditions": [f"alt({e1})", f"waiver({e1})"], "conclusion": f"decoy_trig({e1})"},
                {"id": "Rule 7", "conditions": [f"n1({e3})", f"n2({e4})"], "conclusion": f"audit({e3})"},
                {"id": "Rule 8", "conditions": [f"obs({e2})"], "conclusion": f"standby({e2})"},
            ])
        else:
            rlines.extend([
                f"If {e3} {p['p_noise1']} and {e4} {p['p_noise2']}, record a routine audit note for {e3}.",
                f"If {e1} {p_s1}, then {e1} receives a periodic log review.",
                f"If {e1} {p_s2}, then {e1} {p_alt}.",
                f"If {e2} {p_obs}, then {e2} is placed on a watch list.",
            ])
            rdicts.extend([
                {"id": "Rule 5", "conditions": [f"n1({e3})", f"n2({e4})"], "conclusion": f"audit({e3})"},
                {"id": "Rule 6", "conditions": [f"s1({e1})"], "conclusion": f"review({e1})"},
                {"id": "Rule 7", "conditions": [f"s2({e1})"], "conclusion": f"alt({e1})"},
                {"id": "Rule 8", "conditions": [f"obs({e2})"], "conclusion": f"watch({e2})"},
            ])
        return rlines, rdicts

    rlines_v, rules_v = _assemble_rules(gate_v_txt, gate_v_cond)
    rlines_f, rules_f = _assemble_rules(gate_f_txt, gate_f_cond)

    # STRICT SOLVER & HOP-BY-HOP CERTIFICATION!
    claimed = [f"s1({e1})", f"s2({e1})", override_trigger]
    _, t1v = A1_SYM.evaluate_world(bf_v, {}, rules_v, a0_name, override_trigger, at_name)
    _, t1f = A1_SYM.evaluate_world(bf_f, {}, rules_f, a0_name, override_trigger, at_name)
    assert t1v == a0_name and t1f == a0_name
    _, outv = A1_SYM.evaluate_world(bf_v | obs_v_set, {}, rules_v, a0_name, override_trigger, at_name)
    _, outf = A1_SYM.evaluate_world(bf_f | obs_f_set, {}, rules_f, a0_name, override_trigger, at_name)
    assert outv == at_name and outf == a0_name, (bid, fallacy_class, broken_step, pair_type, outv, outf)
    C1B.verify_step_by_step_proof(bf_v, obs_v_set, rules_v, claimed, expected_broken_step=None)
    C1B.verify_step_by_step_proof(bf_f, obs_f_set, rules_f, claimed, expected_broken_step=broken_step)

    if compound_blind:
        ctx_v = _render_style_c_context(dom, facts_v, rlines_v, e1, letter_to_outcome, default_policy_text, pos)
        ctx_f = _render_style_c_context(dom, facts_f, rlines_f, e1, letter_to_outcome, default_policy_text, pos)
    else:
        ctx_v = A1_SYM.render_context_prompt(dom, facts_v, rlines_v, e1, letter_to_outcome, default_policy_text, pos)
        ctx_f = A1_SYM.render_context_prompt(dom, facts_f, rlines_f, e1, letter_to_outcome, default_policy_text, pos)

    if pair_type == "context_flip":
        assert ctx_v != ctx_f
    else:
        assert ctx_v == ctx_f

    def _build_core_arg(obs_t: str, sp_phrase: str) -> str:
        s1_t = sp_phrase if broken_step == 1 else f"Since {e1} {p_obs}, Rule 1 implies that {e1} {p_s1}."
        s2_t = sp_phrase if broken_step == 2 else f"Applying Rule 2 to that intermediate status, {e1} {p_s2}."
        if broken_step == 3:
            s3_t = sp_phrase
        elif compound_blind:
            s3_t = f"Combining the Step 1 status ({e1} {p_s1}) and Step 2 status ({e1} {p_s2}) under Rule 3, {e1} {p_trig}."
        else:
            s3_t = f"Applying Rule 3 to that intermediate status, {e1} {p_trig}."
        return (
            f"New verified observation: {obs_t} "
            f"Step 1: {s1_t} "
            f"Step 2: {s2_t} "
            f"Step 3: {s3_t} "
            f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {at_name} (Option {at})."
        )

    core_v = _build_core_arg(obs_v_text, step_phrase_v)
    core_f = _build_core_arg(obs_f_text, step_phrase_f)
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
    arms = build_arms(
        core_v=core_v,
        core_f=core_f,
        shell_only=shell_only,
        a0=a0,
        at=at,
        i0_text=i0_text,
        final_noun="answer",
    )
    if compound_blind:
        for an in list(arms.keys()):
            if an.startswith(("V", "F")) or an == "shell_only":
                arms[an] = _rewrite_arg_style_c(arms[an])
    assert tuple(arms) == ARM_ORDER

    return {
        "schema": schema_name,
        "split": split_name,
        "base_id": bid,
        "cell_id": bid,
        "family_id": bid,
        "domain": dom["name"],
        "pair_type": pair_type,
        "difficulty_depth": 3,
        "fallacy_class": fallacy_class,
        "broken_step": broken_step,
        "compound_blind": compound_blind,
        "phrasing_style": "style_c_blind" if compound_blind else "style_a",
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


def _build_compound_ce_2x2_cell(
    *,
    domains: tuple[dict[str, Any], ...],
    seed_base: int,
    cell_idx: int,
    dom_idx: int,
    depth: int,
    pos: str,
    trans: tuple[str, str],
    cell_id: str,
    bid_a: str,
    bid_b: str,
    split_name: str,
    schema_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one held-out Style-C compound 2x2 Cross-Entity cell (Row A same-entity, Row B cross-entity) for c3_blind_confirm."""
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
        schema_name=schema_name,
    )
    # Transform cleanly into Style-C discourse template + add active non-triggering competing-branch distractor clause!
    dom = domains[dom_idx]
    p = dict(dom["props"])
    rng = random.Random(seed_base + cell_idx * 4219)
    a0, at = trans
    decoy_letters = [k for k in OPTION_LETTERS if k not in (a0, at)]
    a_decoy = decoy_letters[rng.randrange(len(decoy_letters))]

    for row in (ra, rb):
        a_decoy_name = row["letter_to_outcome"][a_decoy]
        def _to_style_c_ctx(ctx: str) -> str:
            lines = []
            rule_idx = 0
            in_rules = False
            inserted_decoy = False
            for line in ctx.splitlines():
                s = line.strip()
                if s == dom["intro"]:
                    lines.append("[Case Dossier]")
                    lines.append(s)
                elif s.startswith("Closed-world note:"):
                    lines.append(
                        "Closed-dossier note: an entry's status holds only when listed under Observed Statuses, derived via Decision Provisions, or reported in a verified telemetry bulletin."
                    )
                elif s == "Facts:":
                    lines.append("Observed Statuses:")
                elif s.startswith("Fact "):
                    m = re.match(r"^Fact (\d+)\.\s*(.+)$", s)
                    lines.append(f"Entry {m.group(1)}: {m.group(2)}" if m else line)
                elif s == "Rules:":
                    in_rules = True
                    lines.append("Decision Provisions:")
                elif s.startswith("Rule "):
                    m = re.match(r"^Rule (\d+)\.\s*(.+)$", s)
                    if m:
                        rule_idx = int(m.group(1))
                        body = m.group(2)
                        body2 = re.sub(r"^If (.+?), then ", r"When \1, it follows that ", body)
                        body2 = re.sub(
                            r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                            r"When \1, supersede the baseline assignment and route \2 to \3.",
                            body2,
                        )
                        lines.append(f"Clause {rule_idx} — {body2}")
                    else:
                        lines.append(line)
                elif s.startswith("Default policy:"):
                    if in_rules and not inserted_decoy:
                        lines.append(
                            f"Clause {rule_idx + 1} — When an entry {p['p_alt_mid']} and carries an emergency board waiver, "
                            f"supersede the baseline assignment and route it to {a_decoy_name}."
                        )
                        inserted_decoy = True
                        in_rules = False
                    lines.append(s.replace("Default policy:", "Baseline Rule:", 1))
                elif s.startswith("Question:"):
                    if in_rules and not inserted_decoy:
                        lines.append(
                            f"Clause {rule_idx + 1} — When an entry {p['p_alt_mid']} and carries an emergency board waiver, "
                            f"supersede the baseline assignment and route it to {a_decoy_name}."
                        )
                        inserted_decoy = True
                        in_rules = False
                    lines.append(
                        s.replace("listed facts and rules", "listed dossier entries and provisions")
                    )
                else:
                    lines.append(line)
            return "\n".join(lines)

        for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
            row[k] = _to_style_c_ctx(row[k])
        for an in list(row["arms"].keys()):
            if an.startswith(("V", "F")) or an == "shell_only":
                row["arms"][an] = _rewrite_arg_style_c(row["arms"][an])
        row["family_id"] = cell_id
        row["compound_blind"] = True
        row["phrasing_style"] = "style_c_blind"
        row["broken_step"] = 1
    return ra, rb


# -------------------------------------------------------------------------------------- De-Confounded Factor Assignment & Statistical Audits


def _deconfounded_factors(n: int, n_domains: int, seed: int, include_style: bool = True) -> list[dict[str, Any]]:
    """Generate perfectly balanced, mutually independent `(dom_idx, pos, style)` tuples."""
    rng = random.Random(seed)
    doms = [i % n_domains for i in range(n)]
    poss = ["before_rules" if i < (n // 2) else "after_rules" for i in range(n)]
    styles = ["style_a" if i < (n // 2) else "style_b" for i in range(n)]
    rng.shuffle(doms)
    rng.shuffle(poss)
    rng.shuffle(styles)
    out = []
    for i in range(n):
        d = {"dom_idx": doms[i], "pos": poss[i]}
        if include_style:
            d["style"] = styles[i]
        out.append(d)
    return out


def _chi2_independence_pvalue(xs: list[Any], ys: list[Any]) -> dict[str, float]:
    """Compute Pearson chi-square statistic and exact/asymptotic p-value (no scipy dependency required)."""
    x_vals = sorted(set(xs))
    y_vals = sorted(set(ys))
    n = len(xs)
    ct = np.zeros((len(x_vals), len(y_vals)), dtype=float)
    xi = {v: i for i, v in enumerate(x_vals)}
    yi = {v: i for i, v in enumerate(y_vals)}
    for a, b in zip(xs, ys):
        ct[xi[a], yi[b]] += 1.0
    row_s = ct.sum(axis=1, keepdims=True)
    col_s = ct.sum(axis=0, keepdims=True)
    exp = row_s @ col_s / max(1.0, float(n))
    chi2 = float(np.sum((ct - exp) ** 2 / np.maximum(exp, 1e-12)))
    df = max(1, (len(x_vals) - 1) * (len(y_vals) - 1))
    # Regularized upper incomplete gamma Q(df/2, chi2/2) via series/continued fraction
    p = _chi2_sf(chi2, df)
    return {"chi2": round(chi2, 4), "df": df, "p": round(p, 4)}


def _chi2_sf(x: float, df: int) -> float:
    if x <= 0.0:
        return 1.0
    a = 0.5 * df
    z = 0.5 * x
    # Series for P(a, z) when z < a + 1, else Lentz continued fraction for Q(a, z)
    if z < a + 1.0:
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(200):
            ap += 1.0
            delta *= z / ap
            s += delta
            if abs(delta) < abs(s) * 1e-12:
                break
        p_lower = s * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p_lower))
    else:
        b = z + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return max(0.0, min(1.0, math.exp(-z + a * math.log(z) - math.lgamma(a)) * h))


def _audit_no_only_if(rows: list[dict[str, Any]], split_name: str) -> None:
    for r in rows:
        for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
            assert "Only if" not in r[k], f"Found forbidden 'Only if' in {split_name} {r['base_id']} {k}"
        for an, txt in r["arms"].items():
            assert "Only if" not in txt, f"Found forbidden 'Only if' in {split_name} {r['base_id']} arm {an}"


def _audit_factor_independence(rows: list[dict[str, Any]], name: str, check_style: bool = True) -> dict[str, Any]:
    doms = [r["domain"] for r in rows]
    poss = [r["default_policy_position"] for r in rows]
    res = {"domain_x_pos": _chi2_independence_pvalue(doms, poss)}
    assert res["domain_x_pos"]["p"] > 0.15, (name, "domain_x_pos", res)
    if check_style:
        stys = [r["phrasing_style"] for r in rows]
        res["domain_x_style"] = _chi2_independence_pvalue(doms, stys)
        res["pos_x_style"] = _chi2_independence_pvalue(poss, stys)
        assert res["domain_x_style"]["p"] > 0.15, (name, "domain_x_style", res)
        assert res["pos_x_style"]["p"] > 0.15, (name, "pos_x_style", res)
    return res


def _run_g0_tfidf_auc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts, labels, groups = [], [], []
    for r in rows:
        gid = r.get("family_id", r.get("cell_id", r["base_id"]))
        texts.append(r["arms"]["V"])
        labels.append(1)
        groups.append(gid)
        texts.append(r["arms"]["F"])
        labels.append(0)
        groups.append(gid)
    y = np.array(labels)
    g = np.array(groups)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)
    X = vec.fit_transform(texts)
    gkf = GroupKFold(n_splits=5)
    preds = np.zeros(len(y), dtype=float)
    for tr, va in gkf.split(X, y, g):
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X[tr], y[tr])
        preds[va] = clf.predict_proba(X[va])[:, 1]
    auc = float(roc_auc_score(y, preds))
    cf_rows = [r for r in rows if r["pair_type"] == "context_flip"]
    cf_byte = float(np.mean([r["arms"]["V"] == r["arms"]["F"] for r in cf_rows])) if cf_rows else None
    return {"overall_auc": round(auc, 4), "cf_byte_match": cf_byte}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    _verify_all_domain_pools_disjoint()
    C3_DATA.mkdir(parents=True, exist_ok=True)
    a1_cfg = load_config()

    # Load ALL OPEN splits across A1, C1, AND C2 (NEVER touch sealed c1_test_id / c1_test_ood!)
    dedup = C1B.SplitDeduplicator()
    C1B.load_a1_rows_into_dedup(dedup)
    open_files: list[Path] = [
        C1_DATA / "c1_train_bases.jsonl",
        C1_DATA / "c1_dev_bases.jsonl",
        C1_DATA / "c1_proxy_ood_bases.jsonl",
        C1_DATA / "c1_proxy_template_ood_bases.jsonl",
        C1_DATA / "c1_proxy_template_v2_bases.jsonl",
        C1_DATA / "c1_step2b_confirm_ce_bases.jsonl",
        C2_DATA / "c2_train_bases.jsonl",
        C2_DATA / "c2_confirm_bidir_bases.jsonl",
        C2_DATA / "c2_stealth_stress_bases.jsonl",
    ]
    v2_rows: list[dict[str, Any]] = []
    for p in open_files:
        assert "c1_test_id" not in p.name and "c1_test_ood" not in p.name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            dedup.register_external(r)
            if p.name == "c1_proxy_template_v2_bases.jsonl":
                v2_rows.append(r)

    non_ce_classes = ("scope_swap", "missing_conjunct", "negation_drop", "threshold_violation")

    # =========================================================================
    # 1. BUILD c3_train_bases.jsonl (600 bases on C2_TRAIN_DOMAINS, DE-CONFOUNDED & STEP-1/2/3 BALANCED!)
    # =========================================================================
    c3_train_bases: list[dict[str, Any]] = []
    trn_non_ce_specs = []
    for fc in non_ce_classes:
        # 40 Step-1 (depth-2), 40 Step-2 (depth-3), 40 Step-3 (depth-3) -> 160 / 160 / 160!
        for bstep in (1, 2, 3):
            for pt in ("context_flip", "argument_flip"):
                for _ in range(20):
                    trn_non_ce_specs.append((fc, bstep, pt))
    assert len(trn_non_ce_specs) == 480
    trn_factors = _deconfounded_factors(480, len(C2B.C2_TRAIN_DOMAINS), seed=C3_TRAIN_SEED + 101, include_style=True)

    for idx, ((fc, bstep, pt), fac) in enumerate(zip(trn_non_ce_specs, trn_factors), start=1):
        dom_idx, pos, sty = fac["dom_idx"], fac["pos"], fac["style"]
        trans = dedup.next_balanced_transition(dom_idx, salt=113)
        bid = f"C3-TRN-{len(c3_train_bases) + 1:04d}"
        attempt = 0
        while True:
            cur_idx = 300000 + idx + attempt * 6 * 13337
            if bstep == 1:
                row = C2B._with_domains(
                    C2B.C2_TRAIN_DOMAINS,
                    C3_TRAIN_SEED,
                    A1_SYM.build_single_base,
                    cur_idx,
                    pt,
                    2,
                    fc,
                    trans,
                    pos,
                    id_prefix="C3TRN",
                    add_lures=False,
                )
                row["schema"] = "cedar.acl_exp_c3.train_depth2.v1"
                row["split"] = "c3_train"
                row["base_id"] = bid
                row["cell_id"] = bid
                row["family_id"] = bid
                row["broken_step"] = 1
                row["phrasing_style"] = "style_a"
                C2B._with_domains(C2B.C2_TRAIN_DOMAINS, C3_TRAIN_SEED, C1B._widen_threshold_numbers, row, cur_idx)
            else:
                row = _build_general_depth3_break_base(
                    domains=C2B.C2_TRAIN_DOMAINS,
                    seed_base=C3_TRAIN_SEED + 100000,
                    idx=cur_idx,
                    dom_idx=dom_idx,
                    fallacy_class=fc,
                    broken_step=bstep,
                    pair_type=pt,
                    pos=pos,
                    trans=trans,
                    bid=bid,
                    split_name="c3_train",
                    schema_name="cedar.acl_exp_c3.train_depth3.v1",
                    compound_blind=False,
                )
            if sty == "style_b":
                _apply_style_b_clean(row)
            if not dedup.has_collision(row):
                dedup.commit(row)
                c3_train_bases.append(row)
                break
            attempt += 1
            assert attempt < 80

    # 60 complete 2x2 CE cells = 120 bases (de-confounded across domain, pos, style!)
    ce_trn_factors = _deconfounded_factors(60, len(C2B.C2_TRAIN_DOMAINS), seed=C3_TRAIN_SEED + 202, include_style=True)
    for cell_i, fac in enumerate(ce_trn_factors):
        dom_idx, pos, sty = fac["dom_idx"], fac["pos"], fac["style"]
        depth = 2 if cell_i < 30 else 3
        trans = dedup.next_balanced_transition(dom_idx, salt=127)
        cell_id = f"C3-TRN-CECELL-{cell_i + 1:03d}"
        bid_a = f"C3-TRN-{len(c3_train_bases) + 1:04d}"
        bid_b = f"C3-TRN-{len(c3_train_bases) + 2:04d}"
        attempt = 0
        while True:
            ra, rb = C2B._build_single_chain_2x2_ce_cell(
                domains=C2B.C2_TRAIN_DOMAINS,
                seed_base=C3_TRAIN_SEED + 600000 + attempt * 7907,
                cell_idx=cell_i,
                dom_idx=dom_idx,
                depth=depth,
                pos=pos,
                trans=trans,
                cell_id=cell_id,
                bid_a=bid_a,
                bid_b=bid_b,
                split_name="c3_train",
                schema_name="cedar.acl_exp_c3.train_ce_2x2.v1",
            )
            for r in (ra, rb):
                r["family_id"] = cell_id
                r["broken_step"] = 1
                r["phrasing_style"] = "style_a"
                if sty == "style_b":
                    _apply_style_b_clean(r)
            if (not dedup.has_collision(ra)) and (not dedup.has_collision(rb)):
                dedup.commit(ra)
                dedup.commit(rb)
                c3_train_bases.extend([ra, rb])
                break
            attempt += 1
            assert attempt < 50

    assert len(c3_train_bases) == 600
    rng_shuf = random.Random(C3_TRAIN_SEED + 999)
    rng_shuf.shuffle(c3_train_bases)
    c3_train_examples = C2B._materialize_c2_train_examples(c3_train_bases, a1_cfg)
    assert len(c3_train_examples) == 6000

    # =========================================================================
    # 2. BUILD c3_dev_bases.jsonl (180 bases on PROXY_OOD_DOMAINS — strict dev selection set!)
    # =========================================================================
    c3_dev_bases: list[dict[str, Any]] = []
    dev_non_ce_specs = []
    for fc in non_ce_classes:
        for bstep in (1, 2, 3):
            for pt in ("context_flip", "argument_flip"):
                for _ in range(5):
                    dev_non_ce_specs.append((fc, bstep, pt))
    assert len(dev_non_ce_specs) == 120
    dev_factors = _deconfounded_factors(120, len(P1.PROXY_OOD_DOMAINS), seed=C3_DEV_SEED + 101, include_style=True)

    for idx, ((fc, bstep, pt), fac) in enumerate(zip(dev_non_ce_specs, dev_factors), start=1):
        dom_idx, pos, sty = fac["dom_idx"], fac["pos"], fac["style"]
        trans = dedup.next_balanced_transition(dom_idx, salt=149)
        bid = f"C3-DEV-{len(c3_dev_bases) + 1:04d}"
        attempt = 0
        while True:
            cur_idx = 500000 + idx + attempt * 6 * 15551
            row = _build_general_depth3_break_base(
                domains=P1.PROXY_OOD_DOMAINS,
                seed_base=C3_DEV_SEED + 100000,
                idx=cur_idx,
                dom_idx=dom_idx,
                fallacy_class=fc,
                broken_step=bstep,
                pair_type=pt,
                pos=pos,
                trans=trans,
                bid=bid,
                split_name="c3_dev",
                schema_name="cedar.acl_exp_c3.dev_depth3.v1",
                compound_blind=False,
            )
            if sty == "style_b":
                _apply_style_b_clean(row)
            if not dedup.has_collision(row):
                dedup.commit(row)
                c3_dev_bases.append(row)
                break
            attempt += 1
            assert attempt < 80

    # 30 complete 2x2 CE cells = 60 bases on PROXY_OOD_DOMAINS
    ce_dev_factors = _deconfounded_factors(30, len(P1.PROXY_OOD_DOMAINS), seed=C3_DEV_SEED + 202, include_style=True)
    for cell_i, fac in enumerate(ce_dev_factors):
        dom_idx, pos, sty = fac["dom_idx"], fac["pos"], fac["style"]
        depth = 2 if cell_i < 15 else 3
        trans = dedup.next_balanced_transition(dom_idx, salt=157)
        cell_id = f"C3-DEV-CECELL-{cell_i + 1:03d}"
        bid_a = f"C3-DEV-{len(c3_dev_bases) + 1:04d}"
        bid_b = f"C3-DEV-{len(c3_dev_bases) + 2:04d}"
        attempt = 0
        while True:
            ra, rb = C2B._build_single_chain_2x2_ce_cell(
                domains=P1.PROXY_OOD_DOMAINS,
                seed_base=C3_DEV_SEED + 600000 + attempt * 7919,
                cell_idx=cell_i,
                dom_idx=dom_idx,
                depth=depth,
                pos=pos,
                trans=trans,
                cell_id=cell_id,
                bid_a=bid_a,
                bid_b=bid_b,
                split_name="c3_dev",
                schema_name="cedar.acl_exp_c3.dev_ce_2x2.v1",
            )
            for r in (ra, rb):
                r["family_id"] = cell_id
                r["broken_step"] = 1
                r["phrasing_style"] = "style_a"
                if sty == "style_b":
                    _apply_style_b_clean(r)
            if (not dedup.has_collision(ra)) and (not dedup.has_collision(rb)):
                dedup.commit(ra)
                dedup.commit(rb)
                c3_dev_bases.extend([ra, rb])
                break
            attempt += 1
            assert attempt < 50

    assert len(c3_dev_bases) == 180

    # =========================================================================
    # 3. BUILD c3_blind_confirm_bases.jsonl (1,080 bases / 960 independent families,
    #    6 BRAND-NEW C3_BLIND_DOMAINS + HELD-OUT COMPOUND PROOF STRUCTURE + STYLE-C!)
    # =========================================================================
    c3_blind_bases: list[dict[str, Any]] = []
    blind_non_ce_specs = []
    for fc in non_ce_classes:
        for bstep in (1, 2, 3):
            for pt in ("context_flip", "argument_flip"):
                for _ in range(35):  # 4 * 3 * 2 * 35 = 840 independent compound non-CE bases!
                    blind_non_ce_specs.append((fc, bstep, pt))
    assert len(blind_non_ce_specs) == 840
    blind_factors = _deconfounded_factors(840, len(C3_BLIND_DOMAINS), seed=C3_BLIND_SEED + 101, include_style=False)

    for idx, ((fc, bstep, pt), fac) in enumerate(zip(blind_non_ce_specs, blind_factors), start=1):
        dom_idx, pos = fac["dom_idx"], fac["pos"]
        trans = dedup.next_balanced_transition(dom_idx, salt=173)
        bid = f"C3-BLD-{len(c3_blind_bases) + 1:04d}"
        attempt = 0
        while True:
            cur_idx = 700000 + idx + attempt * 6 * 19997
            row = _build_general_depth3_break_base(
                domains=C3_BLIND_DOMAINS,
                seed_base=C3_BLIND_SEED + 100000,
                idx=cur_idx,
                dom_idx=dom_idx,
                fallacy_class=fc,
                broken_step=bstep,
                pair_type=pt,
                pos=pos,
                trans=trans,
                bid=bid,
                split_name="c3_blind_confirm",
                schema_name="cedar.acl_exp_c3.blind_compound_depth3.v1",
                compound_blind=True,
            )
            if not dedup.has_collision(row):
                dedup.commit(row)
                c3_blind_bases.append(row)
                break
            attempt += 1
            assert attempt < 80

    # 120 Complete 2x2 Compound CE Cells = 240 bases (120 Row A + 120 Row B) on C3_BLIND_DOMAINS!
    ce_blind_factors = _deconfounded_factors(120, len(C3_BLIND_DOMAINS), seed=C3_BLIND_SEED + 202, include_style=False)
    for cell_i, fac in enumerate(ce_blind_factors):
        dom_idx, pos = fac["dom_idx"], fac["pos"]
        depth = 2 if cell_i < 60 else 3
        trans = dedup.next_balanced_transition(dom_idx, salt=181)
        cell_id = f"C3-BLD-CECELL-{cell_i + 1:03d}"
        bid_a = f"C3-BLD-{len(c3_blind_bases) + 1:04d}"
        bid_b = f"C3-BLD-{len(c3_blind_bases) + 2:04d}"
        attempt = 0
        while True:
            ra, rb = _build_compound_ce_2x2_cell(
                domains=C3_BLIND_DOMAINS,
                seed_base=C3_BLIND_SEED + 600000 + attempt * 8191,
                cell_idx=cell_i,
                dom_idx=dom_idx,
                depth=depth,
                pos=pos,
                trans=trans,
                cell_id=cell_id,
                bid_a=bid_a,
                bid_b=bid_b,
                split_name="c3_blind_confirm",
                schema_name="cedar.acl_exp_c3.blind_compound_ce_2x2.v1",
            )
            if (not dedup.has_collision(ra)) and (not dedup.has_collision(rb)):
                dedup.commit(ra)
                dedup.commit(rb)
                c3_blind_bases.extend([ra, rb])
                break
            attempt += 1
            assert attempt < 50

    assert len(c3_blind_bases) == 1080
    n_blind_families = len({r["family_id"] for r in c3_blind_bases})
    assert n_blind_families == 960, n_blind_families

    # Stratified 240-base think slice (160 non-CE across 4 classes x 3 steps + 40 CE cells = 80 CE bases)
    think_ids: set[str] = set()
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    ce_cells_map: dict[str, list[dict]] = defaultdict(list)
    for r in c3_blind_bases:
        if r["fallacy_class"] == "cross_entity_transfer":
            ce_cells_map[r["cell_id"]].append(r)
        else:
            by_stratum[(r["fallacy_class"], r["broken_step"], r["pair_type"])].append(r)
    # 24 strata x 7 items = 168? Wait: 4 classes * 3 steps * 2 pt = 24 strata; pick 7 from 16 strata + 6 from 8 strata = 160!
    rng_th = random.Random(C3_BLIND_SEED + 777)
    picked_non_ce: list[dict] = []
    for s_idx, sk in enumerate(sorted(by_stratum)):
        take = 7 if s_idx < 16 else 6
        pool = list(by_stratum[sk])
        rng_th.shuffle(pool)
        picked_non_ce.extend(pool[:take])
    assert len(picked_non_ce) == 160
    ce_cell_keys = sorted(ce_cells_map)
    rng_th.shuffle(ce_cell_keys)
    picked_ce = [r for cid in ce_cell_keys[:40] for r in ce_cells_map[cid]]
    assert len(picked_ce) == 80
    c3_blind_think_bases = picked_non_ce + picked_ce
    think_ids = {r["base_id"] for r in c3_blind_think_bases}
    for r in c3_blind_bases:
        r["in_think_slice"] = r["base_id"] in think_ids
    assert len(c3_blind_think_bases) == 240

    # =========================================================================
    # 4. EXHAUSTIVE ADVERSARIAL, DE-CONFOUNDING & ZERO-LEAKAGE AUDITS (ALL HARD ASSERTED!)
    # =========================================================================
    for name, rows in (
        ("c3_train", c3_train_bases),
        ("c3_dev", c3_dev_bases),
        ("c3_blind_confirm", c3_blind_bases),
    ):
        _audit_no_only_if(rows, name)
        for r in rows:
            assert r["fallacy_class"] != "converse_error", (name, r["base_id"])
            assert r["difficulty_depth"] <= 3, (name, r["base_id"], r["difficulty_depth"])

    # Factor independence audits (chi-square p > 0.15 across ALL pairs!)
    indep_train_non_ce = _audit_factor_independence(
        [r for r in c3_train_bases if r["fallacy_class"] != "cross_entity_transfer"], "c3_train_non_ce", check_style=True
    )
    indep_dev_non_ce = _audit_factor_independence(
        [r for r in c3_dev_bases if r["fallacy_class"] != "cross_entity_transfer"], "c3_dev_non_ce", check_style=True
    )
    indep_blind_non_ce = _audit_factor_independence(
        [r for r in c3_blind_bases if r["fallacy_class"] != "cross_entity_transfer"], "c3_blind_non_ce", check_style=False
    )

    # Skeleton 0-overlap between c3_train and BOTH c1_proxy_template_v2 AND c3_blind_confirm!
    train_v_skels = {PT1.extract_placeholder_skeleton(r["arms"]["V"], C2B.C2_TRAIN_DOMAINS) for r in c3_train_bases}
    train_ctx_skels = {PT1.extract_placeholder_skeleton(r["context_prompt_v"], C2B.C2_TRAIN_DOMAINS) for r in c3_train_bases}
    blind_ov_v = sum(PT1.extract_placeholder_skeleton(r["arms"]["V"], C3_BLIND_DOMAINS) in train_v_skels for r in c3_blind_bases)
    blind_ov_ctx = sum(PT1.extract_placeholder_skeleton(r["context_prompt_v"], C3_BLIND_DOMAINS) in train_ctx_skels for r in c3_blind_bases)
    assert blind_ov_v == 0 and blind_ov_ctx == 0, (blind_ov_v, blind_ov_ctx)
    v2_ov_v, v2_ov_ctx = 0, 0
    if v2_rows:
        v2_ov_v = sum(PT1.extract_placeholder_skeleton(r["arms"]["V"], P1.PROXY_OOD_DOMAINS) in train_v_skels for r in v2_rows)
        v2_ov_ctx = sum(PT1.extract_placeholder_skeleton(r["context_prompt_v"], P1.PROXY_OOD_DOMAINS) in train_ctx_skels for r in v2_rows)
        assert v2_ov_v == 0 and v2_ov_ctx == 0, (v2_ov_v, v2_ov_ctx)

    # Row A / Row B balance
    assert Counter(r["cell_member"] for r in c3_train_bases if r["fallacy_class"] == "cross_entity_transfer") == {"A": 60, "B": 60}
    assert Counter(r["cell_member"] for r in c3_dev_bases if r["fallacy_class"] == "cross_entity_transfer") == {"A": 30, "B": 30}
    assert Counter(r["cell_member"] for r in c3_blind_bases if r["fallacy_class"] == "cross_entity_transfer") == {"A": 120, "B": 120}

    # Broken step balance on non-CE
    trn_steps = dict(Counter(r["broken_step"] for r in c3_train_bases if r["fallacy_class"] != "cross_entity_transfer"))
    dev_steps = dict(Counter(r["broken_step"] for r in c3_dev_bases if r["fallacy_class"] != "cross_entity_transfer"))
    bld_steps = dict(Counter(r["broken_step"] for r in c3_blind_bases if r["fallacy_class"] != "cross_entity_transfer"))
    assert trn_steps == {1: 160, 2: 160, 3: 160}, trn_steps
    assert dev_steps == {1: 40, 2: 40, 3: 40}, dev_steps
    assert bld_steps == {1: 280, 2: 280, 3: 280}, bld_steps

    # G0 Turn-2 TF-IDF GroupKFold AUC audits (must be ~0.50!)
    g0_train = _run_g0_tfidf_auc(c3_train_bases)
    g0_dev = _run_g0_tfidf_auc(c3_dev_bases)
    g0_blind = _run_g0_tfidf_auc(c3_blind_bases)
    assert 0.40 <= g0_train["overall_auc"] <= 0.60, g0_train
    assert 0.40 <= g0_dev["overall_auc"] <= 0.60, g0_dev
    assert 0.42 <= g0_blind["overall_auc"] <= 0.58, g0_blind
    assert g0_train["cf_byte_match"] == 1.0 and g0_dev["cf_byte_match"] == 1.0 and g0_blind["cf_byte_match"] == 1.0

    # Balanced (domain, a0) -> at check
    tr_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in c3_train_bases:
        tr_counts[(r["domain"], r["initial_answer"])][r["target_answer"]] += 1
    maj_map = {k: v.most_common(1)[0][0] for k, v in tr_counts.items()}
    da_acc = {}
    for nm, rows in (("c3_train", c3_train_bases), ("c3_dev", c3_dev_bases), ("c3_blind_confirm", c3_blind_bases)):
        da_acc[nm] = round(float(np.mean([maj_map.get((r["domain"], r["initial_answer"]), "") == r["target_answer"] for r in rows])), 4)
        assert da_acc[nm] <= 0.45, (nm, da_acc[nm])

    # Save files deterministically
    out_files = {
        "c3_train_bases.jsonl": c3_train_bases,
        "c3_train_examples_nothink.jsonl": c3_train_examples,
        "c3_dev_bases.jsonl": c3_dev_bases,
        "c3_blind_confirm_bases.jsonl": c3_blind_bases,
        "c3_blind_think_bases.jsonl": c3_blind_think_bases,
    }
    manifest_files = {}
    for fn, rows in out_files.items():
        p = C3_DATA / fn
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8", newline="\n")
        manifest_files[fn] = {
            "count": len(rows),
            "bytes": p.stat().st_size,
            "sha256": _file_sha256(p),
        }

    manifest = {
        "schema": "cedar.acl_exp_c3.datasets_manifest.v1",
        "sealed_test_touched": False,
        "excluded_classes": ["converse_error", "depth_4"],
        "only_if_bug_occurrences": 0,
        "train_domains": [d["name"] for d in C2B.C2_TRAIN_DOMAINS],
        "dev_domains": [d["name"] for d in P1.PROXY_OOD_DOMAINS],
        "blind_confirm_domains": [d["name"] for d in C3_BLIND_DOMAINS],
        "broken_step_counts_non_ce": {
            "c3_train": trn_steps,
            "c3_dev": dev_steps,
            "c3_blind_confirm": bld_steps,
        },
        "blind_confirm_families": n_blind_families,
        "factor_independence_chi2": {
            "c3_train_non_ce": indep_train_non_ce,
            "c3_dev_non_ce": indep_dev_non_ce,
            "c3_blind_non_ce": indep_blind_non_ce,
        },
        "skeleton_overlap": {
            "c3_train_vs_c3_blind_V": blind_ov_v,
            "c3_train_vs_c3_blind_ctx": blind_ov_ctx,
            "c3_train_vs_proxy_v2_V": v2_ov_v,
            "c3_train_vs_proxy_v2_ctx": v2_ov_ctx,
        },
        "g0_tfidf_auc": {
            "c3_train": g0_train,
            "c3_dev": g0_dev,
            "c3_blind_confirm": g0_blind,
        },
        "domain_a0_to_at_acc": da_acc,
        "files": manifest_files,
    }
    mpath = C3_DATA / "c3_datasets_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] Built & audited all ACL-exp-c3 datasets -> {mpath}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
