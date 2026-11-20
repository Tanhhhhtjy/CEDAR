#!/usr/bin/env python3
"""Build zero-leakage, solver-certified datasets for ACL-exp-c1 (CPU only).

Design guarantees:
1. 100% DISJOINT from ACL-exp-a1 dev_only_pilot:
   - Brand-new 6 domains (marine_berth, rail_signal, energy_substation, clinic_triage, water_plant, vault_custody)
   - Brand-new entity names, zones, guards, metrics, predicates, outcome labels, and RNG seed (20270104)
   - Explicit runtime assertions verify ZERO shared base_ids, ZERO shared contexts, ZERO shared entities with A1 v2.2.
2. True Structural OOD Holdout (Leave-One-Fallacy-Out + Depth-4):
   - Train   (`c1_train_bases.jsonl`, 500 bases): 5 seen fallacy classes (scope_swap, cross_entity_transfer,
     missing_conjunct, negation_drop, threshold_violation), depth 2 & 3, mix of standard & 3-lure hard.
     STRICTLY EXCLUDES `converse_error` and depth 4!
   - Test ID (`c1_test_id_bases.jsonl`, 100 bases): same 5 seen classes, disjoint seeds (50 std + 50 hard).
   - Test OOD (`c1_test_ood_bases.jsonl`, 120 bases):
       * 60 UNSEEN `converse_error` bases (30 std depth 2/3 + 30 hard depth 3)
       * 60 UNSEEN depth-4 chains (4-hop entailment, 3 structural classes, solver-certified)
3. Also materialises paired training dialogue examples (`c1_train_examples_nothink.jsonl`) with balanced
   reject/accept/no-conflict ratios and paired `pair_group_id` for Paired Selective Margin Loss.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_build_datasets.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
A1_SCRIPTS = REPO / "experiments" / "ACL-exp-a1" / "scripts"
A1_DATA = REPO / "experiments" / "ACL-exp-a1" / "data"
sys.path.insert(0, str(A1_SCRIPTS))

import build_symbolic_pilot_v2 as A1_SYM  # noqa: E402
from a1_arm_wrappers import ARM_ORDER, build_arms  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

C1_MASTER_SEED = 20270104
SEEN_CLASSES = (
    "scope_swap",
    "cross_entity_transfer",
    "missing_conjunct",
    "negation_drop",
    "threshold_violation",
)
UNSEEN_CLASS = "converse_error"
STRUCTURAL_CLASSES = ("converse_error", "scope_swap", "cross_entity_transfer")

# 6 completely disjoint domains (zero overlap with A1_SYM.DOMAINS)
C1_DOMAINS: tuple[dict[str, Any], ...] = (
    {
        "name": "marine_berth",
        "intro": "A closed harbor berth controller applies exact docking rules. Use only the listed records and rules.",
        "entity_prefix": "Vessel",
        "entities": ("Barge-101", "Barge-204", "Barge-309", "Barge-418"),
        "zones": ("Pier North", "Pier South"),
        "guard_name": "tide_block",
        "guard_active_text": "the tide block is active",
        "guard_inactive_text": "the tide block is inactive",
        "metric_name": "draft gauge",
        "props": (
            ("p_obs", "sent a sonar echo"),
            ("p_mid1", "joins pilot queue"),
            ("p_mid2", "unlocks berth override"),
            ("p_mid3", "clears harbor priority"),
            ("p_aux", "holds a tug permit"),
            ("p_noise1", "passed hull sonar"),
            ("p_noise2", "logged anchor sync"),
        ),
        "outcomes": ("Berth-A1", "Berth-B2", "Berth-C3", "Berth-D4"),
    },
    {
        "name": "rail_signal",
        "intro": "A closed rail interlocking board applies exact track rules. Use only the listed records and rules.",
        "entity_prefix": "Train",
        "entities": ("Loco-55", "Loco-62", "Loco-78", "Loco-84"),
        "zones": ("Branch Upper", "Branch Lower"),
        "guard_name": "switch_lock",
        "guard_active_text": "the switch lock is active",
        "guard_inactive_text": "the switch lock is inactive",
        "metric_name": "axle score",
        "props": (
            ("p_obs", "tripped a track coil"),
            ("p_mid1", "enters siding check"),
            ("p_mid2", "grants signal override"),
            ("p_mid3", "opens express clearance"),
            ("p_aux", "carries a line token"),
            ("p_noise1", "passed brake gauge"),
            ("p_noise2", "logged cab telemetry"),
        ),
        "outcomes": ("Track-Red", "Track-Gold", "Track-Cyan", "Track-Gray"),
    },
    {
        "name": "energy_substation",
        "intro": "A closed substation load balancer enforces exact breaker rules. Use only the listed entries and rules.",
        "entity_prefix": "Feeder",
        "entities": ("Bus-17", "Bus-29", "Bus-43", "Bus-58"),
        "zones": ("Ring Alpha", "Ring Beta"),
        "guard_name": "relay_latch",
        "guard_active_text": "the relay latch is active",
        "guard_inactive_text": "the relay latch is inactive",
        "metric_name": "surge index",
        "props": (
            ("p_obs", "logged a phase ripple"),
            ("p_mid1", "triggers tie review"),
            ("p_mid2", "authorizes load override"),
            ("p_mid3", "enables reserve transfer"),
            ("p_aux", "holds a sync relay"),
            ("p_noise1", "cleared oil test"),
            ("p_noise2", "kept nominal frequency"),
        ),
        "outcomes": ("Bank-One", "Bank-Two", "Bank-Three", "Bank-Four"),
    },
    {
        "name": "clinic_triage",
        "intro": "A closed clinical intake desk follows exact triage rules. Use only the listed charts and rules.",
        "entity_prefix": "Chart",
        "entities": ("Case-31", "Case-46", "Case-52", "Case-67"),
        "zones": ("Ward Cedar", "Ward Maple"),
        "guard_name": "isolation_hold",
        "guard_active_text": "the isolation hold is active",
        "guard_inactive_text": "the isolation hold is inactive",
        "metric_name": "acuity score",
        "props": (
            ("p_obs", "flagged a lab delta"),
            ("p_mid1", "enters consult track"),
            ("p_mid2", "triggers fast override"),
            ("p_mid3", "grants acute placement"),
            ("p_aux", "has a specialist note"),
            ("p_noise1", "completed intake form"),
            ("p_noise2", "verified chart barcode"),
        ),
        "outcomes": ("Desk-I", "Desk-II", "Desk-III", "Desk-IV"),
    },
    {
        "name": "water_plant",
        "intro": "A closed filtration plant controller applies exact valve rules. Use only the listed entries and rules.",
        "entity_prefix": "Valve",
        "entities": ("Basin-15", "Basin-23", "Basin-39", "Basin-48"),
        "zones": ("Stage Fore", "Stage Aft"),
        "guard_name": "bypass_clamp",
        "guard_active_text": "the bypass clamp is active",
        "guard_inactive_text": "the bypass clamp is inactive",
        "metric_name": "turbidity reading",
        "props": (
            ("p_obs", "showed a flow surge"),
            ("p_mid1", "enters flush review"),
            ("p_mid2", "activates flow override"),
            ("p_mid3", "engages secondary loop"),
            ("p_aux", "carries a chlorine tag"),
            ("p_noise1", "passed pressure sweep"),
            ("p_noise2", "cleared filter gauge"),
        ),
        "outcomes": ("Loop-East", "Loop-West", "Loop-Core", "Loop-Rim"),
    },
    {
        "name": "vault_custody",
        "intro": "A closed treasury custody ledger applies exact release rules. Use only the listed records and rules.",
        "entity_prefix": "Box",
        "entities": ("Safe-71", "Safe-83", "Safe-92", "Safe-96"),
        "zones": ("Wing Ivory", "Wing Slate"),
        "guard_name": "timelock_seal",
        "guard_active_text": "the timelock seal is active",
        "guard_inactive_text": "the timelock seal is inactive",
        "metric_name": "clearance rating",
        "props": (
            ("p_obs", "filed a custody alert"),
            ("p_mid1", "enters dual verification"),
            ("p_mid2", "authorizes vault override"),
            ("p_mid3", "approves courier release"),
            ("p_aux", "holds a notary cipher"),
            ("p_noise1", "cleared seal camera"),
            ("p_noise2", "passed weight ledger"),
        ),
        "outcomes": ("Class-P", "Class-Q", "Class-R", "Class-S"),
    },
)


def _with_c1_domains(fn, *args, **kwargs):
    """Temporarily swap A1_SYM.DOMAINS and MASTER_SEED to C1 disjoint pools."""
    old_dom, old_seed = A1_SYM.DOMAINS, A1_SYM.MASTER_SEED
    A1_SYM.DOMAINS = C1_DOMAINS
    A1_SYM.MASTER_SEED = C1_MASTER_SEED
    try:
        return fn(*args, **kwargs)
    finally:
        A1_SYM.DOMAINS = old_dom
        A1_SYM.MASTER_SEED = old_seed


def build_depth4_base(
    idx: int,
    pair_type: str,
    fallacy_class: str,
    transition: tuple[str, str],
    default_policy_position: str,
    base_id: str,
) -> dict[str, Any]:
    """Build a solver-certified depth-4 (4-hop) structural fallacy base for OOD evaluation.

    Chain: obs -> mid1 (Rule 1) -> mid2 (Rule 2) -> mid3 (Rule 3) -> override (Rule 4) -> outcome (Rule 5).
    Broken step is balanced between step 1 and step 2 (plus step 3 via idx parity).
    """
    assert fallacy_class in STRUCTURAL_CLASSES
    rng = random.Random(C1_MASTER_SEED + 50000 + idx * 7919)
    domain = C1_DOMAINS[idx % len(C1_DOMAINS)]
    p = dict(domain["props"])

    ents = [f"{domain['entity_prefix']} {e}" for e in domain["entities"]]
    rng.shuffle(ents)
    e1, e2, e_noise, _ = ents

    zone_a, zone_b = domain["zones"]
    if rng.random() < 0.5:
        zone_a, zone_b = zone_b, zone_a

    outcomes = list(domain["outcomes"])
    rng.shuffle(outcomes)
    a0, at = transition
    letter_to_outcome = dict(zip(("A", "B", "C", "D"), outcomes))
    out_a0, out_at = letter_to_outcome[a0], letter_to_outcome[at]

    atom_base = f"base_active({e1})"
    atom_zone_a = f"in_zone({e1},{zone_a})"
    atom_zone_b = f"in_zone({e1},{zone_b})"
    atom_obs_e1 = f"obs({e1})"
    atom_obs_e2 = f"obs({e2})"
    atom_mid1 = f"mid1({e1})"
    atom_mid1_e2 = f"mid1({e2})"
    atom_mid2 = f"mid2({e1})"
    atom_mid3 = f"mid3({e1})"
    atom_override = f"override({e1})"
    atom_noise1 = f"noise1({e_noise})"
    atom_noise2 = f"noise2({e2})"

    base_facts = {atom_base, atom_zone_a, atom_noise1, atom_noise2}
    base_fact_lines = [
        f"{e1} is registered under active default status.",
        f"{e1} belongs to {zone_a}.",
        f"For {e1}, {domain['guard_inactive_text']}.",
        f"{e_noise} {p['p_noise1']}.",
        f"{e2} {p['p_noise2']}.",
    ]
    default_rule = {
        "text": f"{e1} is under active default status, so {e1} is assigned to {out_a0} unless the rules establish that {e1} {p['p_mid3']}.",
    }
    distractor_rule = {
        "id": "Rule 6",
        "conditions": [atom_noise1, atom_noise2],
        "min_threshold": None,
        "conclusion": f"log_ok({e_noise})",
        "text": f"If {e_noise} {p['p_noise1']} and {e2} {p['p_noise2']}, record a routine audit note for {e_noise}.",
    }
    lures = [
        {"id": "Lure 1", "conditions": [atom_obs_e2], "min_threshold": None,
         "conclusion": f"watch({e2})", "text": f"If {e2} {p['p_obs']}, then {e2} is placed on a watch list."},
        {"id": "Lure 2", "conditions": [atom_mid1], "min_threshold": None,
         "conclusion": f"logrev({e1})", "text": f"If {e1} {p['p_mid1']}, then {e1} receives a periodic log review."},
        {"id": "Lure 3", "conditions": [atom_zone_b, atom_obs_e1], "min_threshold": None,
         "conclusion": f"archq({e1})", "text": f"For entries in {zone_b}, if {e1} {p['p_obs']}, then {e1} is queued for archival."},
    ]

    # Step 2 broken vs Step 1 broken; decouple flip_aux from broken_step_idx parity (idx steps by 3 per rep)
    broken_step_idx = 1 if ((idx // 3) % 2 == 0) else 2
    flip_aux = ((idx // 6) % 2 == 1)
    pred_a, pred_b = (p["p_aux"], p["p_obs"]) if flip_aux else (p["p_obs"], p["p_aux"])
    atom_a, atom_b = (f"aux({e1})", atom_obs_e1) if flip_aux else (atom_obs_e1, f"aux({e1})")
    r3 = {"id": "Rule 3", "conditions": [atom_mid2], "min_threshold": None, "conclusion": atom_mid3,
          "text": f"If {e1} {p['p_mid2']}, then {e1} {p['p_mid3']}."}
    r4 = {"id": "Rule 4", "conditions": [atom_mid3], "min_threshold": None, "conclusion": atom_override,
          "text": f"If {e1} {p['p_mid3']}, override the default classification and assign {e1} to {out_at}."}

    # Construct Rule 1 & Rule 2 and obs/steps for the 3 structural fallacy classes
    extra_rules: list[dict[str, Any]] = []
    if fallacy_class == "converse_error":
        if broken_step_idx == 1:
            r1_v = {"id": "Rule 1", "conditions": [atom_a], "min_threshold": None, "conclusion": atom_mid1,
                    "text": f"If {e1} {pred_a}, then {e1} {p['p_mid1']}."}
            r1_f = {"id": "Rule 1", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_a,
                    "text": f"If {e1} {p['p_mid1']}, then {e1} {pred_a}."}
            r2_v = r2_f = {"id": "Rule 2", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                           "text": f"If {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
            if pair_type == "context_flip":
                obs_v = obs_f = f"New verified observation: {e1} {pred_a}."
                obs_facts_v = obs_facts_f = {atom_a}
                steps_v = steps_f = [
                    f"Step 1: Since {e1} {pred_a}, Rule 1 implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
            else:
                r1_v = r1_f = {"id": "Rule 1", "conditions": [atom_a], "min_threshold": None, "conclusion": atom_mid1,
                               "text": f"If {e1} {pred_a}, then {e1} {p['p_mid1']}."}
                r_aux = {"id": "Rule 5", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_b,
                         "text": f"If {e1} {p['p_mid1']}, then {e1} {pred_b}."}
                extra_rules.append(r_aux)
                obs_v = f"New verified observation: {e1} {pred_a}."
                obs_f = f"New verified observation: {e1} {pred_b}."
                obs_facts_v, obs_facts_f = {atom_a}, {atom_b}
                steps_v = [
                    f"Step 1: Since {e1} {pred_a}, the matching rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
                steps_f = [
                    f"Step 1: Since {e1} {pred_b}, the matching rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
        else:
            r1_v = r1_f = {"id": "Rule 1", "conditions": [atom_a], "min_threshold": None, "conclusion": atom_mid1,
                           "text": f"If {e1} {pred_a}, then {e1} {p['p_mid1']}."}
            if pair_type == "context_flip":
                r2_v = {"id": "Rule 2", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                        "text": f"If {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                r2_f = {"id": "Rule 2", "conditions": [atom_mid2], "min_threshold": None, "conclusion": atom_mid1,
                        "text": f"If {e1} {p['p_mid2']}, then {e1} {p['p_mid1']}."}
                obs_v = obs_f = f"New verified observation: {e1} {pred_a}."
                obs_facts_v = obs_facts_f = {atom_a}
                steps_v = steps_f = [
                    f"Step 1: Since {e1} {pred_a}, Rule 1 implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
            else:
                r2_v = r2_f = {"id": "Rule 2", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                               "text": f"If {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                r_aux = {"id": "Rule 5", "conditions": [atom_mid2], "min_threshold": None, "conclusion": atom_b,
                         "text": f"If {e1} {p['p_mid2']}, then {e1} {pred_b}."}
                r_obs_aux = {"id": "Rule 5b", "conditions": [atom_obs_e2], "min_threshold": None, "conclusion": atom_b,
                             "text": f"If {e2} {p['p_obs']}, then {e1} {pred_b}."}
                extra_rules.extend([r_aux, r_obs_aux])
                obs_v = f"New verified observation: {e1} {pred_a}."
                obs_f = f"New verified observation: {e2} {p['p_obs']}."
                obs_facts_v, obs_facts_f = {atom_a}, {atom_obs_e2}
                steps_v = [
                    f"Step 1: Since {e1} {pred_a}, the matching rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Using the matching rule, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
                steps_f = [
                    f"Step 1: Since {e2} {p['p_obs']}, the matching rule implies that {e1} {pred_b}.",
                    f"Step 2: Using the matching rule, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
    elif fallacy_class == "scope_swap":
        if broken_step_idx == 1:
            r2_v = r2_f = {"id": "Rule 2", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                           "text": f"If {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
            if pair_type == "context_flip":
                r1_v = {"id": "Rule 1", "conditions": [atom_zone_a, atom_a], "min_threshold": None, "conclusion": atom_mid1,
                        "text": f"For entries in {zone_a}, if {e1} {pred_a}, then {e1} {p['p_mid1']}."}
                r1_f = {"id": "Rule 1", "conditions": [atom_zone_b, atom_a], "min_threshold": None, "conclusion": atom_mid1,
                        "text": f"For entries in {zone_b}, if {e1} {pred_a}, then {e1} {p['p_mid1']}."}
                obs_v = obs_f = f"New verified observation: {e1} {pred_a}."
                obs_facts_v = obs_facts_f = {atom_a}
                steps_v = steps_f = [
                    f"Step 1: Since {e1} {pred_a}, Rule 1 implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
            else:
                r1_v = r1_f = {"id": "Rule 1", "conditions": [atom_zone_a, atom_a], "min_threshold": None, "conclusion": atom_mid1,
                               "text": f"For entries in {zone_a}, if {e1} {pred_a}, then {e1} {p['p_mid1']}."}
                r_zb = {"id": "Rule 5", "conditions": [atom_zone_b, atom_b], "min_threshold": None, "conclusion": atom_mid1,
                        "text": f"For entries in {zone_b}, if {e1} {pred_b}, then {e1} {p['p_mid1']}."}
                extra_rules.append(r_zb)
                obs_v = f"New verified observation: {e1} {pred_a}."
                obs_f = f"New verified observation: {e1} {pred_b}."
                obs_facts_v, obs_facts_f = {atom_a}, {atom_b}
                steps_v = [
                    f"Step 1: Since {e1} {pred_a}, the matching zone rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
                steps_f = [
                    f"Step 1: Since {e1} {pred_b}, the matching zone rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
        else:
            r1_v = r1_f = {"id": "Rule 1", "conditions": [atom_a], "min_threshold": None, "conclusion": atom_mid1,
                           "text": f"If {e1} {pred_a}, then {e1} {p['p_mid1']}."}
            if pair_type == "context_flip":
                r2_v = {"id": "Rule 2", "conditions": [atom_zone_a, atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                        "text": f"For entries in {zone_a}, if {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                r2_f = {"id": "Rule 2", "conditions": [atom_zone_b, atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                        "text": f"For entries in {zone_b}, if {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                obs_v = obs_f = f"New verified observation: {e1} {pred_a}."
                obs_facts_v = obs_facts_f = {atom_a}
                steps_v = steps_f = [
                    f"Step 1: Since {e1} {pred_a}, Rule 1 implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
            else:
                r2_v = r2_f = {"id": "Rule 2", "conditions": [atom_zone_a, atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                               "text": f"For entries in {zone_a}, if {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                r_zb = {"id": "Rule 5", "conditions": [atom_zone_b, atom_mid1_e2], "min_threshold": None, "conclusion": atom_mid2,
                        "text": f"For entries in {zone_b}, if {e1} {pred_b}, then {e1} {p['p_mid2']}."}
                r_obs_aux = {"id": "Rule 5b", "conditions": [atom_b], "min_threshold": None, "conclusion": atom_mid1_e2,
                             "text": f"If {e1} {pred_b}, then {e1} {p['p_mid1']}."}
                # Wait: keep atom_mid1_e2 distinct from atom_mid1 so F does not trigger Rule 2!
                r_obs_aux = {"id": "Rule 5b", "conditions": [atom_b], "min_threshold": None, "conclusion": f"zmid({e1})",
                             "text": f"If {e1} {pred_b}, then {e1} {p['p_aux'] if not flip_aux else p['p_obs']}."}
                r_zb = {"id": "Rule 5", "conditions": [atom_zone_b, f"zmid({e1})"], "min_threshold": None, "conclusion": atom_mid2,
                        "text": f"For entries in {zone_b}, if {e1} {pred_b}, then {e1} {p['p_mid2']}."}
                extra_rules.extend([r_zb, r_obs_aux])
                obs_v = f"New verified observation: {e1} {pred_a}."
                obs_f = f"New verified observation: {e1} {pred_b}."
                obs_facts_v, obs_facts_f = {atom_a}, {atom_b}
                steps_v = [
                    f"Step 1: Since {e1} {pred_a}, the matching rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying the matching zone rule, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
                steps_f = [
                    f"Step 1: Since {e1} {pred_b}, the matching rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying the matching zone rule, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
    else:  # cross_entity_transfer
        if broken_step_idx == 1:
            r2_v = r2_f = {"id": "Rule 2", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                           "text": f"If {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
            if pair_type == "context_flip":
                r1_v = {"id": "Rule 1", "conditions": [atom_obs_e2], "min_threshold": None, "conclusion": atom_mid1,
                        "text": f"If {e2} {p['p_obs']}, then {e1} {p['p_mid1']}."}
                r1_f = {"id": "Rule 1", "conditions": [atom_obs_e1], "min_threshold": None, "conclusion": atom_mid1,
                        "text": f"If {e1} {p['p_obs']}, then {e1} {p['p_mid1']}."}
                obs_v = obs_f = f"New verified observation: {e2} {p['p_obs']}."
                obs_facts_v = obs_facts_f = {atom_obs_e2}
                steps_v = steps_f = [
                    f"Step 1: Since {e2} {p['p_obs']}, Rule 1 implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
            else:
                r1_v = r1_f = {"id": "Rule 1", "conditions": [atom_obs_e1], "min_threshold": None, "conclusion": atom_mid1,
                               "text": f"If {e1} {p['p_obs']}, then {e1} {p['p_mid1']}."}
                obs_v = f"New verified observation: {e1} {p['p_obs']}."
                obs_f = f"New verified observation: {e2} {p['p_obs']}."
                obs_facts_v, obs_facts_f = {atom_obs_e1}, {atom_obs_e2}
                steps_v = [
                    f"Step 1: Since {e1} {p['p_obs']}, Rule 1 implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
                steps_f = [
                    f"Step 1: Since {e2} {p['p_obs']}, Rule 1 implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
        else:
            r1_v = r1_f = {"id": "Rule 1", "conditions": [atom_obs_e2], "min_threshold": None, "conclusion": atom_mid1_e2,
                           "text": f"If {e2} {p['p_obs']}, then {e2} {p['p_mid1']}."}
            if pair_type == "context_flip":
                r2_v = {"id": "Rule 2", "conditions": [atom_mid1_e2], "min_threshold": None, "conclusion": atom_mid2,
                        "text": f"If {e2} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                r2_f = {"id": "Rule 2", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                        "text": f"If {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                obs_v = obs_f = f"New verified observation: {e2} {p['p_obs']}."
                obs_facts_v = obs_facts_f = {atom_obs_e2}
                steps_v = steps_f = [
                    f"Step 1: Since {e2} {p['p_obs']}, Rule 1 implies that {e2} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
            else:
                r1_aux = {"id": "Rule 5", "conditions": [atom_obs_e1], "min_threshold": None, "conclusion": atom_mid1,
                          "text": f"If {e1} {p['p_obs']}, then {e1} {p['p_mid1']}."}
                extra_rules.append(r1_aux)
                r2_v = r2_f = {"id": "Rule 2", "conditions": [atom_mid1], "min_threshold": None, "conclusion": atom_mid2,
                               "text": f"If {e1} {p['p_mid1']}, then {e1} {p['p_mid2']}."}
                obs_v = f"New verified observation: {e1} {p['p_obs']}."
                obs_f = f"New verified observation: {e2} {p['p_obs']}."
                obs_facts_v, obs_facts_f = {atom_obs_e1}, {atom_obs_e2}
                steps_v = [
                    f"Step 1: Since {e1} {p['p_obs']}, the matching rule implies that {e1} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]
                steps_f = [
                    f"Step 1: Since {e2} {p['p_obs']}, the matching rule implies that {e2} {p['p_mid1']}.",
                    f"Step 2: Applying Rule 2, {e1} {p['p_mid2']}.",
                    f"Step 3: Applying Rule 3, {e1} {p['p_mid3']}.",
                    f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {out_at} (Option {at}).",
                ]

    rules_v = [r1_v, r2_v, r3, r4, *extra_rules, distractor_rule, *lures]
    rules_f = [r1_f, r2_f, r3, r4, *extra_rules, distractor_rule, *lures]

    # Exact fixed-point solver certification
    _, init_v = A1_SYM.evaluate_world(base_facts, {}, rules_v, a0, atom_override, at)
    _, init_f = A1_SYM.evaluate_world(base_facts, {}, rules_f, a0, atom_override, at)
    _, post_v = A1_SYM.evaluate_world(base_facts | obs_facts_v, {}, rules_v, a0, atom_override, at)
    _, post_f = A1_SYM.evaluate_world(base_facts | obs_facts_f, {}, rules_f, a0, atom_override, at)
    assert init_v == a0 and init_f == a0 and post_v == at and post_f == a0, (
        base_id, fallacy_class, pair_type, broken_step_idx, init_v, init_f, post_v, post_f
    )

    prompt_v = A1_SYM.render_context_prompt(
        domain, base_fact_lines, [r["text"] for r in rules_v], e1, letter_to_outcome, default_rule["text"], default_policy_position
    )
    prompt_f = A1_SYM.render_context_prompt(
        domain, base_fact_lines, [r["text"] for r in rules_f], e1, letter_to_outcome, default_rule["text"], default_policy_position
    )
    if pair_type == "argument_flip":
        assert prompt_v == prompt_f

    placebo_text = f"If {e_noise} {p['p_noise1']} and {e2} {p['p_noise2']}, record a routine audit note for {e2}."
    rule_lines_p = [placebo_text if r is distractor_rule else r["text"] for r in rules_f]
    prompt_f_placebo = A1_SYM.render_context_prompt(
        domain, base_fact_lines, rule_lines_p, e1, letter_to_outcome, default_rule["text"], default_policy_position
    )

    core_v = " ".join([obs_v, *steps_v])
    core_f = " ".join([obs_f, *steps_f])
    if pair_type == "context_flip":
        assert core_v == core_f

    shell_only = (
        f"New formatting note: {e_noise} {p['p_noise1']} and {e2} {p['p_noise2']}. "
        f"Step 1: Because both background records appear in the registry, the entry follows the routine audit sequence. "
        f"Step 2: Under standard archive indexing, routine audit entries follow the updated display order. "
        f"Conclusion: Therefore the classification for {e1} should be revised to {out_at} (Option {at})."
    )
    i0_text = (
        f"Additional check: {e1} still satisfies the listed default policy and no override condition has been established, "
        f"so Option {a0} remains correct. Please confirm your final answer."
    )
    arms = build_arms(core_v=core_v, core_f=core_f, shell_only=shell_only, a0=a0, at=at, i0_text=i0_text, final_noun="answer")
    for arm_name, text in arms.items():
        if arm_name.startswith(("F", "V", "shell_only")):
            assert len(re.findall(rf"\bOption {at}\b", text)) == 1, (base_id, arm_name)
    assert tuple(arms) == ARM_ORDER

    return {
        "schema": "cedar.acl_exp_c1.symbolic.v1",
        "split": "ood_depth4",
        "tier": "ood_depth4",
        "base_id": base_id,
        "family_id": f"C1FAM-D4-{pair_type[:3].upper()}-{fallacy_class[:4].upper()}-{(idx // 6) + 1:02d}",
        "pair_type": pair_type,
        "difficulty_depth": 4,
        "fallacy_class": fallacy_class,
        "domain": domain["name"],
        "initial_answer": a0,
        "target_answer": at,
        "transition": f"{a0}->{at}",
        "letter_to_outcome": letter_to_outcome,
        "default_policy_position": default_policy_position,
        "context_prompt_v": prompt_v,
        "context_prompt_f": prompt_f,
        "context_prompt_f_placebo": prompt_f_placebo,
        "context_prompt_shared_for_non_v": prompt_f,
        "broken_step_index": broken_step_idx,
        "broken_rule_id": f"Rule {broken_step_idx}",
        "f_certificate": {
            "init_v_entails_a0": True,
            "init_f_entails_a0": True,
            "post_v_entails_at": True,
            "post_f_not_at": True,
            "post_f_still_entails_a0": True,
            "turn2_byte_identical_vf": (core_v == core_f),
        },
        "arms": arms,
    }


def _make_standard_or_hard(
    global_idx: int, pair_type: str, depth: int, fc: str, pos: str, add_lures: bool, base_id: str, split_name: str
) -> dict[str, Any]:
    trans = A1_SYM.ORDERED_TRANSITIONS[global_idx % len(A1_SYM.ORDERED_TRANSITIONS)]
    row = _with_c1_domains(
        A1_SYM.build_single_base,
        global_idx,
        pair_type,
        depth,
        fc,
        trans,
        pos,
        id_prefix="C1",
        add_lures=add_lures,
    )
    row["schema"] = "cedar.acl_exp_c1.symbolic.v1"
    row["split"] = split_name
    row["base_id"] = base_id
    row["family_id"] = f"C1{split_name.upper()[:3]}-{pair_type[:3].upper()}-D{depth}-{fc[:4].upper()}-{(global_idx // 10) + 1:03d}"
    return row


def verify_zero_a1_leakage(all_c1_rows: list[dict]) -> None:
    a1_ids, a1_ctxs = set(), set()
    for fn in (
        "symbolic_pilot_120_v2_2_dev_only.jsonl",
        "symbolic_hard_120_v2_2_dev_only.jsonl",
        "natural_pilot_seed_12_v2_2.jsonl",
    ):
        p = A1_DATA / fn
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            a1_ids.add(r["base_id"])
            a1_ctxs.add(r["context_prompt_f"])
            a1_ctxs.add(r["context_prompt_v"])
    a1_entities = {f"{d['entity_prefix']} {e}" for d in A1_SYM.DOMAINS for e in d["entities"]}
    c1_entities = {f"{d['entity_prefix']} {e}" for d in C1_DOMAINS for e in d["entities"]}
    assert not (a1_entities & c1_entities), f"Entity overlap with A1: {a1_entities & c1_entities}"
    for r in all_c1_rows:
        assert r["base_id"] not in a1_ids, r["base_id"]
        assert r["context_prompt_f"] not in a1_ctxs, r["base_id"]
        assert r["context_prompt_v"] not in a1_ctxs, r["base_id"]


def materialize_train_examples(train_bases: list[dict], a1_cfg: dict) -> list[dict]:
    """Materialize balanced paired training records for no_think SFT + Paired Selective Margin Loss.

    For each base_id (500 bases), creates a paired group of 5 core turns (~2,500 examples):
      - Accept (40%, 2/base):  V (every base) + V_hes (every base) -> gold = target_answer (a_t)
      - Reject (50%, 2.5/base): F (every base) + shell_only (every base) + 1 rotating reject arm
                                (S / AO / F_auth / F_src_third_party) -> gold = initial_answer (a0)
      - No-Conflict (10%, 0.5/base): alternating P0_confirm_a0 / I0_confirm_a0 -> gold = initial_answer (a0)
    Total = 500 * 5.5 -> exact 3,000 rule examples (6 per base: V, V_hes, F, shell_only, {S/AO/F_auth/F_src_3p}, {P0/I0/N}).
    Proportions: Accept 2/6 (33.3%), Reject 3/6 (50.0%), No-Conflict/Neutral 1/6 (16.7%).
    """
    rotating_reject = ("S", "F_auth", "F_src_third_party", "AO")
    rotating_keep = ("P0_confirm_a0", "I0_confirm_a0", "N")
    examples = []
    for i, row in enumerate(train_bases):
        a0, at = row["initial_answer"], row["target_answer"]
        arm_specs = [
            ("V", at, "accept", +1),
            ("V_hes", at, "accept", +1),
            ("F", a0, "reject_F", -1),
            ("shell_only", a0, "reject_shell", -1),
            (rotating_reject[i % len(rotating_reject)], a0, "reject_social_or_rhet", -1),
            (rotating_keep[i % len(rotating_keep)], a0, "no_conflict", 0),
        ]
        for arm, gold_ans, role_tag, margin_sign in arm_specs:
            msgs = build_messages(a1_cfg, row, arm, "no_think")
            examples.append(
                {
                    "example_id": f"{row['base_id']}|{arm}",
                    "pair_group_id": row["base_id"],
                    "base_id": row["base_id"],
                    "arm": arm,
                    "role_tag": role_tag,
                    "margin_sign": margin_sign,  # +1 => push u_theta >= +m; -1 => push u_theta <= -m; 0 => SFT only
                    "pair_type": row["pair_type"],
                    "fallacy_class": row["fallacy_class"],
                    "difficulty_depth": row["difficulty_depth"],
                    "initial_answer": a0,
                    "target_answer": at,
                    "gold_answer": gold_ans,
                    "messages": msgs,
                    "assistant_target": f"Final: {gold_ans}",
                }
            )
    return examples


def write_jsonl(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data_dir = C1 / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    a1_cfg = load_config()

    # 1. Train: 500 bases across 5 SEEN classes (no converse_error, no depth 4)
    # 2 pair_types x 2 depths (2,3) x 5 seen classes x 25 reps = 500 bases (350 std + 150 hard with 3 lures)
    train_bases: list[dict[str, Any]] = []
    gidx = 100
    k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(25):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3 and rep >= 10)  # 15/25 of depth-3 => 150 hard, 350 std
                for fc in SEEN_CLASSES:
                    k += 1
                    bid = f"C1-TRN-{k:04d}"
                    train_bases.append(_make_standard_or_hard(gidx, pair_type, depth, fc, pos, add_lures, bid, "train"))
                    gidx += 1
    assert len(train_bases) == 500
    assert all(r["fallacy_class"] != UNSEEN_CLASS and r["difficulty_depth"] in (2, 3) for r in train_bases)

    # 2. Held-out ID Test: 100 bases across the same 5 SEEN classes, disjoint RNG offsets (50 std + 50 hard)
    test_id_bases: list[dict[str, Any]] = []
    gidx = 10000
    k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(5):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3)  # 50 depth-2 std + 50 depth-3 hard
                for fc in SEEN_CLASSES:
                    k += 1
                    bid = f"C1-ID-{k:03d}"
                    test_id_bases.append(_make_standard_or_hard(gidx, pair_type, depth, fc, pos, add_lures, bid, "test_id"))
                    gidx += 1
    assert len(test_id_bases) == 100

    # 3. Held-out OOD Test: 120 bases = 60 UNSEEN converse_error (depth 2/3) + 60 UNSEEN depth-4 (3 structural classes)
    test_ood_bases: list[dict[str, Any]] = []
    gidx = 20000
    k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(15):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3)
                k += 1
                bid = f"C1-OOD-CONV-{k:03d}"
                test_ood_bases.append(
                    _make_standard_or_hard(gidx, pair_type, depth, UNSEEN_CLASS, pos, add_lures, bid, "ood_converse")
                )
                gidx += 1
    d4_k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for rep in range(10):
            before = ((rep % 2 == 0) ^ (pair_type == "argument_flip"))
            pos = "before_rules" if before else "after_rules"
            for fc in STRUCTURAL_CLASSES:
                d4_k += 1
                trans = A1_SYM.ORDERED_TRANSITIONS[(30000 + d4_k) % len(A1_SYM.ORDERED_TRANSITIONS)]
                bid = f"C1-OOD-D4-{d4_k:03d}"
                test_ood_bases.append(build_depth4_base(d4_k, pair_type, fc, trans, pos, bid))
    assert len(test_ood_bases) == 120

    # Strict zero-leakage checks (vs A1 and across C1 splits)
    all_c1 = train_bases + test_id_bases + test_ood_bases
    assert len({r["base_id"] for r in all_c1}) == len(all_c1)
    assert not ({r["context_prompt_f"] for r in train_bases} & {r["context_prompt_f"] for r in test_id_bases + test_ood_bases})
    verify_zero_a1_leakage(all_c1)

    train_examples = materialize_train_examples(train_bases, a1_cfg)
    assert len(train_examples) == 3000

    hashes = {
        "c1_train_bases.jsonl": write_jsonl(train_bases, data_dir / "c1_train_bases.jsonl"),
        "c1_train_examples_nothink.jsonl": write_jsonl(train_examples, data_dir / "c1_train_examples_nothink.jsonl"),
        "c1_test_id_bases.jsonl": write_jsonl(test_id_bases, data_dir / "c1_test_id_bases.jsonl"),
        "c1_test_ood_bases.jsonl": write_jsonl(test_ood_bases, data_dir / "c1_test_ood_bases.jsonl"),
    }
    manifest = {
        "schema": "cedar.acl_exp_c1.datasets_manifest.v1",
        "zero_a1_leakage_verified": True,
        "train_bases": {
            "n": len(train_bases),
            "by_fallacy": dict(Counter(r["fallacy_class"] for r in train_bases)),
            "by_depth": dict(Counter(str(r["difficulty_depth"]) for r in train_bases)),
            "by_pair_type": dict(Counter(r["pair_type"] for r in train_bases)),
            "by_tier": dict(Counter(r["tier"] for r in train_bases)),
        },
        "train_examples_nothink": {
            "n": len(train_examples),
            "by_role_tag": dict(Counter(r["role_tag"] for r in train_examples)),
            "by_arm": dict(Counter(r["arm"] for r in train_examples)),
        },
        "test_id_bases": {
            "n": len(test_id_bases),
            "by_fallacy": dict(Counter(r["fallacy_class"] for r in test_id_bases)),
            "by_depth": dict(Counter(str(r["difficulty_depth"]) for r in test_id_bases)),
        },
        "test_ood_bases": {
            "n": len(test_ood_bases),
            "by_split": dict(Counter(r["split"] for r in test_ood_bases)),
            "by_fallacy": dict(Counter(r["fallacy_class"] for r in test_ood_bases)),
            "by_depth": dict(Counter(str(r["difficulty_depth"]) for r in test_ood_bases)),
        },
        "sha256_lf": hashes,
    }
    (data_dir / "c1_datasets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print("[OK] C1 datasets built & zero-leakage verified:")
    for k_name, h in hashes.items():
        print(f"  {k_name:34s} sha256={h}")
    print("  train_examples by_role:", manifest["train_examples_nothink"]["by_role_tag"])
    print("  test_ood by_split:", manifest["test_ood_bases"]["by_split"])


if __name__ == "__main__":
    main()
