#!/usr/bin/env python3
"""Build zero-leakage, solver-certified, shortcut-audited datasets for ACL-exp-c1 (CPU only).

Design guarantees (updated after independent audit):
1. 100% DISJOINT from ACL-exp-a1 dev_only_pilot:
   - Brand-new 6 domains (marine_berth, rail_signal, energy_substation, clinic_triage, water_plant, vault_custody)
   - Brand-new entity names, zones, guards, metrics, predicates, outcome labels, and RNG seed (20270104)
   - Explicit runtime assertions verify ZERO shared base_ids, ZERO shared V/F contexts, ZERO shared entities with A1 v2.2.
2. Strict Split Isolation (Train / Dev / Test-ID / Test-OOD):
   - Train   (`c1_train_bases.jsonl`, 500 bases): 5 seen fallacy classes (scope_swap, cross_entity_transfer,
     missing_conjunct, negation_drop, threshold_violation), depth 2 & 3, 350 std + 150 hard.
     STRICTLY EXCLUDES `converse_error` and depth 4!
   - Dev     (`c1_dev_bases.jsonl`, 80 bases): same 5 seen classes (40 depth-2 std + 40 depth-3 hard), disjoint seeds.
     Used for Phase-0 recipe/hyperparameter selection so `c1_test_id` and `c1_test_ood` stay 100% sealed.
   - Test ID (`c1_test_id_bases.jsonl`, 100 bases): same 5 seen classes, disjoint seeds (50 std + 50 hard).
   - Test OOD (`c1_test_ood_bases.jsonl`, 120 bases):
       * 60 UNSEEN `converse_error` bases (30 std depth 2/3 + 30 hard depth 3)
       * 60 TRUE depth-4 chains (4 visible steps Step 1..4 -> Rule 5 override, 3 structural classes,
         broken steps balanced 20/20/20 across Steps 1, 2, 3, step-by-step proof verified!)
   - Cross-split deduplication checks BOTH `context_prompt_v` AND `context_prompt_f` AND `(ctx, arm)` message hashes
     across ALL 780 C1 bases and A1.
3. Shortcut Eliminations (all asserted at build time):
   - Decoupled `(domain, a0) -> at`: per-domain balanced transition cycling so empirical accuracy <= 0.45 (chance 0.333).
   - Counterbalanced `cross_entity_transfer/argument_flip` entity matching across depth 2, 3, and 4 -> exact 0.500.
   - Widened `threshold_violation` numeric range (`T ~ U[14, 92]`, `delta in {1, 2}`) -> single-number AUC <= 0.60.
   - Exact 50/50 rhetoric-tone symmetry in `c1_train_examples_nothink.jsonl`:
     `V_hes (250) vs F_hes (250)`, `V_auth (250) vs F_auth (250)`, `V (500) vs F (500)` paired within each base group.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_build_datasets.py
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
A1_SCRIPTS = REPO / "experiments" / "ACL-exp-a1" / "scripts"
A1_DATA = REPO / "experiments" / "ACL-exp-a1" / "data"
sys.path.insert(0, str(A1_SCRIPTS))

import audit_argument_only_artifacts as A1_AUDIT  # noqa: E402
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
# Added p_mid4 (4th intermediate predicate for true 4-hop OOD) and p_alt_mid (distinct decoy predicate)
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
            ("p_mid4", "receives final docking dispatch"),
            ("p_aux", "holds a tug permit"),
            ("p_alt_mid", "carries a channel clearance"),
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
            ("p_mid4", "receives final route lock"),
            ("p_aux", "carries a line token"),
            ("p_alt_mid", "holds a junction pass"),
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
            ("p_mid4", "receives final breaker release"),
            ("p_aux", "holds a sync relay"),
            ("p_alt_mid", "carries a feeder seal"),
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
            ("p_mid4", "receives final triage escalation"),
            ("p_aux", "has a specialist note"),
            ("p_alt_mid", "holds a senior signoff"),
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
            ("p_mid4", "receives final manifold transfer"),
            ("p_aux", "carries a chlorine tag"),
            ("p_alt_mid", "holds a intake voucher"),
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
            ("p_mid4", "receives final dispatch clearance"),
            ("p_aux", "holds a notary cipher"),
            ("p_alt_mid", "carries an escrow seal"),
            ("p_noise1", "cleared seal camera"),
            ("p_noise2", "passed weight ledger"),
        ),
        "outcomes": ("Class-P", "Class-Q", "Class-R", "Class-S"),
    },
)
DOMAIN_BY_NAME = {d["name"]: d for d in C1_DOMAINS}


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


def verify_step_by_step_proof(
    base_facts: set[str],
    obs_facts: set[str],
    rules: list[dict[str, Any]],
    claimed_step_atoms: list[str],
    expected_broken_step: int | None,
) -> dict[str, Any]:
    """Verify exact hop-by-hop entailment (shortest proof depth and exact first broken step).

    - If expected_broken_step is None (valid V chain):
      every claimed_step_atoms[k-1] MUST be derived at EXACT shortest hop depth == k
      (never already present in base_facts | obs_facts, never reachable in < k hops).
    - If expected_broken_step is b in {1, ..., len(claimed_step_atoms)} (invalid F chain):
      every step k < b MUST be genuinely derived at exact hop depth == k,
      and step b's claimed atom MUST NOT be entailed even at full fixed-point closure!
    """
    init_facts = set(base_facts) | set(obs_facts)
    # BFS / layered forward chaining to compute shortest derivation depth of every derivable atom
    depth_of: dict[str, int] = {a: 0 for a in init_facts}
    derived = set(init_facts)
    step_num = 0
    while True:
        step_num += 1
        newly: set[str] = set()
        for r in rules:
            if all(c in derived for c in r["conditions"]):
                conc = r["conclusion"]
                if conc not in derived:
                    newly.add(conc)
        if not newly:
            break
        for a in newly:
            depth_of[a] = step_num
        derived |= newly

    if expected_broken_step is None:
        for k, atom in enumerate(claimed_step_atoms, 1):
            assert atom in depth_of, f"Valid chain failed to derive step {k}: {atom}"
            assert depth_of[atom] == k, (
                f"Valid chain step {k} ({atom}) has shortest proof depth {depth_of[atom]} != {k}"
            )
        return {"shortest_valid_proof_hops": len(claimed_step_atoms), "first_invalid_step": None}
    else:
        first_invalid = None
        for k, atom in enumerate(claimed_step_atoms, 1):
            if k < expected_broken_step:
                assert atom in depth_of, f"Invalid chain broke early at step {k} (< {expected_broken_step}): {atom}"
                assert depth_of[atom] == k, (
                    f"Invalid chain prefix step {k} ({atom}) has depth {depth_of[atom]} != {k} (tautology or skip)"
                )
            elif k == expected_broken_step:
                assert atom not in depth_of, (
                    f"Invalid chain unexpectedly derived broken step {k} ({atom}) at depth {depth_of[atom]}"
                )
                first_invalid = k
        assert first_invalid == expected_broken_step
        return {"shortest_valid_proof_hops": None, "first_invalid_step": first_invalid}


def build_depth4_base(
    idx: int,
    rep: int,
    fc_idx: int,
    pair_type: str,
    fallacy_class: str,
    transition: tuple[str, str],
    default_policy_position: str,
    base_id: str,
) -> dict[str, Any]:
    """Build a TRUE 4-hop (Step 1..4 -> Rule 5 override) solver- & step-verified OOD base.

    Chain:
      Step 1 (Rule 1): obs  -> mid1
      Step 2 (Rule 2): mid1 -> mid2
      Step 3 (Rule 3): mid2 -> mid3
      Step 4 (Rule 4): mid3 -> mid4 (override_trigger)
      Conclusion (Rule 5): mid4 -> override assignment to out_at
    Broken steps are balanced 20/20/20 across Steps 1, 2, and 3!
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
    atom_noise1 = f"noise1({e_noise})"
    atom_noise2 = f"noise2({e2})"

    def mid_atom(level: int, ent: str) -> str:
        return f"obs({ent})" if level == 0 else f"mid{level}({ent})"

    def mid_pred(level: int) -> str:
        return p["p_obs"] if level == 0 else p[f"p_mid{level}"]

    atom_override = mid_atom(4, e1)  # mid4(e1) triggers Rule 5 override

    base_facts = {atom_base, atom_zone_a, atom_noise1, atom_noise2}
    base_fact_lines = [
        f"{e1} is registered under active default status.",
        f"{e1} belongs to {zone_a}.",
        f"For {e1}, {domain['guard_inactive_text']}.",
        f"{e_noise} {p['p_noise1']}.",
        f"{e2} {p['p_noise2']}.",
    ]
    default_rule = {
        "text": (
            f"Default policy: {e1} is under active default status, so {e1} is assigned to {out_a0} "
            f"unless the rules establish that {e1} {p['p_mid4']}."
        ),
    }
    r5_override = {
        "id": "Rule 5",
        "conditions": [atom_override],
        "min_threshold": None,
        "conclusion": f"final_is({at})",
        "text": f"If {e1} {p['p_mid4']}, override the default classification and assign {e1} to {out_at}.",
    }
    distractor_rule = {
        "id": "Rule 6",
        "conditions": [atom_noise1, atom_noise2],
        "min_threshold": None,
        "conclusion": f"log_ok({e_noise})",
        "text": f"If {e_noise} {p['p_noise1']} and {e2} {p['p_noise2']}, record a routine audit note for {e_noise}.",
    }
    lures = [
        {
            "id": "Lure 1",
            "conditions": [atom_obs_e2],
            "min_threshold": None,
            "conclusion": f"watch({e2})",
            "text": f"If {e2} {p['p_obs']}, then {e2} is placed on a watch list.",
        },
        {
            "id": "Lure 2",
            "conditions": [mid_atom(1, e1)],
            "min_threshold": None,
            "conclusion": f"logrev({e1})",
            "text": f"If {e1} {p['p_mid1']}, then {e1} receives a periodic log review.",
        },
        {
            "id": "Lure 3",
            "conditions": [atom_zone_b, atom_obs_e1],
            "min_threshold": None,
            "conclusion": f"archq({e1})",
            "text": f"For entries in {zone_b}, if {e1} {p['p_obs']}, then {e1} is queued for archival.",
        },
    ]

    # Exact 20/20/20 balance across broken steps 1, 2, 3!
    broken_step_idx = ((rep + fc_idx) % 3) + 1
    # Counterbalance auxiliary predicate identity AND entity match 50/50
    flip_aux = (rep % 2 == 1)
    flip_ent = (rep % 2 == 1)
    pred_req, pred_decoy = (p["p_alt_mid"], p["p_aux"]) if flip_aux else (p["p_aux"], p["p_alt_mid"])
    atom_req, atom_decoy = (f"alt_mid({e1})", f"aux({e1})") if flip_aux else (f"aux({e1})", f"alt_mid({e1})")

    chain_rules_v: list[dict[str, Any]] = []
    chain_rules_f: list[dict[str, Any]] = []
    extra_rules: list[dict[str, Any]] = []
    steps_v: list[str] = []
    steps_f: list[str] = []
    claimed_v: list[str] = []
    claimed_f: list[str] = []

    b = broken_step_idx
    if fallacy_class == "converse_error":
        if pair_type == "context_flip":
            obs_v = obs_f = f"New verified observation: {e1} {p['p_obs']}."
            obs_facts_v = obs_facts_f = {atom_obs_e1}
            for k in range(1, 5):
                src_a, dst_a = mid_atom(k - 1, e1), mid_atom(k, e1)
                src_t, dst_t = f"{e1} {mid_pred(k - 1)}", f"{e1} {mid_pred(k)}"
                rv = {"id": f"Rule {k}", "conditions": [src_a], "min_threshold": None, "conclusion": dst_a,
                      "text": f"If {src_t}, then {dst_t}."}
                if k == b:
                    rf = {"id": f"Rule {k}", "conditions": [dst_a], "min_threshold": None, "conclusion": src_a,
                          "text": f"If {dst_t}, then {src_t}."}
                else:
                    rf = rv
                chain_rules_v.append(rv)
                chain_rules_f.append(rf)
                st = (
                    f"Step 1: Since {src_t}, Rule 1 implies that {dst_t}."
                    if k == 1
                    else f"Step {k}: Applying Rule {k}, {dst_t}."
                )
                steps_v.append(st)
                steps_f.append(st)
                claimed_v.append(dst_a)
                claimed_f.append(dst_a)
        else:  # argument_flip
            if b == 1:
                obs_v = f"New verified observation: {e1} {pred_req}."
                obs_f = f"New verified observation: {e1} {pred_decoy}."
                obs_facts_v, obs_facts_f = {atom_req}, {atom_decoy}
            else:
                obs_v = f"New verified observation: {e1} {p['p_obs']} and {e1} {pred_req}."
                obs_f = f"New verified observation: {e1} {p['p_obs']} and {e1} {pred_decoy}."
                obs_facts_v = {atom_obs_e1, atom_req}
                obs_facts_f = {atom_obs_e1, atom_decoy}
            for k in range(1, 5):
                src_a, dst_a = mid_atom(k - 1, e1), mid_atom(k, e1)
                src_t, dst_t = f"{e1} {mid_pred(k - 1)}", f"{e1} {mid_pred(k)}"
                if k == b:
                    # Identical Turn-1 has BOTH converse rule (dst -> decoy) and forward rule ([src &] req -> dst)
                    r_conv = {"id": f"Rule {k}", "conditions": [dst_a], "min_threshold": None, "conclusion": atom_decoy,
                              "text": f"If {dst_t}, then {e1} {pred_decoy}."}
                    r_fwd = {
                        "id": "Rule_alt",
                        "conditions": [atom_req] if b == 1 else [src_a, atom_req],
                        "min_threshold": None,
                        "conclusion": dst_a,
                        "text": (
                            f"If {e1} {pred_req}, then {dst_t}."
                            if b == 1
                            else f"If {src_t} and {e1} {pred_req}, then {dst_t}."
                        ),
                    }
                    chain_rules_v.append(r_conv)
                    chain_rules_f.append(r_conv)
                    extra_rules.append(r_fwd)
                    steps_v.append(f"Step {k}: Since {e1} {pred_req}, the listed rule implies that {dst_t}.")
                    steps_f.append(f"Step {k}: Since {e1} {pred_decoy}, the listed rule implies that {dst_t}.")
                else:
                    r_ok = {"id": f"Rule {k}", "conditions": [src_a], "min_threshold": None, "conclusion": dst_a,
                            "text": f"If {src_t}, then {dst_t}."}
                    chain_rules_v.append(r_ok)
                    chain_rules_f.append(r_ok)
                    st = (
                        f"Step 1: Since {src_t}, Rule 1 implies that {dst_t}."
                        if k == 1
                        else f"Step {k}: Applying Rule {k}, {dst_t}."
                    )
                    steps_v.append(st)
                    steps_f.append(st)
                claimed_v.append(dst_a)
                claimed_f.append(dst_a)

    elif fallacy_class == "scope_swap":
        if pair_type == "context_flip":
            obs_v = obs_f = f"New verified observation: {e1} {p['p_obs']}."
            obs_facts_v = obs_facts_f = {atom_obs_e1}
            for k in range(1, 5):
                src_a, dst_a = mid_atom(k - 1, e1), mid_atom(k, e1)
                src_t, dst_t = f"{e1} {mid_pred(k - 1)}", f"{e1} {mid_pred(k)}"
                if k == b:
                    rv = {"id": f"Rule {k}", "conditions": [atom_zone_a, src_a], "min_threshold": None,
                          "conclusion": dst_a, "text": f"For entries in {zone_a}, if {src_t}, then {dst_t}."}
                    rf = {"id": f"Rule {k}", "conditions": [atom_zone_b, src_a], "min_threshold": None,
                          "conclusion": dst_a, "text": f"For entries in {zone_b}, if {src_t}, then {dst_t}."}
                else:
                    rv = rf = {"id": f"Rule {k}", "conditions": [src_a], "min_threshold": None,
                               "conclusion": dst_a, "text": f"If {src_t}, then {dst_t}."}
                chain_rules_v.append(rv)
                chain_rules_f.append(rf)
                st = (
                    f"Step 1: Since {src_t}, Rule 1 implies that {dst_t}."
                    if k == 1
                    else f"Step {k}: Applying Rule {k}, {dst_t}."
                )
                steps_v.append(st)
                steps_f.append(st)
                claimed_v.append(dst_a)
                claimed_f.append(dst_a)
        else:  # argument_flip (zero tautology: zone_a/zone_b already 50/50 randomized!)
            obs_v = f"New verified observation: {e1} {p['p_obs']} and is routed under {zone_b}."
            obs_f = f"New verified observation: {e1} {p['p_obs']} and is routed under {zone_a}."
            obs_facts_v = {atom_obs_e1, atom_zone_b}
            obs_facts_f = {atom_obs_e1}
            for k in range(1, 5):
                src_a, dst_a = mid_atom(k - 1, e1), mid_atom(k, e1)
                src_t, dst_t = f"{e1} {mid_pred(k - 1)}", f"{e1} {mid_pred(k)}"
                if k == b:
                    r_sc = {"id": f"Rule {k}", "conditions": [atom_zone_b, src_a], "min_threshold": None,
                            "conclusion": dst_a, "text": f"For entries in {zone_b}, if {src_t}, then {dst_t}."}
                    chain_rules_v.append(r_sc)
                    chain_rules_f.append(r_sc)
                    steps_v.append(f"Step {k}: Since {src_t} under {zone_b}, Rule {k} implies that {dst_t}.")
                    steps_f.append(f"Step {k}: Since {src_t} under {zone_a}, Rule {k} implies that {dst_t}.")
                else:
                    r_ok = {"id": f"Rule {k}", "conditions": [src_a], "min_threshold": None,
                            "conclusion": dst_a, "text": f"If {src_t}, then {dst_t}."}
                    chain_rules_v.append(r_ok)
                    chain_rules_f.append(r_ok)
                    st = (
                        f"Step 1: Since {src_t}, Rule 1 implies that {dst_t}."
                        if k == 1
                        else f"Step {k}: Applying Rule {k}, {dst_t}."
                    )
                    steps_v.append(st)
                    steps_f.append(st)
                claimed_v.append(dst_a)
                claimed_f.append(dst_a)

    else:  # cross_entity_transfer
        if pair_type == "context_flip":
            obs_ent = e2 if flip_ent else e1
            bad_ent = e1 if flip_ent else e2
            obs_v = obs_f = f"New verified observation: {obs_ent} {p['p_obs']}."
            obs_facts_v = obs_facts_f = {mid_atom(0, obs_ent)}
            for k in range(1, 5):
                if k < b:
                    src_a, dst_a = mid_atom(k - 1, obs_ent), mid_atom(k, obs_ent)
                    src_t, dst_t = f"{obs_ent} {mid_pred(k - 1)}", f"{obs_ent} {mid_pred(k)}"
                    rv = rf = {"id": f"Rule {k}", "conditions": [src_a], "min_threshold": None,
                               "conclusion": dst_a, "text": f"If {src_t}, then {dst_t}."}
                    st = (
                        f"Step 1: Since {src_t}, Rule 1 implies that {dst_t}."
                        if k == 1
                        else f"Step {k}: Applying Rule {k}, {dst_t}."
                    )
                    claimed_v.append(dst_a)
                    claimed_f.append(dst_a)
                elif k == b:
                    dst_a, dst_t = mid_atom(k, e1), f"{e1} {mid_pred(k)}"
                    src_t_v = f"{obs_ent} {mid_pred(k - 1)}"
                    src_t_f = f"{bad_ent} {mid_pred(k - 1)}"
                    rv = {"id": f"Rule {k}", "conditions": [mid_atom(k - 1, obs_ent)], "min_threshold": None,
                          "conclusion": dst_a, "text": f"If {src_t_v}, then {dst_t}."}
                    rf = {"id": f"Rule {k}", "conditions": [mid_atom(k - 1, bad_ent)], "min_threshold": None,
                          "conclusion": dst_a, "text": f"If {src_t_f}, then {dst_t}."}
                    st = (
                        f"Step 1: Since {src_t_v}, Rule 1 implies that {dst_t}."
                        if k == 1
                        else f"Step {k}: Applying Rule {k}, {dst_t}."
                    )
                    claimed_v.append(dst_a)
                    claimed_f.append(dst_a)
                else:
                    src_a, dst_a = mid_atom(k - 1, e1), mid_atom(k, e1)
                    src_t, dst_t = f"{e1} {mid_pred(k - 1)}", f"{e1} {mid_pred(k)}"
                    rv = rf = {"id": f"Rule {k}", "conditions": [src_a], "min_threshold": None,
                               "conclusion": dst_a, "text": f"If {src_t}, then {dst_t}."}
                    st = f"Step {k}: Applying Rule {k}, {dst_t}."
                    claimed_v.append(dst_a)
                    claimed_f.append(dst_a)
                chain_rules_v.append(rv)
                chain_rules_f.append(rf)
                steps_v.append(st)
                steps_f.append(st)
        else:  # argument_flip: counterbalance req_ent (e1 vs e2) 50/50 so entity-match acc == 0.500!
            req_ent = e2 if flip_ent else e1
            bad_ent = e1 if flip_ent else e2
            obs_v = f"New verified observation: {req_ent} {p['p_obs']}."
            obs_f = f"New verified observation: {bad_ent} {p['p_obs']}."
            obs_facts_v = {mid_atom(0, req_ent)}
            obs_facts_f = {mid_atom(0, bad_ent)}
            for k in range(1, 5):
                if k < b:
                    # Parallel valid prefix rules for BOTH e1 and e2 so both V and F genuinely pass Steps 1..b-1
                    r_e1 = {"id": f"Rule {k}", "conditions": [mid_atom(k - 1, e1)], "min_threshold": None,
                            "conclusion": mid_atom(k, e1),
                            "text": f"If {e1} {mid_pred(k - 1)}, then {e1} {mid_pred(k)}."}
                    r_e2 = {"id": f"Rule {k}b", "conditions": [mid_atom(k - 1, e2)], "min_threshold": None,
                            "conclusion": mid_atom(k, e2),
                            "text": f"If {e2} {mid_pred(k - 1)}, then {e2} {mid_pred(k)}."}
                    chain_rules_v.append(r_e1)
                    chain_rules_f.append(r_e1)
                    extra_rules.append(r_e2)
                    sv = (
                        f"Step 1: Since {req_ent} {mid_pred(0)}, the matching rule implies that {req_ent} {mid_pred(1)}."
                        if k == 1
                        else f"Step {k}: Applying the matching rule, {req_ent} {mid_pred(k)}."
                    )
                    sf = (
                        f"Step 1: Since {bad_ent} {mid_pred(0)}, the matching rule implies that {bad_ent} {mid_pred(1)}."
                        if k == 1
                        else f"Step {k}: Applying the matching rule, {bad_ent} {mid_pred(k)}."
                    )
                    steps_v.append(sv)
                    steps_f.append(sf)
                    claimed_v.append(mid_atom(k, req_ent))
                    claimed_f.append(mid_atom(k, bad_ent))
                elif k == b:
                    dst_a, dst_t = mid_atom(k, e1), f"{e1} {mid_pred(k)}"
                    r_crit = {"id": f"Rule {k}", "conditions": [mid_atom(k - 1, req_ent)], "min_threshold": None,
                              "conclusion": dst_a,
                              "text": f"If {req_ent} {mid_pred(k - 1)}, then {dst_t}."}
                    chain_rules_v.append(r_crit)
                    chain_rules_f.append(r_crit)
                    steps_v.append(
                        f"Step {k}: Since {req_ent} {mid_pred(k - 1)}, Rule {k} implies that {dst_t}."
                    )
                    steps_f.append(
                        f"Step {k}: Since {bad_ent} {mid_pred(k - 1)}, Rule {k} implies that {dst_t}."
                    )
                    claimed_v.append(dst_a)
                    claimed_f.append(dst_a)
                else:
                    src_a, dst_a = mid_atom(k - 1, e1), mid_atom(k, e1)
                    src_t, dst_t = f"{e1} {mid_pred(k - 1)}", f"{e1} {mid_pred(k)}"
                    r_ok = {"id": f"Rule {k}", "conditions": [src_a], "min_threshold": None,
                            "conclusion": dst_a, "text": f"If {src_t}, then {dst_t}."}
                    chain_rules_v.append(r_ok)
                    chain_rules_f.append(r_ok)
                    st = f"Step {k}: Applying Rule {k}, {dst_t}."
                    steps_v.append(st)
                    steps_f.append(st)
                    claimed_v.append(dst_a)
                    claimed_f.append(dst_a)

    concl_line = f"Conclusion: Under the override rule (Rule 5), {e1} is reassigned to {out_at} (Option {at})."
    steps_v.append(concl_line)
    steps_f.append(concl_line)

    rules_v = [*chain_rules_v, r5_override, *extra_rules, distractor_rule, *lures]
    rules_f = [*chain_rules_f, r5_override, *extra_rules, distractor_rule, *lures]

    # 1. Exact fixed-point solver certification
    _, init_v = A1_SYM.evaluate_world(base_facts, {}, rules_v, a0, atom_override, at)
    _, init_f = A1_SYM.evaluate_world(base_facts, {}, rules_f, a0, atom_override, at)
    _, post_v = A1_SYM.evaluate_world(base_facts | obs_facts_v, {}, rules_v, a0, atom_override, at)
    _, post_f = A1_SYM.evaluate_world(base_facts | obs_facts_f, {}, rules_f, a0, atom_override, at)
    assert init_v == a0 and init_f == a0 and post_v == at and post_f == a0, (
        base_id, fallacy_class, pair_type, broken_step_idx, init_v, init_f, post_v, post_f
    )

    # 2. Exact step-by-step shortest-proof & first-invalid-step verification
    pv = verify_step_by_step_proof(base_facts, obs_facts_v, rules_v, claimed_v, expected_broken_step=None)
    pf = verify_step_by_step_proof(base_facts, obs_facts_f, rules_f, claimed_f, expected_broken_step=broken_step_idx)
    assert pv["shortest_valid_proof_hops"] == 4
    assert pf["first_invalid_step"] == broken_step_idx

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
    assert "Step 4:" in core_v and "Rule 5" in core_v, f"Depth-4 missing Step 4 / Rule 5: {core_v}"

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
        "schema": "cedar.acl_exp_c1.symbolic.v2",
        "split": "ood_depth4",
        "tier": "ood_depth4",
        "base_id": base_id,
        "family_id": f"C1FAM-D4-{pair_type[:3].upper()}-{fallacy_class[:4].upper()}-{rep + 1:02d}",
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
            "shortest_valid_proof_hops": 4,
            "first_invalid_step": broken_step_idx,
            "turn2_byte_identical_vf": (core_v == core_f),
        },
        "arms": arms,
    }


def _widen_threshold_numbers(row: dict[str, Any], seed_val: int) -> None:
    """Eliminate single-number obs_val marginal shortcut in threshold_violation (AUC ~0.85 -> <= 0.58).

    Widens base_threshold to U[14, 92] with tight delta in {1, 2} while preserving exact solver semantics.
    """
    if row["fallacy_class"] != "threshold_violation":
        return
    rng = random.Random(C1_MASTER_SEED + 900001 + seed_val * 3571)
    metric = DOMAIN_BY_NAME[row["domain"]]["metric_name"]
    t_low = rng.randint(14, 92)
    d1 = rng.randint(1, 2)
    d2 = rng.randint(1, 2)
    re_thresh = re.compile(rf"(and the {re.escape(metric)} is at least )(\d+)(,)")
    re_obs = re.compile(rf"(with a {re.escape(metric)} of )(\d+)(\.)")

    if row["pair_type"] == "argument_flip":
        new_t, new_v, new_f = t_low, t_low + d1, t_low - d2
        assert new_v >= new_t and new_f < new_t
        for key in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
            new_txt, n = re_thresh.subn(rf"\g<1>{new_t}\g<3>", row[key])
            assert n == 1, f"threshold replace failed in {key} for {row['base_id']}"
            row[key] = new_txt
        for arm_name, txt in list(row["arms"].items()):
            if arm_name in ("V", "V_auth", "V_hes"):
                new_txt, n = re_obs.subn(rf"\g<1>{new_v}\g<3>", txt)
                assert n == 1, (row["base_id"], arm_name)
                row["arms"][arm_name] = new_txt
            elif arm_name.startswith("F"):
                new_txt, n = re_obs.subn(rf"\g<1>{new_f}\g<3>", txt)
                assert n == 1, (row["base_id"], arm_name)
                row["arms"][arm_name] = new_txt
        assert row["context_prompt_v"] == row["context_prompt_f"]
    else:  # context_flip
        new_tv = t_low
        new_obs = t_low + d1
        new_tf = new_obs + d2
        assert new_obs >= new_tv and new_obs < new_tf
        row["context_prompt_v"], nv = re_thresh.subn(rf"\g<1>{new_tv}\g<3>", row["context_prompt_v"])
        assert nv == 1
        for key in ("context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
            row[key], nf = re_thresh.subn(rf"\g<1>{new_tf}\g<3>", row[key])
            assert nf == 1
        for arm_name, txt in list(row["arms"].items()):
            if arm_name in ("V", "V_auth", "V_hes") or arm_name.startswith("F"):
                new_txt, n = re_obs.subn(rf"\g<1>{new_obs}\g<3>", txt)
                assert n == 1, (row["base_id"], arm_name)
                row["arms"][arm_name] = new_txt
        assert row["arms"]["V"] == row["arms"]["F"]


def _counterbalance_depth2_cross_entity(row: dict[str, Any], flip_ent: bool) -> None:
    """Counterbalance depth-2 cross_entity_transfer/argument_flip so obs_entity == conclusion_entity is 50/50."""
    if not flip_ent:
        return
    if not (
        row["fallacy_class"] == "cross_entity_transfer"
        and row["pair_type"] == "argument_flip"
        and row["difficulty_depth"] == 2
    ):
        return
    dom = DOMAIN_BY_NAME[row["domain"]]
    p = dict(dom["props"])
    m_v = re.match(rf"^New verified observation: ([^.]+?) {re.escape(p['p_obs'])}\.", row["arms"]["V"])
    m_f = re.match(rf"^New verified observation: ([^.]+?) {re.escape(p['p_obs'])}\.", row["arms"]["F"])
    assert m_v and m_f, f"Could not parse entities in {row['base_id']}"
    e1, e2 = m_v.group(1), m_f.group(1)
    assert e1 != e2
    old_r1 = f"Rule 1. If {e1} {p['p_obs']}, then {e1} {p['p_mid1']}."
    new_r1 = f"Rule 1. If {e2} {p['p_obs']}, then {e1} {p['p_mid1']}."
    for key in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
        assert old_r1 in row[key], (row["base_id"], key)
        row[key] = row[key].replace(old_r1, new_r1, 1)
    # Swap V <-> F core texts cleanly (re-wrapping all rhetoric & source arms)
    old_v_core = row["arms"]["V"]
    old_f_core = row["arms"]["F"]
    # New V uses old F core (observes e2, now satisfies Rule 1!); new F uses old V core (observes e1, now fails Rule 1!)
    arms = build_arms(
        core_v=old_f_core,
        core_f=old_v_core,
        shell_only=row["arms"]["shell_only"],
        a0=row["initial_answer"],
        at=row["target_answer"],
        i0_text=row["arms"]["I0_confirm_a0"],
        final_noun="classification",
    )
    row["arms"] = arms
    assert row["context_prompt_v"] == row["context_prompt_f"]


class SplitDeduplicator:
    """Guarantees ZERO collision on base_id, context_prompt_v, context_prompt_f, or (ctx, arm) message hashes."""

    def __init__(self) -> None:
        self.base_ids: set[str] = set()
        self.ctx_all: set[str] = set()
        self.msg_hashes: set[str] = set()
        self.per_domain_trans_count: Counter[int] = Counter()

    def next_balanced_transition(self, domain_idx: int, salt: int) -> tuple[str, str]:
        c = self.per_domain_trans_count[domain_idx]
        self.per_domain_trans_count[domain_idx] = c + 1
        # Independent per-domain balanced cycle over all 12 ordered transitions
        t_idx = (c + domain_idx * 7 + salt) % len(A1_SYM.ORDERED_TRANSITIONS)
        return A1_SYM.ORDERED_TRANSITIONS[t_idx]

    @staticmethod
    def _row_msg_hashes(row: dict[str, Any]) -> list[str]:
        hs = []
        for arm in ("V", "V_hes", "V_auth"):
            hs.append(hashlib.sha256((row["context_prompt_v"] + "\n||\n" + row["arms"][arm]).encode()).hexdigest())
        for arm in ("F", "F_auth", "F_hes"):
            hs.append(hashlib.sha256((row["context_prompt_f"] + "\n||\n" + row["arms"][arm]).encode()).hexdigest())
        return hs

    def register_external(self, row: dict[str, Any]) -> None:
        self.base_ids.add(row["base_id"])
        self.ctx_all.add(row["context_prompt_v"])
        self.ctx_all.add(row["context_prompt_f"])
        for h in self._row_msg_hashes(row):
            self.msg_hashes.add(h)

    def has_collision(self, row: dict[str, Any]) -> bool:
        if row["base_id"] in self.base_ids:
            return True
        if row["context_prompt_v"] in self.ctx_all or row["context_prompt_f"] in self.ctx_all:
            return True
        if any(h in self.msg_hashes for h in self._row_msg_hashes(row)):
            return True
        return False

    def commit(self, row: dict[str, Any]) -> None:
        assert not self.has_collision(row), f"Collision on {row['base_id']}"
        self.register_external(row)


def _make_standard_or_hard(
    global_idx: int,
    pair_type: str,
    depth: int,
    fc: str,
    pos: str,
    add_lures: bool,
    base_id: str,
    split_name: str,
    dedup: SplitDeduplicator,
    split_salt: int,
    flip_d2_ent: bool = False,
) -> dict[str, Any]:
    dom_idx = global_idx % len(C1_DOMAINS)
    trans = dedup.next_balanced_transition(dom_idx, split_salt)
    attempt = 0
    while True:
        # Keep domain fixed on retries by stepping by multiple of 6 (len(C1_DOMAINS))
        cur_idx = global_idx + attempt * 6 * 16667
        row = _with_c1_domains(
            A1_SYM.build_single_base,
            cur_idx,
            pair_type,
            depth,
            fc,
            trans,
            pos,
            id_prefix="C1",
            add_lures=add_lures,
        )
        row["schema"] = "cedar.acl_exp_c1.symbolic.v2"
        row["split"] = split_name
        row["base_id"] = base_id
        row["family_id"] = (
            f"C1{split_name.upper()[:3]}-{pair_type[:3].upper()}-D{depth}-{fc[:4].upper()}-{(global_idx // 10) + 1:03d}"
        )
        _widen_threshold_numbers(row, seed_val=cur_idx)
        _counterbalance_depth2_cross_entity(row, flip_ent=flip_d2_ent)
        if not dedup.has_collision(row):
            dedup.commit(row)
            return row
        attempt += 1
        assert attempt < 50, f"Too many dedup retries for {base_id}"


def load_a1_rows_into_dedup(dedup: SplitDeduplicator) -> tuple[set[str], set[str]]:
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
            dedup.register_external(r)
    return a1_ids, a1_ctxs


def verify_zero_leakage(
    splits: dict[str, list[dict[str, Any]]],
    a1_ids: set[str],
    a1_ctxs: set[str],
) -> None:
    a1_entities = {f"{d['entity_prefix']} {e}" for d in A1_SYM.DOMAINS for e in d["entities"]}
    c1_entities = {f"{d['entity_prefix']} {e}" for d in C1_DOMAINS for e in d["entities"]}
    assert not (a1_entities & c1_entities), f"Entity overlap with A1: {a1_entities & c1_entities}"

    all_rows = [r for rows in splits.values() for r in rows]
    assert len({r["base_id"] for r in all_rows}) == len(all_rows)
    for r in all_rows:
        assert r["base_id"] not in a1_ids, r["base_id"]
        assert r["context_prompt_f"] not in a1_ctxs, r["base_id"]
        assert r["context_prompt_v"] not in a1_ctxs, r["base_id"]

    # Pairwise cross-split check on V context, F context, AND (ctx, arm) message hashes
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            ctx1 = {r["context_prompt_v"] for r in splits[s1]} | {r["context_prompt_f"] for r in splits[s1]}
            ctx2 = {r["context_prompt_v"] for r in splits[s2]} | {r["context_prompt_f"] for r in splits[s2]}
            assert not (ctx1 & ctx2), f"Context overlap between {s1} and {s2}: {len(ctx1 & ctx2)}"
            h1 = {h for r in splits[s1] for h in SplitDeduplicator._row_msg_hashes(r)}
            h2 = {h for r in splits[s2] for h in SplitDeduplicator._row_msg_hashes(r)}
            assert not (h1 & h2), f"Full message overlap between {s1} and {s2}: {len(h1 & h2)}"


def materialize_train_examples(train_bases: list[dict], a1_cfg: dict) -> list[dict]:
    """Materialize tone-symmetric, paired training records for no_think SFT + Signed Readout Margin Loss.

    For each base_id (500 bases), creates an ATOMIC 6-item group (3,000 rule examples total):
      1. V           -> gold=at, margin_sign=+1  (500)
      2. F           -> gold=a0, margin_sign=-1  (500)
      3. V_hes/V_auth-> gold=at, margin_sign=+1  (250 V_hes + 250 V_auth)
      4. F_hes/F_auth-> gold=a0, margin_sign=-1  (250 F_hes + 250 F_auth, SAME tone as #3 within each base!)
      5. shell_only  -> gold=a0, margin_sign=-1  (500)
      6. Rotating    -> S (125, -1), P0_confirm_a0 (125, 0), I0_confirm_a0 (125, 0),
                        and AO (42, -1) / F_src_third_party (42, -1) / N (41, 0)

    Exact symmetries asserted:
      - P(accept | plain) == 0.500 (500 V vs 500 F)
      - P(accept | hes)   == 0.500 (250 V_hes vs 250 F_hes)
      - P(accept | auth)  == 0.500 (250 V_auth vs 250 F_auth)
    """
    extra_cycle = (
        ("AO", "reject_social", -1),
        ("F_src_third_party", "reject_source", -1),
        ("N", "no_conflict", 0),
    )
    examples = []
    extra_k = 0
    for i, row in enumerate(train_bases):
        a0, at = row["initial_answer"], row["target_answer"]
        if i % 2 == 0:
            rhet_v, rhet_f = "V_hes", "F_hes"
        else:
            rhet_v, rhet_f = "V_auth", "F_auth"

        slot = i % 4
        if slot == 0:
            s6 = ("S", a0, "reject_social", -1)
        elif slot == 1:
            s6 = ("P0_confirm_a0", a0, "no_conflict", 0)
        elif slot == 2:
            s6 = ("I0_confirm_a0", a0, "no_conflict", 0)
        else:
            arm_e, tag_e, sign_e = extra_cycle[extra_k % len(extra_cycle)]
            extra_k += 1
            s6 = (arm_e, a0, tag_e, sign_e)

        arm_specs = [
            ("V", at, "accept", +1),
            ("F", a0, "reject_F", -1),
            (rhet_v, at, "accept_rhet", +1),
            (rhet_f, a0, "reject_rhet", -1),
            ("shell_only", a0, "reject_shell", -1),
            s6,
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
                    "margin_sign": margin_sign,
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

    cnt = Counter(e["arm"] for e in examples)
    assert cnt["V"] == cnt["F"] == 500
    assert cnt["V_hes"] == cnt["F_hes"] == 250
    assert cnt["V_auth"] == cnt["F_auth"] == 250
    assert len(examples) == 3000
    return examples


def run_shortcut_audits(splits: dict[str, list[dict[str, Any]]], train_examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Run comprehensive shortcut + G0 artifact audits across all splits and assert hard thresholds."""
    report: dict[str, Any] = {}

    # 1. (domain, a0) -> at lookup shortcut (fit majority map on train, test on every split)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in splits["train"]:
        counts[(r["domain"], r["initial_answer"])][r["target_answer"]] += 1
    maj_map = {k: v.most_common(1)[0][0] for k, v in counts.items()}
    da_acc = {}
    for name, rows in splits.items():
        acc = float(np.mean([maj_map.get((r["domain"], r["initial_answer"]), "") == r["target_answer"] for r in rows]))
        da_acc[name] = round(acc, 4)
        assert acc <= 0.45, f"(domain, a0) -> at shortcut detected on {name}: {acc:.4f}"
    report["domain_a0_to_at_acc"] = da_acc

    # 2. cross_entity_transfer/argument_flip entity-match shortcut (obs_entity == conclusion_entity)
    ent_stats = {}
    for name, rows in splits.items():
        sub = [r for r in rows if r["fallacy_class"] == "cross_entity_transfer" and r["pair_type"] == "argument_flip"]
        if not sub:
            continue
        correct, total = 0, 0
        for r in sub:
            dom = DOMAIN_BY_NAME[r["domain"]]
            ep = dom["entity_prefix"]
            pat = re.compile(rf"\b{re.escape(ep)} [A-Za-z]+-\d+\b")
            for arm, label in (("V", 1), ("F", 0)):
                txt = r["arms"][arm]
                obs_part, concl_part = txt.split("Conclusion:")
                obs_ents = pat.findall(obs_part)
                concl_ents = pat.findall(concl_part)
                assert obs_ents and concl_ents, (r["base_id"], arm)
                pred = 1 if (obs_ents[0] == concl_ents[0]) else 0
                correct += int(pred == label)
                total += 1
        acc = correct / total
        ent_stats[name] = {"n_pairs": len(sub), "entity_match_acc": round(acc, 4)}
        assert 0.45 <= acc <= 0.55, f"Entity-match shortcut on {name}: {acc:.4f}"
    report["cross_entity_af_entity_match"] = ent_stats

    # 3. threshold_violation/argument_flip single-number obs_val ROC AUC
    num_stats = {}
    for name, rows in splits.items():
        sub = [r for r in rows if r["fallacy_class"] == "threshold_violation" and r["pair_type"] == "argument_flip"]
        if not sub:
            continue
        vals, ys = [], []
        for r in sub:
            metric = DOMAIN_BY_NAME[r["domain"]]["metric_name"]
            pat = re.compile(rf"with a {re.escape(metric)} of (\d+)\.")
            mv = pat.search(r["arms"]["V"])
            mf = pat.search(r["arms"]["F"])
            assert mv and mf, r["base_id"]
            vals.extend([int(mv.group(1)), int(mf.group(1))])
            ys.extend([1, 0])
        auc = float(roc_auc_score(ys, vals))
        num_stats[name] = {"n_pairs": len(sub), "obs_val_auc": round(auc, 4)}
        assert 0.40 <= auc <= 0.60, f"threshold_violation numeric shortcut on {name}: AUC={auc:.4f}"
    report["threshold_af_obs_val_auc"] = num_stats

    # 4. Tone symmetry in train_examples
    tone_stats = {}
    for tone, a_v, a_f in (("plain", "V", "F"), ("hes", "V_hes", "F_hes"), ("auth", "V_auth", "F_auth")):
        nv = sum(1 for e in train_examples if e["arm"] == a_v and e["gold_answer"] == e["target_answer"])
        nf = sum(1 for e in train_examples if e["arm"] == a_f and e["gold_answer"] == e["initial_answer"])
        tone_stats[tone] = {"accept": nv, "reject": nf, "p_accept": round(nv / (nv + nf), 4)}
        assert nv == nf, (tone, nv, nf)
    report["train_tone_symmetry"] = tone_stats

    # 5. G0 Argument-only TF-IDF GroupKFold audit (two-sided!)
    g0 = {}
    for name, rows in splits.items():
        cf = [r for r in rows if r["pair_type"] == "context_flip"]
        af = [r for r in rows if r["pair_type"] == "argument_flip"]
        ov = A1_AUDIT.run_cv_audit(rows, f"{name}_all")
        cf_res = A1_AUDIT.run_cv_audit(cf, f"{name}_cf")
        af_res = A1_AUDIT.run_cv_audit(af, f"{name}_af")
        assert cf_res["exact_turn2_byte_match_rate"] == 1.0, name
        ov_auc = ov["word_plus_char_ngram_lr"]["oof_roc_auc"]
        af_auc = af_res["word_plus_char_ngram_lr"]["oof_roc_auc"]
        assert 0.38 <= ov_auc <= 0.62, f"G0 overall AUC out of range on {name}: {ov_auc}"
        assert 0.35 <= af_auc <= 0.65, f"G0 argument_flip AUC out of range on {name}: {af_auc}"
        g0[name] = {
            "overall_combo_auc": ov_auc,
            "cf_byte_match": cf_res["exact_turn2_byte_match_rate"],
            "af_combo_auc": af_auc,
            "af_word_auc": af_res["word_ngram_lr"]["oof_roc_auc"],
        }
    report["g0_argument_only_audit"] = g0
    return report


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

    dedup = SplitDeduplicator()
    a1_ids, a1_ctxs = load_a1_rows_into_dedup(dedup)

    # 1. Train: 500 bases across 5 SEEN classes (no converse_error, no depth 4)
    train_bases: list[dict[str, Any]] = []
    gidx = 100
    k = 0
    d2_ce_af_k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(25):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3 and rep >= 10)  # 150 hard, 350 std
                for fc in SEEN_CLASSES:
                    k += 1
                    bid = f"C1-TRN-{k:04d}"
                    flip_d2 = False
                    if depth == 2 and fc == "cross_entity_transfer" and pair_type == "argument_flip":
                        flip_d2 = (d2_ce_af_k % 2 == 1)
                        d2_ce_af_k += 1
                    train_bases.append(
                        _make_standard_or_hard(gidx, pair_type, depth, fc, pos, add_lures, bid, "train", dedup, split_salt=0, flip_d2_ent=flip_d2)
                    )
                    gidx += 1
    assert len(train_bases) == 500
    assert all(r["fallacy_class"] != UNSEEN_CLASS and r["difficulty_depth"] in (2, 3) for r in train_bases)

    # 2. Dedicated Dev Set: 80 bases across 5 SEEN classes (40 depth-2 std + 40 depth-3 hard), disjoint seeds
    # Used for Phase-0 recipe selection so c1_test_id & c1_test_ood stay strictly sealed!
    dev_bases: list[dict[str, Any]] = []
    gidx = 5000
    k = 0
    d2_ce_af_k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(4):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3)  # 40 std + 40 hard
                for fc in SEEN_CLASSES:
                    k += 1
                    bid = f"C1-DEV-{k:03d}"
                    flip_d2 = False
                    if depth == 2 and fc == "cross_entity_transfer" and pair_type == "argument_flip":
                        flip_d2 = (d2_ce_af_k % 2 == 1)
                        d2_ce_af_k += 1
                    dev_bases.append(
                        _make_standard_or_hard(gidx, pair_type, depth, fc, pos, add_lures, bid, "dev", dedup, split_salt=3, flip_d2_ent=flip_d2)
                    )
                    gidx += 1
    assert len(dev_bases) == 80

    # 3. Held-out ID Test: 100 bases across the same 5 SEEN classes (50 std + 50 hard)
    test_id_bases: list[dict[str, Any]] = []
    gidx = 10000
    k = 0
    d2_ce_af_k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(5):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3)
                for fc in SEEN_CLASSES:
                    k += 1
                    bid = f"C1-ID-{k:03d}"
                    flip_d2 = False
                    if depth == 2 and fc == "cross_entity_transfer" and pair_type == "argument_flip":
                        flip_d2 = (d2_ce_af_k % 2 == 1)
                        d2_ce_af_k += 1
                    test_id_bases.append(
                        _make_standard_or_hard(gidx, pair_type, depth, fc, pos, add_lures, bid, "test_id", dedup, split_salt=5, flip_d2_ent=flip_d2)
                    )
                    gidx += 1
    assert len(test_id_bases) == 100

    # 4. Held-out OOD Test: 120 bases = 60 UNSEEN converse_error (depth 2/3) + 60 TRUE depth-4 (3 structural classes)
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
                    _make_standard_or_hard(gidx, pair_type, depth, UNSEEN_CLASS, pos, add_lures, bid, "ood_converse", dedup, split_salt=7)
                )
                gidx += 1

    d4_k = 0
    for pair_type in ("context_flip", "argument_flip"):
        for rep in range(10):
            before = ((rep % 2 == 0) ^ (pair_type == "argument_flip"))
            pos = "before_rules" if before else "after_rules"
            for fc_idx, fc in enumerate(STRUCTURAL_CLASSES):
                d4_k += 1
                bid = f"C1-OOD-D4-{d4_k:03d}"
                dom_idx = d4_k % len(C1_DOMAINS)
                trans = dedup.next_balanced_transition(dom_idx, salt=11)
                attempt = 0
                while True:
                    cur_idx = d4_k + attempt * 6 * 16667
                    row = build_depth4_base(cur_idx, rep, fc_idx, pair_type, fc, trans, pos, bid)
                    if not dedup.has_collision(row):
                        dedup.commit(row)
                        test_ood_bases.append(row)
                        break
                    attempt += 1
                    assert attempt < 50
    assert len(test_ood_bases) == 120
    d4_steps = Counter(r["broken_step_index"] for r in test_ood_bases if r["difficulty_depth"] == 4)
    assert d4_steps == {1: 20, 2: 20, 3: 20}, f"Depth-4 broken step imbalance: {d4_steps}"

    splits = {
        "train": train_bases,
        "dev": dev_bases,
        "test_id": test_id_bases,
        "test_ood": test_ood_bases,
    }
    verify_zero_leakage(splits, a1_ids, a1_ctxs)

    train_examples = materialize_train_examples(train_bases, a1_cfg)
    shortcut_report = run_shortcut_audits(splits, train_examples)

    hashes = {
        "c1_train_bases.jsonl": write_jsonl(train_bases, data_dir / "c1_train_bases.jsonl"),
        "c1_dev_bases.jsonl": write_jsonl(dev_bases, data_dir / "c1_dev_bases.jsonl"),
        "c1_train_examples_nothink.jsonl": write_jsonl(train_examples, data_dir / "c1_train_examples_nothink.jsonl"),
        "c1_test_id_bases.jsonl": write_jsonl(test_id_bases, data_dir / "c1_test_id_bases.jsonl"),
        "c1_test_ood_bases.jsonl": write_jsonl(test_ood_bases, data_dir / "c1_test_ood_bases.jsonl"),
    }

    manifest = {
        "schema": "cedar.acl_exp_c1.dataset_manifest.v2",
        "master_seed": C1_MASTER_SEED,
        "zero_a1_leakage_verified": True,
        "zero_cross_split_vf_leakage_verified": True,
        "counts": {
            "train_bases": len(train_bases),
            "train_examples_nothink": len(train_examples),
            "dev_bases": len(dev_bases),
            "test_id_bases": len(test_id_bases),
            "test_ood_bases": len(test_ood_bases),
            "train_by_class": dict(Counter(r["fallacy_class"] for r in train_bases)),
            "train_by_depth": dict(Counter(str(r["difficulty_depth"]) for r in train_bases)),
            "dev_by_class": dict(Counter(r["fallacy_class"] for r in dev_bases)),
            "test_id_by_class": dict(Counter(r["fallacy_class"] for r in test_id_bases)),
            "test_ood_by_split": dict(Counter(r["split"] for r in test_ood_bases)),
            "test_ood_depth4_broken_steps": {str(k): v for k, v in sorted(d4_steps.items())},
            "train_examples_by_arm": dict(Counter(e["arm"] for e in train_examples)),
            "train_examples_by_margin_sign": dict(Counter(str(e["margin_sign"]) for e in train_examples)),
        },
        "shortcut_audits": shortcut_report,
        "sha256": hashes,
    }
    (data_dir / "c1_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[OK] Built & verified all C1 datasets (v2):")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
