#!/usr/bin/env python3
"""Build frozen zero-training Entity-Count-Balanced Diagnostic Benchmark (`c3_entity_balanced_v1`).

Directly answers Astra's Stage 5 Audit Priority #2 (entity-count shortcut diagnostic on `missing_conjunct`):
  In `v2`/`v3` `missing_conjunct / argument_flip`, counting distinct entities in Turn-2's first sentence separated
  212/212 V vs F (`V` had 1 entity `{e1}`, `F` had 2 entities `{e1, e2}`).

This generator creates 480 strictly solver-certified dual-premise bases (`480` independent generation units) across
6 balanced 80-base diagnostic cells (using the 6 unseen V3 domains & 2 unseen V3 styles, ZERO overlap with any open
train/dev/v1/v2/v3 split, NEVER touching sealed `c1_test_id` / `c1_test_ood`):

  1. `1v2_orig` (`argument_flip`, 80 bases): rule requires `aux(e1)`; V obs has 1 entity `{e1}`, F obs has 2 `{e1, e2}`
  2. `2v1_anti` (`argument_flip`, 80 bases): rule requires `aux(e2)`; V obs has 2 entities `{e1, e2}`, F obs has 1 `{e1}`
     -> EXACT ANTI-SHORTCUT REVERSAL! A pure "1 entity = valid" heuristic scores 0% here!
  3. `1v1_same` (`argument_flip`, 80 bases): rule requires `aux(e1)`; BOTH V & F obs have 1 entity `{e1}`
     (V asserts `aux(e1)`, F asserts wrong property `alt_cred(e1)` on the SAME entity `e1`).
  4. `2v2_same` (`argument_flip`, 80 bases): rule requires `aux(e2)`; BOTH V & F obs have 2 entities
     (V asserts `aux(e2)` -> `{e1, e2}`, F asserts `aux(e3)` -> `{e1, e3}` — must match the rule's entity!).
  5. `1v1_ctx`  (`context_flip`,  80 bases): Turn-2 obs is BYTE-FOR-BYTE IDENTICAL 1-entity (`{e1}`);
     context rule requires `aux(e1)` (V) vs `alt_cred(e1)` (F).
  6. `2v2_ctx`  (`context_flip`,  80 bases): Turn-2 obs is BYTE-FOR-BYTE IDENTICAL 2-entity (`{e1, e2}`);
     context rule requires `aux(e2)` (V) vs `aux(e3)` (F).

Exact mathematical balance hard-asserted on disk:
  - Distinct entity count in Turn-2 observation sentence:
      V: 240 one-entity (50.0%) + 240 two-entity (50.0%)
      F: 240 one-entity (50.0%) + 240 two-entity (50.0%)
    -> Exact entity-count classifier AUC = 0.5000 (50.00% accuracy)!
  - Every single base hard-asserted by independent DAG solver:
      full V -> target_answer, drop obsA -> initial_answer, drop obsB -> initial_answer, full F -> initial_answer!

Usage:
  python experiments/ACL-exp-c3.5/scripts/c35_build_entity_balanced.py
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
C35 = HERE.parents[0]
REPO = C35.parents[1]
A1_SCRIPTS = REPO / "experiments" / "ACL-exp-a1" / "scripts"
C1_SCRIPTS = REPO / "experiments" / "ACL-exp-c1" / "scripts"
C1_DATA = REPO / "experiments" / "ACL-exp-c1" / "data"
C2_DATA = REPO / "experiments" / "ACL-exp-c2" / "data"
C3_SCRIPTS = REPO / "experiments" / "ACL-exp-c3" / "scripts"
C3_DATA = REPO / "experiments" / "ACL-exp-c3" / "data"
C35_DATA = C35 / "data"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(A1_SCRIPTS))
sys.path.insert(0, str(C1_SCRIPTS))
sys.path.insert(0, str(C3_SCRIPTS))

import build_symbolic_pilot_v2 as A1_SYM  # noqa: E402
import c1_build_datasets as C1B  # noqa: E402
import c3_build_datasets as C3B  # noqa: E402
import c35_build_blind_v2 as B2  # noqa: E402
import c35_build_blind_v3 as B3  # noqa: E402

ARM_ORDER = B2.ARM_ORDER
build_arms = B2.build_arms

ENTITY_BAL_SEED = 20310419
OPTION_LETTERS = ("A", "B", "C", "D")

# Brand-new secondary credential predicate per V3 domain (disjoint from p_aux & all other predicates!)
V3_ALT_CRED: dict[str, str] = {
    "botanical_conservatory": "carries a botanical transit pass",
    "tidal_barrage": "holds a spillway telemetry token",
    "archive_bindery": "bears a provenance seal",
    "glacier_station": "holds a cryogenics manifest",
    "wind_tunnel": "carries an instrumentation clearance",
    "kiln_foundry": "holds a refractory batch slip",
}

CELL_SPECS: tuple[tuple[str, str, int, int], ...] = (
    # (cell_name, pair_type, v_ent_count, f_ent_count)
    ("1v2_orig", "argument_flip", 1, 2),
    ("2v1_anti", "argument_flip", 2, 1),
    ("1v1_same", "argument_flip", 1, 1),
    ("2v2_same", "argument_flip", 2, 2),
    ("1v1_ctx", "context_flip", 1, 1),
    ("2v2_ctx", "context_flip", 2, 2),
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_entities_in_first_sentence(arm_text: str, domain_entities: tuple[str, ...]) -> set[str]:
    """Count distinct domain entity mentions in Turn-2's first sentence (exact Astra shortcut audit)."""
    first_sent = arm_text.split(".", 1)[0]
    return {e for e in domain_entities if e in first_sent}


def _build_entity_balanced_base(
    *,
    domains: tuple[dict[str, Any], ...],
    seed_base: int,
    idx: int,
    dom_idx: int,
    cell_name: str,
    pair_type: str,
    expected_v_ents: int,
    expected_f_ents: int,
    broken_step: int,
    pos: str,
    style: str,
    trans: tuple[str, str],
    bid: str,
    split_name: str,
) -> dict[str, Any]:
    dom = domains[dom_idx]
    rng = random.Random(seed_base + idx * 9187)
    ents = list(dom["entities"])
    rng.shuffle(ents)
    e1, e2, e3, e4 = ents[0], ents[1], ents[2], ents[3]

    zones = list(dom["zones"])
    rng.shuffle(zones)
    z1, z2 = zones[0], zones[1]

    outcomes = list(dom["outcomes"])
    rng.shuffle(outcomes)
    a0, at = trans
    other_letters = [k for k in OPTION_LETTERS if k not in (a0, at)]
    letter_to_outcome = {a0: outcomes[0], at: outcomes[1], other_letters[0]: outcomes[2], other_letters[1]: outcomes[3]}
    a0_name, at_name, a_decoy_name = letter_to_outcome[a0], letter_to_outcome[at], letter_to_outcome[other_letters[0]]

    p = dict(dom["props"])
    p_obsA = p["p_obs"]
    p_obsB = B3.V3_DOMAIN_OBS_B[dom["name"]]
    p_bA = p["p_mid1"]
    p_bB = p["p_mid2"]
    p_trig = p["p_mid3"]
    p_aux = p["p_aux"]
    p_alt_cred = V3_ALT_CRED[dom["name"]]
    p_alt = p["p_alt_mid"]
    override_trigger = f"trig({e1})"

    step_in_text = {
        1: f"{e1} {p_obsA}",
        2: f"{e1} {p_obsB}",
        3: f"{e1} {p_bA} and {e1} {p_bB}",
    }
    step_out_text = {
        1: f"{e1} {p_bA}",
        2: f"{e1} {p_bB}",
        3: f"{e1} {p_trig}",
    }
    step_in_pred = {
        1: [f"obsA({e1})"],
        2: [f"obsB({e1})"],
        3: [f"bA({e1})", f"bB({e1})"],
    }
    step_out_pred = {1: f"bA({e1})", 2: f"bB({e1})", 3: override_trigger}

    # Shared baseline facts (all 4 entities appear in background facts so entity names are never unseen!)
    facts_shared = [
        f"{e1} is registered under active default status.",
        f"{e1} belongs to {z1}.",
        f"{e2} belongs to {z2}.",
        f"For {e1}, {dom['guard_inactive_text']}.",
        f"{e3} {p['p_noise1']}.",
        f"{e4} {p['p_noise2']}.",
    ]
    bf_shared = {f"default({e1})", f"in_zone({e1},{z1})", f"in_zone({e2},{z2})", f"guard_inactive({e1})", f"n1({e3})", f"n2({e4})"}
    facts_v, facts_f = list(facts_shared), list(facts_shared)
    bf_v, bf_f = set(bf_shared), set(bf_shared)

    if cell_name == "1v2_orig":
        gate_v_txt = f"If {step_in_text[broken_step]} and {e1} {p_aux}, then {step_out_text[broken_step]}."
        gate_f_txt = gate_v_txt
        gate_v_cond = step_in_pred[broken_step] + [f"aux({e1})"]
        gate_f_cond = list(gate_v_cond)
        obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} {p_aux}."
        obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e2} {p_aux}."
        obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e1})"}
        obs_f_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e2})"}

    elif cell_name == "2v1_anti":
        gate_v_txt = f"If {step_in_text[broken_step]} and {e2} {p_aux}, then {step_out_text[broken_step]}."
        gate_f_txt = gate_v_txt
        gate_v_cond = step_in_pred[broken_step] + [f"aux({e2})"]
        gate_f_cond = list(gate_v_cond)
        obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e2} {p_aux}."
        obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} {p_aux}."
        obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e2})"}
        obs_f_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e1})"}

    elif cell_name == "1v1_same":
        # Flip which credential is required on odd/even idx so neither predicate string correlates with V!
        req_p, bad_p = (p_aux, p_alt_cred) if (idx % 2 == 0) else (p_alt_cred, p_aux)
        req_k, bad_k = ("aux", "altcred") if (idx % 2 == 0) else ("altcred", "aux")
        gate_v_txt = f"If {step_in_text[broken_step]} and {e1} {req_p}, then {step_out_text[broken_step]}."
        gate_f_txt = gate_v_txt
        gate_v_cond = step_in_pred[broken_step] + [f"{req_k}({e1})"]
        gate_f_cond = list(gate_v_cond)
        obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} {req_p}."
        obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} {bad_p}."
        obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"{req_k}({e1})"}
        obs_f_set = {f"obsA({e1})", f"obsB({e1})", f"{bad_k}({e1})"}

    elif cell_name == "2v2_same":
        gate_v_txt = f"If {step_in_text[broken_step]} and {e2} {p_aux}, then {step_out_text[broken_step]}."
        gate_f_txt = gate_v_txt
        gate_v_cond = step_in_pred[broken_step] + [f"aux({e2})"]
        gate_f_cond = list(gate_v_cond)
        obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e2} {p_aux}."
        obs_f_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e3} {p_aux}."
        obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e2})"}
        obs_f_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e3})"}

    elif cell_name == "1v1_ctx":
        req_p, bad_p = (p_aux, p_alt_cred) if (idx % 2 == 0) else (p_alt_cred, p_aux)
        req_k, bad_k = ("aux", "altcred") if (idx % 2 == 0) else ("altcred", "aux")
        gate_v_txt = f"If {step_in_text[broken_step]} and {e1} {req_p}, then {step_out_text[broken_step]}."
        gate_f_txt = f"If {step_in_text[broken_step]} and {e1} {bad_p}, then {step_out_text[broken_step]}."
        gate_v_cond = step_in_pred[broken_step] + [f"{req_k}({e1})"]
        gate_f_cond = step_in_pred[broken_step] + [f"{bad_k}({e1})"]
        obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e1} {req_p}."
        obs_f_text = obs_v_text  # BYTE-FOR-BYTE IDENTICAL!
        obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"{req_k}({e1})"}
        obs_f_set = set(obs_v_set)

    elif cell_name == "2v2_ctx":
        gate_v_txt = f"If {step_in_text[broken_step]} and {e2} {p_aux}, then {step_out_text[broken_step]}."
        gate_f_txt = f"If {step_in_text[broken_step]} and {e3} {p_aux}, then {step_out_text[broken_step]}."
        gate_v_cond = step_in_pred[broken_step] + [f"aux({e2})"]
        gate_f_cond = step_in_pred[broken_step] + [f"aux({e3})"]
        obs_v_text = f"{e1} {p_obsA}, independently {e1} {p_obsB}, and {e2} {p_aux}."
        obs_f_text = obs_v_text  # BYTE-FOR-BYTE IDENTICAL!
        obs_v_set = {f"obsA({e1})", f"obsB({e1})", f"aux({e2})"}
        obs_f_set = set(obs_v_set)

    else:
        raise ValueError(cell_name)

    sp_shared = f"Applying Rule {broken_step} with that reported credential, {step_out_text[broken_step]}."
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

    # STRICT INDEPENDENT SOLVER CERTIFICATION
    _, t1v = A1_SYM.evaluate_world(bf_v, {}, rules_v, a0_name, override_trigger, at_name)
    _, t1f = A1_SYM.evaluate_world(bf_f, {}, rules_f, a0_name, override_trigger, at_name)
    assert t1v == a0_name and t1f == a0_name
    _, outv = A1_SYM.evaluate_world(bf_v | obs_v_set, {}, rules_v, a0_name, override_trigger, at_name)
    _, outf = A1_SYM.evaluate_world(bf_f | obs_f_set, {}, rules_f, a0_name, override_trigger, at_name)
    assert outv == at_name and outf == a0_name, (bid, cell_name, outv, outf)
    # Dual-premise independence check
    _, drop_a = A1_SYM.evaluate_world(bf_v | (obs_v_set - {f"obsA({e1})"}), {}, rules_v, a0_name, override_trigger, at_name)
    _, drop_b = A1_SYM.evaluate_world(bf_v | (obs_v_set - {f"obsB({e1})"}), {}, rules_v, a0_name, override_trigger, at_name)
    assert drop_a == a0_name and drop_b == a0_name
    B2._verify_dual_premise_dag(bf_v, obs_v_set, rules_v, f"bA({e1})", f"bB({e1})", override_trigger, expected_broken_step=None)
    B2._verify_dual_premise_dag(bf_f, obs_f_set, rules_f, f"bA({e1})", f"bB({e1})", override_trigger, expected_broken_step=broken_step)

    ctx_v = B3._render_v3_context(dom, facts_v, rlines_v, e1, letter_to_outcome, default_policy_text, pos, style)
    ctx_f = B3._render_v3_context(dom, facts_f, rlines_f, e1, letter_to_outcome, default_policy_text, pos, style)
    if pair_type == "context_flip":
        assert ctx_v != ctx_f
    else:
        assert ctx_v == ctx_f

    def _build_core(obs_t: str) -> str:
        s1_t = sp_shared if broken_step == 1 else f"Since {e1} {p_obsA}, Rule 1 establishes that {e1} {p_bA}."
        s2_t = sp_shared if broken_step == 2 else f"Independently, since {e1} {p_obsB}, Rule 2 establishes that {e1} {p_bB}."
        s3_t = sp_shared if broken_step == 3 else f"Combining both independent statuses ({e1} {p_bA} and {e1} {p_bB}) under Rule 3, {e1} {p_trig}."
        return (
            f"New verified observation: {obs_t} "
            f"Step 1: {s1_t} "
            f"Step 2: {s2_t} "
            f"Step 3: {s3_t} "
            f"Conclusion: Under the override rule (Rule 4), {e1} is reassigned to {at_name} (Option {at})."
        )

    core_v = _build_core(obs_v_text)
    core_f = _build_core(obs_f_text)
    if pair_type == "context_flip":
        assert core_v == core_f

    # Hard-assert exact entity counts in Turn-2 first sentence BEFORE and AFTER v3 discourse rewrite!
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
            arms[an] = B3._rewrite_arg_v3_style(arms[an], style)
    assert tuple(arms) == ARM_ORDER

    v_ents = _count_entities_in_first_sentence(arms["V"], dom["entities"])
    f_ents = _count_entities_in_first_sentence(arms["F"], dom["entities"])
    assert len(v_ents) == expected_v_ents, (bid, cell_name, len(v_ents), expected_v_ents, arms["V"])
    assert len(f_ents) == expected_f_ents, (bid, cell_name, len(f_ents), expected_f_ents, arms["F"])
    # Also verify tone variants keep identical first-sentence entity counts!
    for tone_suf in ("_auth", "_hes"):
        assert len(_count_entities_in_first_sentence(arms[f"V{tone_suf}"], dom["entities"])) == expected_v_ents
        assert len(_count_entities_in_first_sentence(arms[f"F{tone_suf}"], dom["entities"])) == expected_f_ents

    return {
        "schema": "cedar.acl_exp_c35.entity_balanced_diag.v1",
        "split": split_name,
        "base_id": bid,
        "cell_id": bid,
        "family_id": bid,
        "unit_id": bid,
        "domain": dom["name"],
        "pair_type": pair_type,
        "difficulty_depth": 3,
        "fallacy_class": "missing_conjunct",
        "entity_balance_cell": cell_name,
        "v_first_sent_entities": expected_v_ents,
        "f_first_sent_entities": expected_f_ents,
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


def main() -> None:
    C35_DATA.mkdir(parents=True, exist_ok=True)
    # Deduplicate against ALL open splits INCLUDING v2 & v3 (NEVER touch sealed c1_test_id / c1_test_ood!)
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
        C3_DATA / "c3_blind_confirm_bases.jsonl",
        C35_DATA / "c3_blind_confirm_v2_bases.jsonl",
        C35_DATA / "c3_blind_confirm_v3_bases.jsonl",
    ]
    for p in open_files:
        assert "c1_test_id" not in p.name and "c1_test_ood" not in p.name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                dedup.register_external(json.loads(line))

    n_per_cell = 80
    total_n = len(CELL_SPECS) * n_per_cell
    assert total_n == 480

    facs = C3B._deconfounded_factors(total_n, len(B3.V3_BLIND_DOMAINS), seed=ENTITY_BAL_SEED + 77, include_style=True)
    rows: list[dict[str, Any]] = []
    idx_global = 0
    for cell_name, pair_type, v_ent, f_ent in CELL_SPECS:
        for rep in range(n_per_cell):
            idx_global += 1
            fac = facs[idx_global - 1]
            dom_idx, pos = fac["dom_idx"], fac["pos"]
            style = "style_e_brief" if fac["style"] == "style_a" else "style_f_ledger"
            bstep = (rep % 3) + 1
            trans = dedup.next_balanced_transition(dom_idx, salt=509)
            bid = f"C35EB-{cell_name.upper()}-{rep + 1:03d}"
            attempt = 0
            while True:
                cur_idx = 3100000 + idx_global + attempt * 19997
                row = _build_entity_balanced_base(
                    domains=B3.V3_BLIND_DOMAINS,
                    seed_base=ENTITY_BAL_SEED,
                    idx=cur_idx,
                    dom_idx=dom_idx,
                    cell_name=cell_name,
                    pair_type=pair_type,
                    expected_v_ents=v_ent,
                    expected_f_ents=f_ent,
                    broken_step=bstep,
                    pos=pos,
                    style=style,
                    trans=trans,
                    bid=bid,
                    split_name="c3_entity_balanced_v1",
                )
                if not dedup.has_collision(row):
                    dedup.commit(row)
                    rows.append(row)
                    break
                attempt += 1

    # STRICT ENTITY-COUNT SHORTCUT ELIMINATION AUDIT
    v_counts = Counter(r["v_first_sent_entities"] for r in rows)
    f_counts = Counter(r["f_first_sent_entities"] for r in rows)
    assert v_counts == {1: 240, 2: 240}, v_counts
    assert f_counts == {1: 240, 2: 240}, f_counts
    # Exact ROC AUC of entity-count feature `y=1(V), score=-n_entities`:
    # Positives: 240 at -1, 240 at -2. Negatives: 240 at -1, 240 at -2 -> AUC = 0.5000 exactly!
    concordant = v_counts[1] * f_counts[2]
    ties = v_counts[1] * f_counts[1] + v_counts[2] * f_counts[2]
    entity_count_auc = (concordant + 0.5 * ties) / (len(rows) * len(rows))
    assert abs(entity_count_auc - 0.5) < 1e-12, entity_count_auc

    # Verify G0 TF-IDF separability on V vs F argument text using exact C3 grouped CV audit
    g0_audit = C3B._run_g0_tfidf_auc(rows)
    assert 0.42 <= g0_audit["overall_auc"] <= 0.58 and g0_audit["cf_byte_match"] == 1.0, g0_audit

    out_jsonl = C35_DATA / "c3_entity_balanced_v1_bases.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8", newline="\n")
    out_sha = _sha256_file(out_jsonl)

    manifest = {
        "schema": "cedar.acl_exp_c35.entity_balanced_manifest.v1",
        "seed": ENTITY_BAL_SEED,
        "n_bases": len(rows),
        "n_independent_units": len({r["unit_id"] for r in rows}),
        "sha256": out_sha,
        "by_cell": dict(Counter(r["entity_balance_cell"] for r in rows)),
        "by_pair_type": dict(Counter(r["pair_type"] for r in rows)),
        "by_broken_step": dict(Counter(r["broken_step"] for r in rows)),
        "by_domain": dict(Counter(r["domain"] for r in rows)),
        "by_style": dict(Counter(r["phrasing_style"] for r in rows)),
        "v_first_sent_entity_counts": dict(v_counts),
        "f_first_sent_entity_counts": dict(f_counts),
        "entity_count_shortcut_auc": round(entity_count_auc, 6),
        "g0_tfidf_audit": g0_audit,
        "solver_certified_all": True,
        "sealed_test_untouched": True,
    }
    (C35_DATA / "c3_entity_balanced_v1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] Built {out_jsonl.name}: n={len(rows)}, sha256={out_sha[:16]}, entity_count_auc={entity_count_auc:.4f}, g0_audit={g0_audit}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
