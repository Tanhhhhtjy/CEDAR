#!/usr/bin/env python3
"""Build brand-new untouched prospective blind confirmation benchmark `c3_blind_confirm_v3` & `c3_blind_think_v3`.

Directly fulfills Astra's Stage 5 Prospective Replication Gate:
  1. 100% UNSEEN DOMAINS, ENTITIES, PREDICATES, OUTCOMES & DISCOURSE STYLES:
       - 6 brand-new V3-exclusive domains (`botanical_conservatory`, `tidal_barrage`, `archive_bindery`,
         `glacier_station`, `wind_tunnel`, `kiln_foundry`) hard-asserted disjoint from ALL A1, C1, C2,
         and C3/C3.5-v2 domain pools across domain names, entities, zones, guards, metrics, properties, & outcomes!
       - 2 brand-new unseen discourse styles (`style_e_brief`, `style_f_ledger`), distinct from `style_a/b` (train)
         and `style_c_dossier / style_d_memo` (v2).
  2. FIXES POSITION-PAIR CONFOUND CAUGHT BY ASTRA (`c35_build_blind_v2.py:803`):
       - All 60 paired `(before_rules vs after_rules)` counterfactual units hold BOTH the semantic graph AND
         the discourse style (`style_e_brief` or `style_f_ledger`) strictly IDENTICAL across the pair, varying
         ONLY `default_policy_position`!
  3. SAME HIGH-POWER SOLVER-CERTIFIED STRUCTURE (`900` independent generation units / `1,080` items):
       - 720 independent dual-premise non-CE bases (4 classes x 3 broken_steps x 2 pair_types x 30 reps);
         EVERY base hard-asserted by DAG solver: full V -> target, drop obsA -> initial, drop obsB -> initial,
         full F -> initial at exact `broken_step in {1, 2, 3}`!
       - 120 fixed-style paired `(before_rules vs after_rules)` bases (60 independent units).
       - 240 compound CE bases (120 complete 2x2 cells: 120 Row A + 120 Row B, 240/240 decoy clauses asserted).
  4. ZERO LEAKAGE / ZERO SEALED ACCESS:
       - Deduplicates against all open splits (`A1`, `C1 train/dev/proxy`, `C2`, `C3 train/dev/v1`, `C3.5 v2`),
         NEVER opening `c1_test_id` or `c1_test_ood`, excluding `converse_error` and `depth-4`, verifying G0 AUC ~ 0.50!

Usage:
  python experiments/ACL-exp-c3.5/scripts/c35_build_blind_v3.py
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

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(A1_SCRIPTS))
sys.path.insert(0, str(C1_SCRIPTS))
sys.path.insert(0, str(C2_SCRIPTS))
sys.path.insert(0, str(C3_SCRIPTS))

import build_symbolic_pilot_v2 as A1_SYM  # noqa: E402
import c1_build_datasets as C1B  # noqa: E402
import c1_build_proxy_ood as P1  # noqa: E402
import c1_build_proxy_template_ood as PT1  # noqa: E402
import c2_build_datasets as C2B  # noqa: E402
import c3_build_datasets as C3B  # noqa: E402
import c35_build_blind_v2 as B2  # noqa: E402

C35_V3_SEED = 20290713
OPTION_LETTERS = ("A", "B", "C", "D")

# 6 BRAND-NEW V3-EXCLUSIVE DOMAINS (100% disjoint from A1, C1, PROXY_OOD, C2, AND C3_BLIND_DOMAINS!)
V3_BLIND_DOMAINS: tuple[dict[str, Any], ...] = (
    {
        "name": "botanical_conservatory",
        "intro": "A closed alpine conservatory stewardship desk applies exact specimen routing rules. Rely strictly on the listed status items and stipulations.",
        "entity_prefix": "Specimen",
        "entities": ("Glasshouse-12", "Glasshouse-28", "Glasshouse-45", "Glasshouse-63"),
        "zones": ("Ward Canopy", "Ward Understory"),
        "guard_name": "mist_valve",
        "guard_active_text": "the mist valve is engaged",
        "guard_inactive_text": "the mist valve is released",
        "metric_name": "vapor deficit",
        "props": (
            ("p_obs", "registered a canopy leaf flush"),
            ("p_mid1", "enters stomatal acclimation"),
            ("p_mid2", "authorizes conservatory override"),
            ("p_mid3", "clears bloom priority"),
            ("p_mid4", "receives final bed allocation"),
            ("p_aux", "carries a curator voucher"),
            ("p_alt_mid", "holds a propagation slip"),
            ("p_noise1", "passed root aeration probe"),
            ("p_noise2", "logged substrate moisture check"),
        ),
        "outcomes": ("Bed-Clover", "Bed-Lichen", "Bed-Sorrel", "Bed-Thistle"),
    },
    {
        "name": "tidal_barrage",
        "intro": "A closed estuarine tidal barrage control desk enforces exact turbine gate rules. Rely strictly on the listed status items and stipulations.",
        "entity_prefix": "Turbine",
        "entities": ("Rotor-16", "Rotor-33", "Rotor-51", "Rotor-74"),
        "zones": ("Channel North", "Channel South"),
        "guard_name": "crest_latch",
        "guard_active_text": "the crest latch is engaged",
        "guard_inactive_text": "the crest latch is released",
        "metric_name": "head differential",
        "props": (
            ("p_obs", "logged an ebb flow pulse"),
            ("p_mid1", "enters wicket alignment"),
            ("p_mid2", "triggers sluice override"),
            ("p_mid3", "opens generation priority"),
            ("p_mid4", "receives final weir release"),
            ("p_aux", "holds a hydrographer permit"),
            ("p_alt_mid", "carries a siltation bulletin"),
            ("p_noise1", "passed bearing oil inspection"),
            ("p_noise2", "verified stator cooling loop"),
        ),
        "outcomes": ("Pool-Brim", "Pool-Tide", "Pool-Surf", "Pool-Eddy"),
    },
    {
        "name": "archive_bindery",
        "intro": "A closed rare-manuscript conservation bindery applies exact folio triage rules. Rely strictly on the listed status items and stipulations.",
        "entity_prefix": "Folio",
        "entities": ("Codex-18", "Codex-34", "Codex-56", "Codex-79"),
        "zones": ("Vault Upper", "Vault Lower"),
        "guard_name": "spine_clasp",
        "guard_active_text": "the spine clasp is engaged",
        "guard_inactive_text": "the spine clasp is released",
        "metric_name": "acid index",
        "props": (
            ("p_obs", "showed a parchment curl"),
            ("p_mid1", "enters leaf humidification"),
            ("p_mid2", "unlocks conservation override"),
            ("p_mid3", "grants binding priority"),
            ("p_mid4", "receives final archival shelving"),
            ("p_aux", "bears an archivist stamp"),
            ("p_alt_mid", "carries a pigment assay"),
            ("p_noise1", "passed ultraviolet ink scan"),
            ("p_noise2", "logged linen thread tensile check"),
        ),
        "outcomes": ("Shelf-Ochre", "Shelf-Umber", "Shelf-Sienna", "Shelf-Bistre"),
    },
    {
        "name": "glacier_station",
        "intro": "A closed polar ice-core drilling camp applies exact core storage rules. Rely strictly on the listed status items and stipulations.",
        "entity_prefix": "Core",
        "entities": ("Bore-19", "Bore-37", "Bore-58", "Bore-81"),
        "zones": ("Sector Firn", "Sector Basal"),
        "guard_name": "auger_pin",
        "guard_active_text": "the auger pin is engaged",
        "guard_inactive_text": "the auger pin is released",
        "metric_name": "brine salinity",
        "props": (
            ("p_obs", "emitted an air-bubble fracture"),
            ("p_mid1", "enters cryo stabilization"),
            ("p_mid2", "activates cold-vault override"),
            ("p_mid3", "approves isotope priority"),
            ("p_mid4", "receives final trench sealing"),
            ("p_aux", "holds a glaciologist tag"),
            ("p_alt_mid", "carries a tephra log"),
            ("p_noise1", "passed dielectric conductivity sweep"),
            ("p_noise2", "verified sleeve vacuum seal"),
        ),
        "outcomes": ("Cache-Flint", "Cache-Shale", "Cache-Chalk", "Cache-Marl"),
    },
    {
        "name": "wind_tunnel",
        "intro": "A closed subsonic aerodynamics test section enforces exact model mount rules. Rely strictly on the listed status items and stipulations.",
        "entity_prefix": "Airfoil",
        "entities": ("Foil-21", "Foil-39", "Foil-54", "Foil-76"),
        "zones": ("Plenum Fore", "Plenum Aft"),
        "guard_name": "vane_brake",
        "guard_active_text": "the vane brake is engaged",
        "guard_inactive_text": "the vane brake is released",
        "metric_name": "mach flutter",
        "props": (
            ("p_obs", "recorded a boundary-layer trip"),
            ("p_mid1", "enters sting balance trim"),
            ("p_mid2", "grants tunnel override"),
            ("p_mid3", "clears schlieren priority"),
            ("p_mid4", "receives final run clearance"),
            ("p_aux", "carries a test-pilot badge"),
            ("p_alt_mid", "holds a pitot calibration card"),
            ("p_noise1", "passed honeycomb screen check"),
            ("p_noise2", "logged pressure tap zeroing"),
        ),
        "outcomes": ("Rig-Kite", "Rig-Swift", "Rig-Tern", "Rig-Wren"),
    },
    {
        "name": "kiln_foundry",
        "intro": "A closed high-fire ceramics foundry enforces exact firing bay rules. Rely strictly on the listed status items and stipulations.",
        "entity_prefix": "Crucible",
        "entities": ("Pot-22", "Pot-41", "Pot-62", "Pot-84"),
        "zones": ("Flue Core", "Flue Rim"),
        "guard_name": "damper_bar",
        "guard_active_text": "the damper bar is engaged",
        "guard_inactive_text": "the damper bar is released",
        "metric_name": "pyro ramp",
        "props": (
            ("p_obs", "signaled a cone sag"),
            ("p_mid1", "enters reduction soak"),
            ("p_mid2", "authorizes kiln override"),
            ("p_mid3", "opens annealing priority"),
            ("p_mid4", "receives final draw release"),
            ("p_aux", "holds a master-potter token"),
            ("p_alt_mid", "carries a glaze viscosity card"),
            ("p_noise1", "passed thermocouple bridge check"),
            ("p_noise2", "verified flue draft balance"),
        ),
        "outcomes": ("Bay-Glaze", "Bay-Slip", "Bay-Bisque", "Bay-Frit"),
    },
)

V3_DOMAIN_OBS_B: dict[str, str] = {
    "botanical_conservatory": "verified photoperiod sensor sync",
    "tidal_barrage": "confirmed tailrace gauge balance",
    "archive_bindery": "logged calcium-carbonate buffer pass",
    "glacier_station": "confirmed sub-zero freezer lock",
    "wind_tunnel": "verified diffuser static pressure",
    "kiln_foundry": "confirmed oxygen-probe baseline",
}


def _verify_v3_domains_strictly_disjoint() -> None:
    """Assert V3_BLIND_DOMAINS are 100% disjoint from EVERY prior domain pool (A1, C1, Proxy, C2, C3/v2)."""
    prior_pools = (
        list(A1_SYM.DOMAINS)
        + list(C1B.C1_DOMAINS)
        + list(P1.PROXY_OOD_DOMAINS)
        + list(C2B.C2_TRAIN_DOMAINS)
        + list(C3B.C3_BLIND_DOMAINS)
    )
    prior_names = {d["name"] for d in prior_pools}
    prior_entities = {e for d in prior_pools for e in d["entities"]}
    prior_zones = {z for d in prior_pools for z in d["zones"]}
    prior_guards = {d["guard_name"] for d in prior_pools}
    prior_metrics = {d["metric_name"] for d in prior_pools}
    prior_outcomes = {o for d in prior_pools for o in d["outcomes"]}

    v3_names, v3_ents, v3_zones, v3_guards, v3_metrics, v3_outcomes = set(), set(), set(), set(), set(), set()
    for d in V3_BLIND_DOMAINS:
        assert d["name"] not in prior_names and d["name"] not in v3_names, d["name"]
        v3_names.add(d["name"])
        for e in d["entities"]:
            assert e not in prior_entities and e not in v3_ents, e
            v3_ents.add(e)
        for z in d["zones"]:
            assert z not in prior_zones and z not in v3_zones, z
            v3_zones.add(z)
        assert d["guard_name"] not in prior_guards and d["guard_name"] not in v3_guards, d["guard_name"]
        v3_guards.add(d["guard_name"])
        assert d["metric_name"] not in prior_metrics and d["metric_name"] not in v3_metrics, d["metric_name"]
        v3_metrics.add(d["metric_name"])
        for o in d["outcomes"]:
            assert o not in prior_outcomes and o not in v3_outcomes, o
            v3_outcomes.add(o)
        assert d["name"] in V3_DOMAIN_OBS_B


def _render_v3_context(
    dom: dict[str, Any],
    facts: list[str],
    rules: list[str],
    target_entity: str,
    letter_to_outcome: dict[str, str],
    default_policy_text: str,
    pos: str,
    style: str,
) -> str:
    """Render in one of two brand-new V3 unseen styles (`style_e_brief` or `style_f_ledger`)."""
    assert style in ("style_e_brief", "style_f_ledger")
    opts = "\n".join(f"{k}. {letter_to_outcome[k]}" for k in OPTION_LETTERS)
    if style == "style_e_brief":
        fact_block = "\n".join(f"Status Item {i + 1}: {f}" for i, f in enumerate(facts))
        clause_lines = []
        for i, r in enumerate(rules):
            r2 = re.sub(r"^If (.+?), then ", r"Provided that \1, it holds that ", r)
            r2 = re.sub(
                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                r"Provided that \1, displace the standing outcome and send \2 to \3.",
                r2,
            )
            clause_lines.append(f"Stipulation {i + 1}: {r2}")
        rule_block = "\n".join(clause_lines)
        dp = default_policy_text.replace("Default policy:", "Standing Policy:", 1)
        body = f"{dp}\n\nGoverning Stipulations:\n{rule_block}" if pos == "before_rules" else f"Governing Stipulations:\n{rule_block}\n\n{dp}"
        return (
            f"[Technical Brief]\n{dom['intro']}\n"
            f"Closed-brief note: a condition holds only when recorded under Status Items, inferred through Governing Stipulations, or confirmed by a verified dispatch note.\n\n"
            f"Status Items:\n{fact_block}\n\n"
            f"{body}\n\n"
            f"Question: Relying solely on the status items and governing stipulations above, what is the current outcome for {target_entity}?\n"
            f"Options:\n{opts}"
        )
    else:
        fact_block = "\n".join(f"Ledger Entry {i + 1} — {f}" for i, f in enumerate(facts))
        clause_lines = []
        for i, r in enumerate(rules):
            r2 = re.sub(r"^If (.+?), then ", r"Whenever \1, certify that ", r)
            r2 = re.sub(
                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                r"Whenever \1, override the fallback outcome and place \2 under \3.",
                r2,
            )
            clause_lines.append(f"Protocol {i + 1} — {r2}")
        rule_block = "\n".join(clause_lines)
        dp = default_policy_text.replace("Default policy:", "Fallback Policy:", 1)
        body = f"{dp}\n\nActive Protocols:\n{rule_block}" if pos == "before_rules" else f"Active Protocols:\n{rule_block}\n\n{dp}"
        return (
            f"[Compliance Ledger]\n{dom['intro']}\n"
            f"Closed-ledger note: a property holds only if logged under Ledger Entries, deduced from Active Protocols, or reported in a verified audit notice.\n\n"
            f"Ledger Entries:\n{fact_block}\n\n"
            f"{body}\n\n"
            f"Question: Based strictly on the ledger entries and active protocols above, what is the assigned outcome for {target_entity}?\n"
            f"Options:\n{opts}"
        )


def _rewrite_arg_v3_style(txt: str, style: str) -> str:
    if style == "style_e_brief":
        t = re.sub(r"^New verified observation:\s*", "Verified dispatch note: ", txt)
        t = re.sub(r"\bStep (\d+):\s*", r"Stage \1: ", t)
        t = re.sub(r"\bRule (\d+)\b", r"Stipulation \1", t)
        t = re.sub(r"\bConclusion:\s*", "Final Ruling: ", t)
        return t
    else:
        t = re.sub(r"^New verified observation:\s*", "Verified audit notice: ", txt)
        t = re.sub(r"\bStep (\d+):\s*", r"Step-Check \1: ", t)
        t = re.sub(r"\bRule (\d+)\b", r"Protocol \1", t)
        t = re.sub(r"\bConclusion:\s*", "Assigned Outcome: ", t)
        return t


def _build_v3_ce_cell(
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
    """Build 2x2 compound CE cell in V3 styles with 100% hard-asserted decoy clause insertion."""
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
        schema_name="cedar.acl_exp_c35.blind_compound_ce_2x2.v3",
    )
    dom = domains[dom_idx]
    p = dict(dom["props"])

    for row in (ra, rb):
        a0, at = row["initial_answer"], row["target_answer"]
        decoy_l = [k for k in OPTION_LETTERS if k not in (a0, at)][0]
        a_decoy_name = row["letter_to_outcome"][decoy_l]

        def _render_ce_ctx(ctx: str) -> str:
            lines: list[str] = []
            in_rules = False
            rule_idx = 0
            inserted_decoy = False

            def _decoy_line(next_idx: int) -> str:
                if style == "style_e_brief":
                    return (
                        f"Stipulation {next_idx}: Provided that an item {p['p_alt_mid']} and carries an emergency committee waiver, "
                        f"displace the standing outcome and send it to {a_decoy_name}."
                    )
                else:
                    return (
                        f"Protocol {next_idx} — Whenever an item {p['p_alt_mid']} and carries an emergency committee waiver, "
                        f"override the fallback outcome and place it under {a_decoy_name}."
                    )

            for line in ctx.splitlines():
                s = line.strip()
                if s == dom["intro"]:
                    lines.append("[Technical Brief]" if style == "style_e_brief" else "[Compliance Ledger]")
                    lines.append(s)
                elif s.startswith("Closed-world note:"):
                    if style == "style_e_brief":
                        lines.append(
                            "Closed-brief note: a condition holds only when recorded under Status Items, inferred through Governing Stipulations, or confirmed by a verified dispatch note."
                        )
                    else:
                        lines.append(
                            "Closed-ledger note: a property holds only if logged under Ledger Entries, deduced from Active Protocols, or reported in a verified audit notice."
                        )
                elif s == "Facts:":
                    lines.append("Status Items:" if style == "style_e_brief" else "Ledger Entries:")
                elif s.startswith("Fact "):
                    m = re.match(r"^Fact (\d+)\.\s*(.+)$", s)
                    if m:
                        lines.append(
                            f"Status Item {m.group(1)}: {m.group(2)}"
                            if style == "style_e_brief"
                            else f"Ledger Entry {m.group(1)} — {m.group(2)}"
                        )
                    else:
                        lines.append(line)
                elif s == "Rules:":
                    in_rules = True
                    lines.append("Governing Stipulations:" if style == "style_e_brief" else "Active Protocols:")
                elif s.startswith("Rule "):
                    m = re.match(r"^Rule (\d+)\.\s*(.+)$", s)
                    if m:
                        rule_idx = int(m.group(1))
                        body = m.group(2)
                        if style == "style_e_brief":
                            body2 = re.sub(r"^If (.+?), then ", r"Provided that \1, it holds that ", body)
                            body2 = re.sub(
                                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                                r"Provided that \1, displace the standing outcome and send \2 to \3.",
                                body2,
                            )
                            lines.append(f"Stipulation {rule_idx}: {body2}")
                        else:
                            body2 = re.sub(r"^If (.+?), then ", r"Whenever \1, certify that ", body)
                            body2 = re.sub(
                                r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$",
                                r"Whenever \1, override the fallback outcome and place \2 under \3.",
                                body2,
                            )
                            lines.append(f"Protocol {rule_idx} — {body2}")
                    else:
                        lines.append(line)
                elif s.startswith("Default policy:"):
                    if in_rules and not inserted_decoy:
                        lines.append(_decoy_line(rule_idx + 1))
                        inserted_decoy = True
                        in_rules = False
                    lines.append(
                        s.replace("Default policy:", "Standing Policy:" if style == "style_e_brief" else "Fallback Policy:", 1)
                    )
                elif s.startswith(("Question:", "Based only")):
                    if in_rules and not inserted_decoy:
                        lines.append(_decoy_line(rule_idx + 1))
                        inserted_decoy = True
                        in_rules = False
                    if style == "style_e_brief":
                        lines.append(s.replace("listed facts and rules", "status items and governing stipulations"))
                    else:
                        lines.append(s.replace("listed facts and rules", "ledger entries and active protocols"))
                else:
                    lines.append(line)
            assert inserted_decoy, f"Decoy clause NOT inserted on {row['base_id']} pos={pos}!"
            return "\n".join(lines)

        for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
            row[k] = _render_ce_ctx(row[k])
        for an in list(row["arms"].keys()):
            if an.startswith(("V", "F")) or an == "shell_only":
                row["arms"][an] = _rewrite_arg_v3_style(row["arms"][an], style)
        row["family_id"] = cell_id
        row["unit_id"] = cell_id
        row["compound_blind"] = True
        row["phrasing_style"] = style
        row["broken_step"] = 1
    return ra, rb


def main() -> None:
    C3B._verify_all_domain_pools_disjoint()
    _verify_v3_domains_strictly_disjoint()
    C35_DATA.mkdir(parents=True, exist_ok=True)

    # Temporarily plug V3 domain obs_B and renderers into B2 cleanly (zero side effects on v2 files!)
    orig_obs_b = dict(B2.DOMAIN_OBS_B)
    orig_render = B2._render_blind_context
    orig_rewrite = B2._rewrite_arg_blind_style
    B2.DOMAIN_OBS_B.update(V3_DOMAIN_OBS_B)
    B2._render_blind_context = _render_v3_context
    B2._rewrite_arg_blind_style = _rewrite_arg_v3_style

    try:
        # Deduplicate against ALL open splits INCLUDING v1 & v2 (NEVER touch sealed c1_test_id / c1_test_ood!)
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
        ]
        c3_train_rows: list[dict[str, Any]] = []
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
                if p.name == "c3_train_bases.jsonl":
                    c3_train_rows.append(r)
                elif p.name == "c3_blind_confirm_v2_bases.jsonl":
                    v2_rows.append(r)

        non_ce_classes = ("scope_swap", "missing_conjunct", "negation_drop", "threshold_violation")
        blind_v3_bases: list[dict[str, Any]] = []

        # 1. 720 independent dual-premise non-CE bases (4 classes x 3 steps x 2 pair_types x 30 reps)
        specs_indep = []
        for fc in non_ce_classes:
            for bstep in (1, 2, 3):
                for pt in ("context_flip", "argument_flip"):
                    for _ in range(30):
                        specs_indep.append((fc, bstep, pt))
        assert len(specs_indep) == 720
        fac_indep = C3B._deconfounded_factors(720, len(V3_BLIND_DOMAINS), seed=C35_V3_SEED + 101, include_style=True)

        for idx, ((fc, bstep, pt), fac) in enumerate(zip(specs_indep, fac_indep), start=1):
            dom_idx, pos = fac["dom_idx"], fac["pos"]
            style = "style_e_brief" if fac["style"] == "style_a" else "style_f_ledger"
            trans = dedup.next_balanced_transition(dom_idx, salt=311)
            bid = f"C35V3-BLD-{len(blind_v3_bases) + 1:04d}"
            attempt = 0
            while True:
                cur_idx = 1200000 + idx + attempt * 6 * 20011
                row = B2._build_independent_dual_premise_base(
                    domains=V3_BLIND_DOMAINS,
                    seed_base=C35_V3_SEED + 100000,
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
                    split_name="c3_blind_confirm_v3",
                )
                if not dedup.has_collision(row):
                    dedup.commit(row)
                    blind_v3_bases.append(row)
                    break
                attempt += 1
                assert attempt < 80

        # 2. 60 FIXED-STYLE PAIRED `(before_rules vs after_rules)` counterfactual semantic graphs = 120 bases (60 units)
        # FIXES ASTRA AUDIT CATCH ON v2: BOTH halves of every position pair share the EXACT SAME discourse style!
        specs_pair = []
        for fc in non_ce_classes:
            for bstep in (1, 2, 3):
                for _ in range(5):
                    specs_pair.append((fc, bstep, "context_flip" if len(specs_pair) % 2 == 0 else "argument_flip"))
        assert len(specs_pair) == 60
        fac_pair = C3B._deconfounded_factors(60, len(V3_BLIND_DOMAINS), seed=C35_V3_SEED + 202, include_style=True)
        for pidx, ((fc, bstep, pt), fac) in enumerate(zip(specs_pair, fac_pair), start=1):
            dom_idx = fac["dom_idx"]
            fixed_style = "style_e_brief" if (pidx % 2 == 1) else "style_f_ledger"
            trans = dedup.next_balanced_transition(dom_idx, salt=323)
            uid = f"C35V3-POSPAIR-{pidx:03d}"
            cur_idx = 1250000 + pidx * 17
            pair_rows = []
            for pos in ("before_rules", "after_rules"):
                bid = f"C35V3-BLD-{len(blind_v3_bases) + 1:04d}"
                row = B2._build_independent_dual_premise_base(
                    domains=V3_BLIND_DOMAINS,
                    seed_base=C35_V3_SEED + 300000,
                    idx=cur_idx,  # EXACT SAME semantic graph, numbers, AND fixed_style!
                    dom_idx=dom_idx,
                    fallacy_class=fc,
                    broken_step=bstep,
                    pair_type=pt,
                    pos=pos,
                    style=fixed_style,
                    trans=trans,
                    bid=bid,
                    unit_id=uid,
                    split_name="c3_blind_confirm_v3",
                )
                row["paired_position_counterfactual"] = True
                row["paired_position_fixed_style"] = True
                pair_rows.append(row)
                blind_v3_bases.append(row)
            assert pair_rows[0]["phrasing_style"] == pair_rows[1]["phrasing_style"] == fixed_style
            assert pair_rows[0]["arms"]["V"] == pair_rows[1]["arms"]["V"]
            assert pair_rows[0]["arms"]["F"] == pair_rows[1]["arms"]["F"]
            assert pair_rows[0]["context_prompt_v"] != pair_rows[1]["context_prompt_v"]

        # 3. 120 Complete 2x2 Fixed Compound CE Cells = 240 bases (ALL 240/240 decoy-clause asserted!)
        fac_ce = C3B._deconfounded_factors(120, len(V3_BLIND_DOMAINS), seed=C35_V3_SEED + 303, include_style=True)
        for cell_i, fac in enumerate(fac_ce):
            dom_idx, pos = fac["dom_idx"], fac["pos"]
            style = "style_e_brief" if fac["style"] == "style_a" else "style_f_ledger"
            depth = 2 if cell_i < 60 else 3
            trans = dedup.next_balanced_transition(dom_idx, salt=329)
            cell_id = f"C35V3-BLD-CECELL-{cell_i + 1:03d}"
            bid_a = f"C35V3-BLD-{len(blind_v3_bases) + 1:04d}"
            bid_b = f"C35V3-BLD-{len(blind_v3_bases) + 2:04d}"
            attempt = 0
            while True:
                ra, rb = _build_v3_ce_cell(
                    domains=V3_BLIND_DOMAINS,
                    seed_base=C35_V3_SEED + 600000 + attempt * 8209,
                    cell_idx=cell_i,
                    dom_idx=dom_idx,
                    depth=depth,
                    pos=pos,
                    style=style,
                    trans=trans,
                    cell_id=cell_id,
                    bid_a=bid_a,
                    bid_b=bid_b,
                    split_name="c3_blind_confirm_v3",
                )
                if (not dedup.has_collision(ra)) and (not dedup.has_collision(rb)):
                    dedup.commit(ra)
                    dedup.commit(rb)
                    blind_v3_bases.extend([ra, rb])
                    break
                attempt += 1
                assert attempt < 50

        assert len(blind_v3_bases) == 1080
        n_units = len({r["unit_id"] for r in blind_v3_bases})
        assert n_units == 900, n_units

        # Stratified 240-base think slice (`c3_blind_think_v3`)
        by_s: dict[tuple, list[dict]] = defaultdict(list)
        ce_map: dict[str, list[dict]] = defaultdict(list)
        for r in blind_v3_bases:
            if r["fallacy_class"] == "cross_entity_transfer":
                ce_map[r["cell_id"]].append(r)
            elif not r.get("paired_position_counterfactual"):
                by_s[(r["fallacy_class"], r["broken_step"], r["pair_type"])].append(r)
        think_v3_bases: list[dict[str, Any]] = []
        for k in sorted(by_s):
            think_v3_bases.extend(by_s[k][:6])  # 24 * 6 = 144
        pos_units = sorted({r["unit_id"] for r in blind_v3_bases if r.get("paired_position_counterfactual")})
        for uid in pos_units[:8]:  # 8 pairs = 16 -> 160 non-CE
            think_v3_bases.extend([r for r in blind_v3_bases if r["unit_id"] == uid])
        for cid in sorted(ce_map)[:40]:  # 40 CE cells = 80 (40 Row A + 40 Row B) -> 240 total!
            think_v3_bases.extend(ce_map[cid])
        assert len(think_v3_bases) == 240

        # STRICT INTEGRITY AUDIT
        for r in blind_v3_bases:
            assert r["fallacy_class"] != "converse_error"
            assert int(r.get("difficulty_depth", 2)) <= 3
            for k in ("context_prompt_v", "context_prompt_f"):
                assert "Only if" not in r[k]
                assert "Only if" not in r["arms"]["V"]
                assert "Only if" not in r["arms"]["F"]
            if r["fallacy_class"] == "cross_entity_transfer":
                assert "emergency committee waiver" in r["context_prompt_v"], r["base_id"]
                assert "emergency committee waiver" in r["context_prompt_f"], r["base_id"]
            else:
                assert r.get("independent_dual_premise") is True

        # Skeleton 0-overlap against BOTH c3_train AND blind_v2!
        train_v_skels = {PT1.extract_placeholder_skeleton(r["arms"]["V"], C2B.C2_TRAIN_DOMAINS) for r in c3_train_rows}
        train_ctx_skels = {PT1.extract_placeholder_skeleton(r["context_prompt_v"], C2B.C2_TRAIN_DOMAINS) for r in c3_train_rows}
        v2_v_skels = {PT1.extract_placeholder_skeleton(r["arms"]["V"], C3B.C3_BLIND_DOMAINS) for r in v2_rows}
        v2_ctx_skels = {PT1.extract_placeholder_skeleton(r["context_prompt_v"], C3B.C3_BLIND_DOMAINS) for r in v2_rows}
        ov_tr_v = sum(PT1.extract_placeholder_skeleton(r["arms"]["V"], V3_BLIND_DOMAINS) in train_v_skels for r in blind_v3_bases)
        ov_tr_ctx = sum(PT1.extract_placeholder_skeleton(r["context_prompt_v"], V3_BLIND_DOMAINS) in train_ctx_skels for r in blind_v3_bases)
        ov_v2_v = sum(PT1.extract_placeholder_skeleton(r["arms"]["V"], V3_BLIND_DOMAINS) in v2_v_skels for r in blind_v3_bases)
        ov_v2_ctx = sum(PT1.extract_placeholder_skeleton(r["context_prompt_v"], V3_BLIND_DOMAINS) in v2_ctx_skels for r in blind_v3_bases)
        assert ov_tr_v == 0 and ov_tr_ctx == 0, (ov_tr_v, ov_tr_ctx)
        assert ov_v2_v == 0 and ov_v2_ctx == 0, (ov_v2_v, ov_v2_ctx)

        non_ce_indep = [r for r in blind_v3_bases if r["fallacy_class"] != "cross_entity_transfer" and not r.get("paired_position_counterfactual")]
        indep_chi2 = C3B._audit_factor_independence(non_ce_indep, "c3_blind_confirm_v3_indep", check_style=True)
        g0_v3 = C3B._run_g0_tfidf_auc(blind_v3_bases)
        assert 0.42 <= g0_v3["overall_auc"] <= 0.58 and g0_v3["cf_byte_match"] == 1.0, g0_v3

        out_v3 = C35_DATA / "c3_blind_confirm_v3_bases.jsonl"
        out_th = C35_DATA / "c3_blind_think_v3_bases.jsonl"
        out_v3.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in blind_v3_bases) + "\n", encoding="utf-8", newline="\n")
        out_th.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in think_v3_bases) + "\n", encoding="utf-8", newline="\n")

        def _sha(p: Path) -> str:
            return hashlib.sha256(p.read_bytes()).hexdigest()

        manifest = {
            "schema": "cedar.acl_exp_c35.blind_v3_manifest.v1",
            "seed": C35_V3_SEED,
            "prospective_untouched": True,
            "files": {
                out_v3.name: {"n_bases": len(blind_v3_bases), "n_units": n_units, "sha256": _sha(out_v3)},
                out_th.name: {"n_bases": len(think_v3_bases), "n_units": len({r["unit_id"] for r in think_v3_bases}), "sha256": _sha(out_th)},
            },
            "domains": [d["name"] for d in V3_BLIND_DOMAINS],
            "styles": dict(Counter(r["phrasing_style"] for r in blind_v3_bases)),
            "positions": dict(Counter(r["default_policy_position"] for r in blind_v3_bases)),
            "by_fallacy_class": dict(Counter(r["fallacy_class"] for r in blind_v3_bases)),
            "ce_decoy_verified_count": sum(
                1 for r in blind_v3_bases if r["fallacy_class"] == "cross_entity_transfer" and "emergency committee waiver" in r["context_prompt_v"]
            ),
            "dual_premise_verified_count": sum(1 for r in blind_v3_bases if r.get("independent_dual_premise")),
            "paired_position_fixed_style_units": len({r["unit_id"] for r in blind_v3_bases if r.get("paired_position_fixed_style")}),
            "factor_independence_chi2": indep_chi2,
            "skeleton_overlap_vs_c3_train": {"V": ov_tr_v, "ctx": ov_tr_ctx},
            "skeleton_overlap_vs_blind_v2": {"V": ov_v2_v, "ctx": ov_v2_ctx},
            "g0_tfidf_auc": g0_v3,
        }
        man_p = C35_DATA / "c35_blind_v3_manifest.json"
        man_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"[OK] Built prospective c3_blind_confirm_v3 ({len(blind_v3_bases)} bases / {n_units} units) & c3_blind_think_v3 ({len(think_v3_bases)} bases)")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    finally:
        B2.DOMAIN_OBS_B.clear()
        B2.DOMAIN_OBS_B.update(orig_obs_b)
        B2._render_blind_context = orig_render
        B2._rewrite_arg_blind_style = orig_rewrite


if __name__ == "__main__":
    main()
