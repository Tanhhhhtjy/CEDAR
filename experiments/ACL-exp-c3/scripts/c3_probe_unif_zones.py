#!/usr/bin/env python3
"""Wave 3.0 Zero-Retraining 3-Zone Ablation Probe on Existing C2 `unif_base/adapter_state.pt` (5 min on 1 GPU).

Answers Astra's exact diagnostic question BEFORE Wave 3.1 tuning:
  "On an already-trained Uniform-LoRA solution (`unif_base`), which zone — Zone 1 (L0-13), Zone 2 (L14-27),
   or Zone 3 (L28-35) — actually carries:
     (1) Legitimate cross-entity Row B binding (`c2_confirm_bidir` Row B, n=36)?
     (2) Late-buried 4-cell content separation AND reversed-tone shortcut (`c2_stealth_stress` non-threshold, n=90)?"

Runs 6 zero-retraining configurations on ONE loaded model instance (ZERO reloading overhead!):
  - `full`       : All zones active (matches C2 `unif_base`)
  - `no_zone1`   : Disable L0-13 LoRA deltas
  - `no_zone2`   : Disable L14-27 LoRA deltas
  - `no_zone3`   : Disable L28-35 LoRA deltas
  - `only_zone1` : Keep ONLY L0-13 (disable Zones 2+3)
  - `m0_base`    : Disable ALL Zones 1+2+3 (exact M0 base reference!)

Usage (1 GPU, ~4 min):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c3/scripts/c3_probe_unif_zones.py \
      --model /mnt/kcl/devuser/ckpts/Qwen3-8B \
      --adapter-state /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c2/showdown-c62cf156/c2_ckpts/42/unif_base/adapter_state.pt \
      --out-dir experiments/ACL-exp-c3/results/wave3_0_zone_probe

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_probe_unif_zones.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]
REPO = C3.parents[1]
C2 = REPO / "experiments" / "ACL-exp-c2"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-c1" / "scripts"))

import c1_eval as C1EV  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c3_modules as M3  # noqa: E402
from render_prompts import load_config  # noqa: E402

PROBE_ARMS = ["V", "F", "V_hes", "V_auth", "F_hes", "F_auth"]


def _analyze_slice(recs: list[dict[str, Any]]) -> dict[str, Any]:
    by_b: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in recs:
        by_b[r["base_id"]][r["arm"]] = r

    n = len(by_b)
    n_4cell = 0
    n_dc_orig = 0
    n_dc_rev = 0
    n_ord_plain = 0
    n_ord_hes = 0
    n_ord_auth = 0
    n_auth_neg_shift = 0
    auth_shifts: list[float] = []

    for bid, m in by_b.items():
        vh, va, fh, fa = m["V_hes"], m["V_auth"], m["F_hes"], m["F_auth"]
        vp, fp = m["V"], m["F"]
        n_4cell += int(vh["is_target"] and va["is_target"] and fh["is_initial"] and fa["is_initial"])
        n_dc_orig += int(vh["is_target"] and fa["is_initial"])
        n_dc_rev += int(va["is_target"] and fh["is_initial"])
        n_ord_plain += int(vp["u_theta"] > fp["u_theta"])
        n_ord_hes += int(vh["u_theta"] > fh["u_theta"])
        n_ord_auth += int(va["u_theta"] > fa["u_theta"])
        # Mean authoritative tone shift: 0.5 * ((u(V_auth) - u(V_hes)) + (u(F_auth) - u(F_hes)))
        shift = 0.5 * ((va["u_theta"] - vh["u_theta"]) + (fa["u_theta"] - fh["u_theta"]))
        auth_shifts.append(shift)
        if shift < 0.0:
            n_auth_neg_shift += 1

    return {
        "n": n,
        "four_cell_all_correct": round(n_4cell / n, 4) if n else None,
        "four_cell_count": n_4cell,
        "dc_vhes_fauth": round(n_dc_orig / n, 4) if n else None,
        "dc_vhes_fauth_count": n_dc_orig,
        "dc_vauth_fhes": round(n_dc_rev / n, 4) if n else None,
        "dc_vauth_fhes_count": n_dc_rev,
        "ord_plain": f"{n_ord_plain}/{n}",
        "ord_hes": f"{n_ord_hes}/{n}",
        "ord_auth": f"{n_ord_auth}/{n}",
        "mean_auth_minus_hes_u": round(sum(auth_shifts) / n, 3) if n else None,
        "auth_neg_shift_frac": round(n_auth_neg_shift / n, 4) if n else None,
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Wave 3.0 Zero-Retraining 3-Zone Ablation Probe (`unif_base`)",
        "",
        f"- **Adapter**: `{report['adapter_state']}` (SHA `{report['adapter_sha256'][:12]}`)",
        f"- **Zone Norms (Full)**: `{report['full_zone_delta_norms']}`",
        "",
        "## Slice 1: Legitimate Cross-Entity Row B (`c2_confirm_bidir`, `n=36`)",
        "",
        "| Config | 4-Cell All-Correct | `DC(V_hes,F_auth)` | `DC(V_auth,F_hes)` | Tone-Matched Ordering (`plain/hes/auth`) | Mean `u(auth)-u(hes)` |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for cfg_name, res in report["configs"].items():
        s = res["ce_row_b"]
        lines.append(
            f"| `{cfg_name}` | **{s['four_cell_count']}/{s['n']}** ({s['four_cell_all_correct']*100:.1f}%) "
            f"| {s['dc_vhes_fauth_count']}/{s['n']} | {s['dc_vauth_fhes_count']}/{s['n']} "
            f"| {s['ord_plain']} / {s['ord_hes']} / {s['ord_auth']} | {s['mean_auth_minus_hes_u']:+.2f} |"
        )
    lines += [
        "",
        "## Slice 2: Late-Buried Non-Threshold (`c2_stealth_stress`, `n=90`)",
        "",
        "| Config | 4-Cell All-Correct | `DC(V_hes,F_auth)` | `DC(V_auth,F_hes)` | Tone-Matched Ordering (`plain/hes/auth`) | Mean `u(auth)-u(hes)` | Neg-Auth Shift % |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for cfg_name, res in report["configs"].items():
        s = res["late_buried_non_thresh"]
        lines.append(
            f"| `{cfg_name}` | **{s['four_cell_count']}/{s['n']}** ({s['four_cell_all_correct']*100:.1f}%) "
            f"| {s['dc_vhes_fauth_count']}/{s['n']} | {s['dc_vauth_fhes_count']}/{s['n']} "
            f"| {s['ord_plain']} / {s['ord_hes']} / {s['ord_auth']} | {s['mean_auth_minus_hes_u']:+.2f} "
            f"| {s['auth_neg_shift_frac']*100:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import b1_lib as B1L
    import c1_train as C1T

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    arch_u = {
        "kind": "uniform",
        "uniform_rank": 2,
        "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
    }
    tok = C1T._TinyTok()
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in C3EV.OPTION_LETTERS}
    model = Qwen3ForCausalLM(cfg).float().eval()
    M3.apply_c3_adapters(model, arch_u)
    with torch.no_grad():
        for pn, p in model.named_parameters():
            if p.requires_grad and "lora_B" in pn:
                p.normal_(0.0, 0.05)

    c2_cfm = [json.loads(l) for l in (C2 / "data" / "c2_confirm_bidir_bases.jsonl").read_text(encoding="utf-8").splitlines()]
    row_b = [r for r in c2_cfm if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "B"][:2]
    c2_sth = [json.loads(l) for l in (C2 / "data" / "c2_stealth_stress_bases.jsonl").read_text(encoding="utf-8").splitlines()]
    late_nt = [r for r in c2_sth if r.get("stealth_category") == "late_buried_depth3" and r["fallacy_class"] != "late_threshold_violation"][:2]
    assert len(row_b) == 2 and len(late_nt) == 2

    orig_enc = B1L.encode_prompt
    def _tiny_enc(t, msgs, pref, enable_thinking=False):
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
        return torch.tensor(t.encode(txt), dtype=torch.long)
    B1L.encode_prompt = _tiny_enc
    try:
        for dz in ((), (1,), (2,), (3,), (1, 2, 3)):
            M3.set_zone_ablation(model, disable_zones=dz)
            recs_b = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, row_b, PROBE_ARMS, opt_ids, batch_size=2)
            recs_l = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, late_nt, PROBE_ARMS, opt_ids, batch_size=2)
            sb = _analyze_slice(recs_b)
            sl = _analyze_slice(recs_l)
            assert sb["n"] == 2 and sl["n"] == 2
    finally:
        B1L.encode_prompt = orig_enc
    print("[OK] c3_probe_unif_zones self-test PASSED!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--adapter-state", type=Path)
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.adapter_state and args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    opt_ids = C1EV.get_cached_option_token_ids(tok)
    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    arch_spec = comp["models"][args.model_key]["architectures"]["Uniform-LoRA"]
    scaling = C3EV._read_scaling(args.adapter_state)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    M3.apply_c3_adapters(model, arch_spec, **scaling)
    M3.load_c3_adapter_state(model, args.adapter_state)
    model.eval()

    full_norms = M3.zone_delta_norms(model)
    sha = hashlib.sha256(args.adapter_state.read_bytes()).hexdigest()

    c2_cfm = [json.loads(l) for l in (C2 / "data" / "c2_confirm_bidir_bases.jsonl").read_text(encoding="utf-8").splitlines()]
    row_b = [r for r in c2_cfm if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "B"]
    c2_sth = [json.loads(l) for l in (C2 / "data" / "c2_stealth_stress_bases.jsonl").read_text(encoding="utf-8").splitlines()]
    late_nt = [r for r in c2_sth if r.get("stealth_category") == "late_buried_depth3" and r["fallacy_class"] != "late_threshold_violation"]
    assert len(row_b) == 36 and len(late_nt) == 90, (len(row_b), len(late_nt))

    configs = [
        ("full", ()),
        ("no_zone1", (1,)),
        ("no_zone2", (2,)),
        ("no_zone3", (3,)),
        ("only_zone1", (2, 3)),
        ("only_zone2", (1, 3)),
        ("only_zone3", (1, 2)),
        ("m0_base", (1, 2, 3)),
    ]
    results: dict[str, Any] = {}
    for cfg_name, dz in configs:
        M3.set_zone_ablation(model, disable_zones=dz)
        recs_b = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, row_b, PROBE_ARMS, opt_ids, batch_size=args.batch_size)
        recs_l = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, late_nt, PROBE_ARMS, opt_ids, batch_size=args.batch_size)
        sb = _analyze_slice(recs_b)
        sl = _analyze_slice(recs_l)
        results[cfg_name] = {"disable_zones": list(dz), "ce_row_b": sb, "late_buried_non_thresh": sl}
        print(
            f"[probe] {cfg_name:11s} | RowB 4cell={sb['four_cell_count']}/{sb['n']} (auth_shift={sb['mean_auth_minus_hes_u']:+.2f}) "
            f"| LateNT 4cell={sl['four_cell_count']}/{sl['n']} DC={sl['dc_vhes_fauth_count']}/{sl['dc_vauth_fhes_count']} "
            f"(auth_shift={sl['mean_auth_minus_hes_u']:+.2f})"
        )

    report = {
        "schema": "cedar.acl_exp_c3.zone_probe.v1",
        "model": args.model,
        "model_key": args.model_key,
        "adapter_state": str(args.adapter_state),
        "adapter_sha256": sha,
        "scaling": scaling,
        "full_zone_delta_norms": full_norms,
        "elapsed_seconds": round(time.time() - t0, 1),
        "configs": results,
    }
    (args.out_dir / "unif_zone_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.out_dir / "UNIF_ZONE_PROBE.md").write_text(_render_md(report), encoding="utf-8", newline="\n")
    print(f"[OK] wrote {args.out_dir / 'UNIF_ZONE_PROBE.md'} in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
