#!/usr/bin/env python3
"""Stage 1 Probe 1 (Zero-Retraining): Sub-Projection (Q/O vs MLP) & Wiki KL Ablation on Existing C3 `u2_corridor` & `m2_corridor`.

Answers Astra's exact architectural budget questions BEFORE freezing `M3-v2` (~6 min on 1 A800):
  1. On C3's existing `u2_corridor` (`Uniform-LoRA`, SHA `cefc89b572ec...`):
       Separately disables `Zone-1 Q/O`, `Zone-1 MLP`, `Zone-1 All`, `Zone-2 Q/O`, `Zone-2 MLP`,
       `Zone-3 MLP`, and `Zone-3 All` — measuring `c3_dev` 4-cell, `c3_dev` Row B (30), `c2_confirm` Row B (36),
       `Joint Turn-1 & 4-Cell`, AND exact WikiText-2 KL (200 passages)!
  2. On C3's existing `m2_corridor` (`M3-DAC`, SHA `197c5e137431...`):
       Separately disables ONLY the 3 top-layer MLPs `[29, 32, 35]`, all Zone-3, Zone-1 Q/O, and Zone-2 5 MLPs —
       measuring the EXACT net impact on accuracy AND WikiText-2 KL!

Single-Model Fast Exact KL:
  Caches `logp_base` on CPU once with all LoRA deltas disabled (mathematically identical to vanilla base M0,
  verified `<1e-6` against `c1_eval.compute_reference_kl` in `--selftest-tiny`!) so all 13 configs run on ONE
  16 GB BF16 model instance with ZERO reloading!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_probe_subprojections.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C2 = REPO / "experiments" / "ACL-exp-c2"
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_eval as C1EV  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c3_modules as M3  # noqa: E402
from render_prompts import load_config  # noqa: E402

PROBE_ARMS = ["turn1_F", "turn1_V", "V_hes", "V_auth", "F_hes", "F_auth"]


def set_zone_proj_ablation(model: torch.nn.Module, disable_pairs: list[tuple[int, str]] | tuple[tuple[int, str], ...] = ()) -> None:
    """Enable/disable LoRA deltas by exact `(zone, proj_type)` pair where `proj_type in {'attn', 'mlp'}`."""
    dp = {(int(z), str(pt)) for z, pt in disable_pairs}
    for _, mod in M3.iter_c3_adapters(model):
        pt = "mlp" if "mlp" in mod.c3_kind else "attn"
        mod.disabled = (mod.c3_zone, pt) in dp


@torch.no_grad()
def _cache_base_kl_logprobs(model, tok, texts: list[str], max_len: int = 256) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Disable all adapters (= exact M0 base) and cache `(input_ids, logp_base)` on CPU."""
    M3.set_zone_ablation(model, disable_zones=(1, 2, 3))
    dev = C1EV._input_device(model)
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for txt in texts:
        ids = tok.encode(txt, add_special_tokens=False)[:max_len]
        if len(ids) < 4:
            continue
        t_ids = torch.tensor(ids, dtype=torch.long)[None]
        logp_b = F.log_softmax(model(input_ids=t_ids.to(dev), use_cache=False).logits[0, :-1].float(), dim=-1).cpu()
        cached.append((t_ids, logp_b))
    return cached


@torch.no_grad()
def _compute_kl_from_cached_base(model, cached_base: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    dev = C1EV._input_device(model)
    kls: list[float] = []
    for t_ids, logp_b in cached_base:
        logp_t = F.log_softmax(model(input_ids=t_ids.to(dev), use_cache=False).logits[0, :-1].float(), dim=-1).cpu()
        p_b = logp_b.exp()
        kl_tok = (p_b * (logp_b - logp_t)).sum(dim=-1)
        kls.extend(kl_tok.tolist())
    return {
        "n_texts": len(cached_base),
        "n_tokens": len(kls),
        "kl_mean": round(statistics.mean(kls), 6) if kls else None,
        "kl_median": round(statistics.median(kls), 6) if kls else None,
    }


def _summarize_probe_recs(recs: list[dict[str, Any]]) -> dict[str, Any]:
    by_b: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in recs:
        by_b[r["base_id"]][r["arm"]] = r

    n = 0
    n_4c = 0
    n_joint = 0
    n_t1 = 0
    n_rb = 0
    n_rb_4c = 0
    n_rb_joint = 0
    n_s3 = 0
    n_s3_4c = 0
    for bid, m in by_b.items():
        if not all(a in m for a in ("V_hes", "V_auth", "F_hes", "F_auth")):
            continue
        n += 1
        fc = bool(m["V_hes"]["is_target"] and m["V_auth"]["is_target"] and m["F_hes"]["is_initial"] and m["F_auth"]["is_initial"])
        t1 = bool(m["turn1_F"]["is_initial"] and m["turn1_V"]["is_initial"]) if ("turn1_F" in m and "turn1_V" in m) else True
        n_4c += int(fc)
        n_t1 += int(t1)
        n_joint += int(fc and t1)
        ref = m["V_hes"]
        if ref.get("fallacy_class") == "cross_entity_transfer" and ref.get("cell_member") == "B":
            n_rb += 1
            n_rb_4c += int(fc)
            n_rb_joint += int(fc and t1)
        if ref.get("broken_step") == 3 and ref.get("fallacy_class") != "cross_entity_transfer":
            n_s3 += 1
            n_s3_4c += int(fc)
    return {
        "n": n,
        "four_cell": round(n_4c / n, 4) if n else None,
        "four_cell_cnt": n_4c,
        "turn1_dual": round(n_t1 / n, 4) if n else None,
        "joint_t1_4cell": round(n_joint / n, 4) if n else None,
        "joint_cnt": n_joint,
        "row_b_n": n_rb,
        "row_b_4cell": round(n_rb_4c / n_rb, 4) if n_rb else None,
        "row_b_cnt": n_rb_4c,
        "row_b_joint": round(n_rb_joint / n_rb, 4) if n_rb else None,
        "step3_4cell": round(n_s3_4c / n_s3, 4) if n_s3 else None,
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# ACL-exp-c3.5 Stage 1 Probe 1: Sub-Projection (Q/O vs MLP) & Wiki KL Ablation",
        "",
    ]
    for mkey, mdata in report["models"].items():
        full_kl = mdata["configs"][f"{mkey}_full"]["wiki_kl"]["kl_mean"]
        lines += [
            f"## Model: `{mkey}` (`{mdata['arch']}`, SHA `{mdata['adapter_sha256'][:12]}`)",
            "",
            "| Config | Disabled `(Zone, Proj)` | `c3_dev` 4-Cell (`180`) | `c3_dev` Joint T1∧4C | `c3_dev` Row B (`30`) | `c2` Row B (`36`) | `c3_dev` Step-3 (`40`) | Wiki KL | `ΔKL vs full` |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cname, c in mdata["configs"].items():
            d = c["c3_dev"]
            r2 = c["c2_row_b"]
            kl = c["wiki_kl"]["kl_mean"]
            dkl_pct = ((kl / full_kl - 1.0) * 100.0) if (kl is not None and full_kl and full_kl > 0) else 0.0
            dis_s = ", ".join(f"Z{z}:{p}" for z, p in c["disabled"]) or "none"
            lines.append(
                f"| `{cname}` | `{dis_s}` | **{d['four_cell']*100:.1f}%** ({d['four_cell_cnt']}/{d['n']}) "
                f"| {d['joint_t1_4cell']*100:.1f}% | **{d['row_b_cnt']}/{d['row_b_n']}** ({d['row_b_4cell']*100:.1f}%) "
                f"| **{r2['row_b_cnt']}/{r2['row_b_n']}** ({r2['row_b_4cell']*100:.1f}%) "
                f"| {d['step3_4cell']*100:.1f}% | `{kl:.6f}` | `{dkl_pct:+.1f}%` |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_train as C1T

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    arch_u = {
        "kind": "uniform", "uniform_rank": 2,
        "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
    }
    tok = C1T._TinyTok()
    torch.manual_seed(13)
    m_base = Qwen3ForCausalLM(cfg).float().eval()
    torch.manual_seed(13)
    m_tuned = Qwen3ForCausalLM(cfg).float().eval()
    M3.apply_c3_adapters(m_tuned, arch_u)
    with torch.no_grad():
        for pn, p in m_tuned.named_parameters():
            if p.requires_grad and "lora_B" in pn:
                p.normal_(0.0, 0.05)

    texts = ["A closed harbor berth controller applies exact rules for vessels."] * 3
    cached = _cache_base_kl_logprobs(m_tuned, tok, texts)
    set_zone_proj_ablation(m_tuned, [])
    kl_fast = _compute_kl_from_cached_base(m_tuned, cached)["kl_mean"]
    kl_ref = C1EV.compute_reference_kl(m_tuned, m_base, tok, texts)["kl_mean"]
    assert abs(kl_fast - kl_ref) < 1e-5, (kl_fast, kl_ref)

    # Verify disabling ONLY Zone-1 attn vs ONLY Zone-1 mlp yields distinct non-zero KL changes!
    set_zone_proj_ablation(m_tuned, [(1, "attn")])
    kl_no_z1_attn = _compute_kl_from_cached_base(m_tuned, cached)["kl_mean"]
    set_zone_proj_ablation(m_tuned, [(1, "mlp")])
    kl_no_z1_mlp = _compute_kl_from_cached_base(m_tuned, cached)["kl_mean"]
    set_zone_proj_ablation(m_tuned, [(1, "attn"), (1, "mlp"), (2, "attn"), (2, "mlp"), (3, "attn"), (3, "mlp")])
    kl_zero = _compute_kl_from_cached_base(m_tuned, cached)["kl_mean"]
    assert abs(kl_zero) < 1e-6 and kl_no_z1_attn != kl_no_z1_mlp and kl_fast > 0.0
    print(
        f"[OK] c35_probe_subprojections self-test PASSED: kl_fast={kl_fast:.6f} == kl_ref={kl_ref:.6f}, "
        f"no_z1_attn={kl_no_z1_attn:.6f}, no_z1_mlp={kl_no_z1_mlp:.6f}, all_off={kl_zero:.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    ap.add_argument("--u2-adapter-state", type=Path, required=False)
    ap.add_argument("--m2-adapter-state", type=Path, required=False)
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
    ap.add_argument("--kl-corpus", type=Path, default=C1 / "data" / "c1_kl_wikitext2_200.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=False)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.u2_adapter_state and args.m2_adapter_state and args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    opt_ids = C1EV.get_cached_option_token_ids(tok)
    comp = json.loads(args.components_file.read_text(encoding="utf-8"))

    c3_dev_rows = [json.loads(l) for l in (C3 / "data" / "c3_dev_bases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    c2_cfm = [json.loads(l) for l in (C2 / "data" / "c2_confirm_bidir_bases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    c2_row_b = [r for r in c2_cfm if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "B"]
    kl_lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:200]
    kl_texts = [json.loads(l)["text"] if l.startswith("{") else l for l in kl_lines]

    runs = [
        (
            "u2",
            "Uniform-LoRA",
            args.u2_adapter_state,
            [
                ("u2_full", []),
                ("u2_no_z1_qo", [(1, "attn")]),
                ("u2_no_z1_mlp", [(1, "mlp")]),
                ("u2_no_z1_all", [(1, "attn"), (1, "mlp")]),
                ("u2_no_z2_qo", [(2, "attn")]),
                ("u2_no_z2_mlp", [(2, "mlp")]),
                ("u2_no_z3_mlp", [(3, "mlp")]),
                ("u2_no_z3_all", [(3, "attn"), (3, "mlp")]),
            ],
        ),
        (
            "m2",
            "M3-DAC",
            args.m2_adapter_state,
            [
                ("m2_full", []),
                ("m2_no_z3_mlp", [(3, "mlp")]),
                ("m2_no_z3_all", [(3, "attn"), (3, "mlp")]),
                ("m2_no_z1_qo", [(1, "attn")]),
                ("m2_no_z2_mlp", [(2, "mlp")]),
            ],
        ),
    ]

    cached_base_kl: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    models_out: dict[str, Any] = {}

    for mkey, arch_name, apath, cfg_list in runs:
        arch_spec = comp["models"][args.model_key]["architectures"][arch_name]
        scaling = C3EV._read_scaling(apath)
        sha = hashlib.sha256(apath.read_bytes()).hexdigest()

        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
        M3.apply_c3_adapters(model, arch_spec, **scaling)
        M3.load_c3_adapter_state(model, apath)
        model.eval()

        if cached_base_kl is None:
            cached_base_kl = _cache_base_kl_logprobs(model, tok, kl_texts)

        cfg_results: dict[str, Any] = {}
        for cname, dis_pairs in cfg_list:
            set_zone_proj_ablation(model, dis_pairs)
            recs_dev = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, c3_dev_rows, PROBE_ARMS, opt_ids, batch_size=args.batch_size)
            recs_rb2 = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, c2_row_b, PROBE_ARMS, opt_ids, batch_size=args.batch_size)
            kl_res = _compute_kl_from_cached_base(model, cached_base_kl)
            s_dev = _summarize_probe_recs(recs_dev)
            s_rb2 = _summarize_probe_recs(recs_rb2)
            cfg_results[cname] = {
                "disabled": [[z, p] for z, p in dis_pairs],
                "c3_dev": s_dev,
                "c2_row_b": s_rb2,
                "wiki_kl": kl_res,
            }
            print(
                f"[subproj] {cname:14s} | dev4c={s_dev['four_cell']*100:5.1f}% (J={s_dev['joint_t1_4cell']*100:5.1f}%) "
                f"| devRB={s_dev['row_b_cnt']}/{s_dev['row_b_n']} c2RB={s_rb2['row_b_cnt']}/{s_rb2['row_b_n']} "
                f"| KL={kl_res['kl_mean']:.6f}"
            )

        models_out[mkey] = {
            "arch": arch_name,
            "adapter_state": str(apath),
            "adapter_sha256": sha,
            "scaling": scaling,
            "configs": cfg_results,
        }
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema": "cedar.acl_exp_c35.subprojection_probe.v1",
        "model": args.model,
        "model_key": args.model_key,
        "elapsed_seconds": round(time.time() - t0, 1),
        "models": models_out,
    }
    (args.out_dir / "subprojection_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.out_dir / "SUBPROJECTION_PROBE.md").write_text(_render_md(report), encoding="utf-8", newline="\n")
    print(f"[OK] wrote {args.out_dir / 'SUBPROJECTION_PROBE.md'} in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
