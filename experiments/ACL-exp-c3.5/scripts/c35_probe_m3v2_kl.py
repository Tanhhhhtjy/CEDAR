#!/usr/bin/env python3
"""Stage 3 Zero-Retraining Sub-Layer KL & Accuracy Probe on Stage 2 `armB_m3v2_corridor`.

Quantifies exact WikiText-2 KL & accuracy contributions of every remaining Zone-1 / Zone-3 sub-band inside `M3-v2`:
  - `armB_full`     : all M3-v2 adapters active (Stage 2 baseline: KL = 0.001092)
  - `armB_no_L8_9`  : disable early late-Z1 layers L8-9 (Q/O + MLP)
  - `armB_no_z1_qo` : disable all L8-13 dense Q/O (keep L8-13 MLP)
  - `armB_no_z1_mlp`: disable all L8-13 MLP (keep L8-13 Q/O)
  - `armB_no_z1_all`: disable ALL Zone-1 L8-13 (pure Zone-2+3 KL floor!)
  - `armB_no_z3`    : disable all 10 Zone-3 heads (L28-35)
  - `armB_pure_z2`  : disable ALL Zone-1 (L8-13) AND ALL Zone-3 (L28-35) -> exact Zone-2-only KL floor!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_probe_m3v2_kl.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C2 = REPO / "experiments" / "ACL-exp-c2"
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import c1_eval as C1EV  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c35_modules as M35  # noqa: E402
import c35_probe_subprojections as P1  # noqa: E402
import c35_stats as S35  # noqa: E402
from render_prompts import load_config  # noqa: E402

C3EV.M3 = M35
P1.M3 = M35

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def set_layer_proj_ablation(
    model: torch.nn.Module,
    *,
    disable_layers: set[int] = frozenset(),
    disable_layer_proj: set[tuple[int, str]] = frozenset(),
) -> None:
    """Disable LoRA adapters matching `layer_idx in disable_layers` or `(layer_idx, 'attn'|'mlp') in disable_layer_proj`."""
    for name, mod in M35.iter_c3_adapters(model):
        m = LAYER_RE.search(name)
        li = int(m.group(1)) if m else -1
        pt = "mlp" if "mlp" in mod.c3_kind else "attn"
        mod.disabled = (li in disable_layers) or ((li, pt) in disable_layer_proj)


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    arch = {
        "kind": "m3_dac",
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        "zone1_dense_qo_layers": [1],
        "zone1_dense_qo_rank": 2,
        "mlp_ranks": {"1": 4, "2": 8, "3": 4},
        "head_ranks": {"2:0": 6, "4:1": 2},
    }
    torch.manual_seed(7)
    m = Qwen3ForCausalLM(cfg).float().eval()
    M35.apply_c35_adapters(m, arch)
    set_layer_proj_ablation(m, disable_layer_proj={(1, "attn")})
    assert m.model.layers[1].self_attn.q_proj.disabled is True
    assert m.model.layers[1].mlp.gate_proj.disabled is False
    set_layer_proj_ablation(m, disable_layers={1, 4, 5})
    assert m.model.layers[1].mlp.gate_proj.disabled is True
    assert m.model.layers[4].self_attn.q_proj.disabled is True
    assert m.model.layers[2].mlp.gate_proj.disabled is False
    print("[OK] c35_probe_m3v2_kl self-test PASSED!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--adapter-state", type=Path)
    ap.add_argument("--arch", default="M3-v2")
    ap.add_argument("--model-key", choices=("qwen3_8b", "qwen3_14b"), default="qwen3_8b")
    ap.add_argument("--components-file", type=Path, default=C35 / "data" / "c35_components_frozen.json")
    ap.add_argument("--kl-corpus", type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.adapter_state and args.kl_corpus and args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    opt_ids = C1EV.get_cached_option_token_ids(tok)
    dev_rows = [json.loads(l) for l in (C3 / "data" / "c3_dev_bases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    c2_rows = [json.loads(l) for l in (C2 / "data" / "c2_confirm_bidir_bases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    c2_rowb = [r for r in c2_rows if r.get("fallacy_class") == "cross_entity_transfer" and r.get("cell_member") == "B"]
    kl_lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:200]
    kl_texts = [json.loads(l)["text"] if l.startswith("{") else l for l in kl_lines]

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    arch_spec = comp["models"][args.model_key]["architectures"][args.arch]
    scaling = C3EV._read_scaling(args.adapter_state)
    sha = hashlib.sha256(args.adapter_state.read_bytes()).hexdigest()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    M35.apply_c35_adapters(model, arch_spec, **scaling)
    M35.load_c3_adapter_state(model, args.adapter_state)
    model.eval()

    cached_base = P1._cache_base_kl_logprobs(model, tok, kl_texts, max_len=256)

    configs: list[tuple[str, str, set[int], set[tuple[int, str]]]] = [
        ("armB_full", "none", set(), set()),
        ("armB_no_L8_9", "L8-9 (Q/O+MLP)", {8, 9}, set()),
        ("armB_no_z1_qo", "L8-13 Q/O only", set(), {(li, "attn") for li in range(8, 14)}),
        ("armB_no_z1_mlp", "L8-13 MLP only", set(), {(li, "mlp") for li in range(8, 14)}),
        ("armB_no_z1_all", "All L8-13 (Z1)", set(range(8, 14)), set()),
        ("armB_no_z3", "All L28-35 (Z3)", set(range(28, 36)), set()),
        ("armB_pure_z2", "All Z1 + Z3 (Pure Z2 L14-27)", set(range(8, 14)) | set(range(28, 36)), set()),
    ]

    dev_meta = {r["base_id"]: r for r in dev_rows}
    c2_meta = {r["base_id"]: r for r in c2_rowb}
    rows_out: list[dict[str, Any]] = []
    full_kl = None

    for cname, desc, dl, dlp in configs:
        set_layer_proj_ablation(model, disable_layers=dl, disable_layer_proj=dlp)
        recs_dev = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, dev_rows, P1.PROBE_ARMS, opt_ids, batch_size=args.batch_size)
        s_dev = S35.summarize_v2(S35.compute_base_metrics_v2(recs_dev, base_meta=dev_meta))
        recs_rb = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, c2_rowb, P1.PROBE_ARMS, opt_ids, batch_size=args.batch_size)
        s_rb = S35.summarize_v2(S35.compute_base_metrics_v2(recs_rb, base_meta=c2_meta))
        kl_d = P1._compute_kl_from_cached_base(model, cached_base)
        kl = kl_d["kl_mean"]
        if full_kl is None:
            full_kl = kl
        dkl_pct = round(((kl / full_kl) - 1.0) * 100.0, 2) if (kl is not None and full_kl) else None
        vs_u2_pct = round(((kl / 0.001362) - 1.0) * 100.0, 2) if kl is not None else None
        row = {
            "config": cname,
            "disabled": desc,
            "c3_dev_4cell": s_dev["four_cell_family_weighted"],
            "c3_dev_joint": s_dev["joint_t1_4cell_unit_weighted"],
            "c3_dev_rowB": (s_dev["subgroups"].get("ce_row_B") or {}).get("four_cell"),
            "c3_dev_step3": (s_dev["subgroups"].get("step_3") or {}).get("four_cell"),
            "c2_rowB_4cell": s_rb["four_cell_item_weighted"],
            "c2_rowB_count": s_rb["four_cell_item_count"],
            "wiki_kl": kl,
            "delta_kl_vs_full_pct": dkl_pct,
            "delta_kl_vs_u2_pct": vs_u2_pct,
        }
        rows_out.append(row)
        print(
            f"[probe_m3v2] {cname:16s} | dev={row['c3_dev_4cell']*100:5.1f}% RowB_c2={row['c2_rowB_count']:2d}/36 "
            f"Step3={row['c3_dev_step3']*100:5.1f}% | KL={kl:.6f} (vs_full={dkl_pct:+.1f}%, vs_u2={vs_u2_pct:+.1f}%)"
        )

    doc = {
        "schema": "cedar.acl_exp_c35.m3v2_kl_probe.v1",
        "arch": args.arch,
        "adapter_state": str(args.adapter_state),
        "adapter_sha256": sha,
        "elapsed_seconds": round(time.time() - t0, 1),
        "rows": rows_out,
    }
    (args.out_dir / "m3v2_kl_probe.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = [
        f"# Stage 3 Zero-Retraining Sub-Layer KL Probe on `armB_m3v2_corridor` (SHA `{sha[:12]}`)",
        "",
        "| Config | Disabled | `c3_dev` 4-Cell | `c3_dev` Row B | `c2` Row B | Step-3 | Wiki KL | `ΔKL vs full` | `ΔKL vs u2` |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_out:
        lines.append(
            f"| `{r['config']}` | {r['disabled']} | **{r['c3_dev_4cell']*100:.1f}%** | {r['c3_dev_rowB']*100:.1f}% "
            f"| {r['c2_rowB_count']}/36 ({r['c2_rowB_4cell']*100:.1f}%) | {r['c3_dev_step3']*100:.1f}% "
            f"| `{r['wiki_kl']:.6f}` | {r['delta_kl_vs_full_pct']:+.1f}% | **{r['delta_kl_vs_u2_pct']:+.1f}%** |"
        )
    (args.out_dir / "M3V2_KL_PROBE.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {args.out_dir / 'M3V2_KL_PROBE.md'}")


if __name__ == "__main__":
    main()
