#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 5 Mandatory 14B Pre-Flight Gate (`Qwen3-14B` Real 1-Step Train + Export + Fixed-ID Parity).

Directly fulfills Astra's Pre-Flight Check #2 before launching full 4-arm 14B training:
  1. Verifies `qwen3_14b` frozen architectures (`Uniform-LoRA`, `M3-v4-PureMid-MLP+`, `M3-v4-PureMid-MLP+-Random-Mask`):
       - Exact Zone 2 = L16-30 (15 layers), ZERO Zone 1 (L0-15), ZERO Zone 3 (L31-39)
       - Exact matched `10,348,800` trainable params (`-1.31%` vs Uniform `10,485,760`), `max_packed_layer_rank == 78 <= 128`
       - Strictly disjoint head coordinates & non-B1 top-5 MLP layers with exact layerwise rank multiset match!
  2. Runs real 1-step training (`--max-steps 1`) on `Qwen3-14B` for `M3-v4-PureMid-MLP+` (checking finite loss,
     non-zero Zone-2 `delta_frob`, exact `0.0` Zone-1/Zone-3 `delta_frob`, zero OOM on visible GPUs).
  3. Exports PEFT safetensors via `c35_export_peft_lora.py` verifying `max_fp32_dw_abs_err < 1e-5` & `r_export == 78`.
  4. Runs fixed-token-ID HF live vs vLLM native LoRA parity via `c35_check_vllm_lora_parity.py --model-key qwen3_14b --tp <TP>`
     and HARD ASSERTS `passed == True` before writing `preflight_14b_passed.json`!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_14b_preflight.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"


def verify_frozen_14b_contract(components_file: Path) -> dict[str, Any]:
    """Hard-assert EVERY frozen 14B architectural & budget invariant."""
    comp = json.loads(components_file.read_text(encoding="utf-8"))
    m14 = comp["models"]["qwen3_14b"]
    ms = m14["model_spec"]
    assert ms["num_layers"] == 40 and ms["hidden_size"] == 5120 and ms["num_heads"] == 40, ms

    unif = m14["architectures"]["Uniform-LoRA"]
    v4 = m14["architectures"]["M3-v4-PureMid-MLP+"]
    rm1 = m14["architectures"]["M3-v4-PureMid-MLP+-Random-Mask"]

    assert unif["trainable_params"] == 10_485_760, unif["trainable_params"]
    assert v4["trainable_params"] == 10_348_800, v4["trainable_params"]
    assert rm1["trainable_params"] == v4["trainable_params"], (rm1["trainable_params"], v4["trainable_params"])
    assert v4["max_packed_layer_rank"] == 78 and rm1["max_packed_layer_rank"] == 78
    assert v4["zones"]["zone2"] == [16, 30], v4["zones"]
    assert v4["zone_trainable_params"] == {"zone1": 0, "zone2": 10_348_800, "zone3": 0}
    assert rm1["zone_trainable_params"] == {"zone1": 0, "zone2": 10_348_800, "zone3": 0}
    assert v4["mlp_layers"] == list(range(16, 31)) and rm1["mlp_layers"] == list(range(16, 31))
    assert sorted(v4["mlp_ranks"].values()) == sorted(rm1["mlp_ranks"].values()) == [8] * 10 + [12] * 5
    assert set(v4["head_ranks"]).isdisjoint(set(rm1["head_ranks"]))

    # Per-layer head rank multiset exact match!
    v4_by_l: dict[int, list[int]] = defaultdict(list)
    rm_by_l: dict[int, list[int]] = defaultdict(list)
    for k, r in v4["head_ranks"].items():
        li = int(k.split(":")[0])
        assert 16 <= li <= 30, k
        v4_by_l[li].append(int(r))
    for k, r in rm1["head_ranks"].items():
        li = int(k.split(":")[0])
        assert 16 <= li <= 30, k
        rm_by_l[li].append(int(r))
    assert set(v4_by_l) == set(rm_by_l)
    for li in sorted(v4_by_l):
        assert sorted(v4_by_l[li]) == sorted(rm_by_l[li]), (li, v4_by_l[li], rm_by_l[li])

    # Also verify 8B Mask1 & Mask2 contract simultaneously!
    m8 = comp["models"]["qwen3_8b"]
    v4_8 = m8["architectures"]["M3-v4-PureMid-MLP+"]
    rm1_8 = m8["architectures"]["M3-v4-PureMid-MLP+-Random-Mask"]
    rm2_8 = m8["architectures"]["M3-v4-PureMid-MLP+-Random-Mask2"]
    assert v4_8["trainable_params"] == rm1_8["trainable_params"] == rm2_8["trainable_params"] == 6_994_944
    assert v4_8["max_packed_layer_rank"] == rm1_8["max_packed_layer_rank"] == rm2_8["max_packed_layer_rank"] == 68
    assert set(v4_8["head_ranks"]).isdisjoint(set(rm1_8["head_ranks"]))
    assert set(v4_8["head_ranks"]).isdisjoint(set(rm2_8["head_ranks"]))
    assert set(rm1_8["head_ranks"]) != set(rm2_8["head_ranks"])

    return {
        "components_file_sha256": hashlib.sha256(components_file.read_bytes()).hexdigest(),
        "qwen3_14b_uniform_params": unif["trainable_params"],
        "qwen3_14b_v4_params": v4["trainable_params"],
        "qwen3_14b_v4_zone2": v4["zones"]["zone2"],
        "qwen3_14b_v4_max_packed_r": v4["max_packed_layer_rank"],
        "qwen3_14b_v4_top5_mlps": v4["mlp_ncf_layers"],
        "qwen3_14b_rm1_top5_mlps": rm1["mlp_top5_random_layers"],
        "qwen3_8b_v4_params": v4_8["trainable_params"],
        "qwen3_8b_rm1_top5_mlps": rm1_8["mlp_top5_random_layers"],
        "qwen3_8b_rm2_top5_mlps": rm2_8["mlp_top5_random_layers"],
    }


def _run(cmd: list[str]) -> None:
    print("[preflight] RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_selftest_tiny() -> None:
    comp_p = C35 / "data" / "c35_components_frozen.json"
    info = verify_frozen_14b_contract(comp_p)
    # Also exercise tiny synthetic M3-v4-PureMid-MLP+ export roundtrip cleanly on CPU
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(C3 / "scripts"))
    sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-c1" / "scripts"))
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c35_modules as M35
    import c35_export_peft_lora as EXP35

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    tiny_v4 = {
        "kind": "m3_dac",
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        "zone1_dense_qo_layers": [],
        "zone1_dense_qo_rank": 0,
        "mlp_ranks": {"2": 8, "3": 4},
        "head_ranks": {"2:0": 6, "2:2": 4, "3:1": 4},
    }
    m = Qwen3ForCausalLM(cfg).float()
    M35.apply_c35_adapters(m, tiny_v4)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.02)
    zn = M35.zone_delta_norms(m)
    assert zn["z1_frob"] == 0.0 and zn["z3_frob"] == 0.0 and zn["z2_frob"] > 0.0, zn
    with tempfile.TemporaryDirectory() as td:
        st_p = Path(td) / "adapter_state.pt"
        M35.save_c3_adapter_state(m, st_p)
        peft_dir = Path(td) / "peft"
        sd = torch.load(st_p, map_location="cpu", weights_only=True)
        meta = EXP35.export_c35_adapter_to_peft(
            state_dict=sd,
            arch_spec=tiny_v4,
            model_spec={"hidden_size": 64, "num_heads": 4, "head_dim": 16, "intermediate_size": 128},
            scaling={"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8},
            out_dir=peft_dir,
            base_model_name_or_path="tiny-qwen3",
            export_dtype=torch.float32,
        )
        assert meta["max_fp32_dw_abs_err"] < 1e-6, meta
    print(f"[OK] c35_14b_preflight self-test PASSED! Contract verified: {json.dumps(info, indent=2)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str, help="Path to Qwen3-14B checkpoint")
    ap.add_argument("--components-file", type=Path, default=C35 / "data" / "c35_components_frozen.json")
    ap.add_argument("--arch", type=str, default="M3-v4-PureMid-MLP+")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=20)
    ap.add_argument("--parity-limit", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=C35 / "results" / "stage5_14b" / "preflight")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model, "--model required"
    t0 = time.time()
    contract = verify_frozen_14b_contract(args.components_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "ckpt_step1"
    peft_dir = args.out_dir / "peft_step1"
    parity_json = args.out_dir / "vllm_lora_parity_14b_preflight.json"

    # Step 1: Real 1-step training on Qwen3-14B!
    _run([
        sys.executable, str(HERE / "c35_train.py"),
        "--model", args.model,
        "--model-key", "qwen3_14b",
        "--components-file", str(args.components_file),
        "--arch", args.arch,
        "--loss-mode", "corridor",
        "--iit-target", "none",
        "--max-steps", "1",
        "--microbatch", str(args.microbatch),
        "--grad-accum", str(args.grad_accum),
        "--lr", "2.5e-5",
        "--lr-z1-mult", "0.0",
        "--lambda-prox-z1", "0.0",
        "--seed", "42",
        "--out-dir", str(ckpt_dir),
    ])
    tmeta = json.loads((ckpt_dir / "train_meta.json").read_text(encoding="utf-8"))
    zn = tmeta["final_zone_delta_norms"]
    assert zn["z1_frob"] == 0.0 and zn["z3_frob"] == 0.0 and zn["z2_frob"] > 0.0, zn

    # Step 2: Export PEFT safetensors on Qwen3-14B!
    _run([
        sys.executable, str(HERE / "c35_export_peft_lora.py"),
        "--base-model", args.model,
        "--model-key", "qwen3_14b",
        "--components-file", str(args.components_file),
        "--arch", args.arch,
        "--adapter-state", str(ckpt_dir / "adapter_state.pt"),
        "--out-dir", str(peft_dir),
    ])
    emeta = json.loads((peft_dir / "c3_peft_export_meta.json").read_text(encoding="utf-8"))
    assert emeta["max_fp32_dw_abs_err"] < 1e-5 and emeta["r_export"] == 78, emeta

    # Step 3: Fixed-token-ID HF live vs vLLM native LoRA parity on Qwen3-14B!
    _run([
        sys.executable, str(HERE / "c35_check_vllm_lora_parity.py"),
        "--base-model", args.model,
        "--model-key", "qwen3_14b",
        "--components-file", str(args.components_file),
        "--arch", args.arch,
        "--adapter-state", str(ckpt_dir / "adapter_state.pt"),
        "--peft-lora-dir", str(peft_dir),
        "--dataset", "c3_dev",
        "--limit", str(args.parity_limit),
        "--tp", str(args.tp),
        "--out-json", str(parity_json),
    ])
    pmeta = json.loads(parity_json.read_text(encoding="utf-8"))
    assert pmeta["passed"] is True, pmeta

    # Step 4: Quick 4-item no_think + reference_kl VRAM smoke check on 14B (verifies model + m_base fit cleanly!)
    kl_corpus = REPO / "experiments" / "ACL-exp-c1" / "data" / "c1_kl_wikitext2_200.jsonl"
    eval_smoke_dir = args.out_dir / "eval_kl_smoke"
    _run([
        sys.executable, str(HERE / "c35_eval.py"),
        "--model", args.model,
        "--base-model", args.model,
        "--kl-corpus", str(kl_corpus),
        "--model-key", "qwen3_14b",
        "--components-file", str(args.components_file),
        "--arch", args.arch,
        "--adapter-state", str(ckpt_dir / "adapter_state.pt"),
        "--regime", "no_think",
        "--backend", "hf",
        "--datasets", "c3_dev",
        "--limit", "4",
        "--batch-size", "4",
        "--alias", "preflight_14b_smoke",
        "--out-dir", str(eval_smoke_dir),
    ])
    ev_smoke = json.loads((eval_smoke_dir / "eval_summary_no_think.json").read_text(encoding="utf-8"))
    assert ev_smoke.get("reference_kl", {}).get("kl_mean") is not None, ev_smoke

    report = {
        "schema": "cedar.acl_exp_c35.preflight_14b.v1",
        "passed": True,
        "model": args.model,
        "arch": args.arch,
        "contract": contract,
        "step1_zone_delta_norms": zn,
        "step1_log": tmeta["tail_logs"][-1] if tmeta.get("tail_logs") else None,
        "export_meta": emeta,
        "parity_passed": pmeta["passed"],
        "parity_lora": pmeta["lora_hf_live_vs_vllm_native"],
        "kl_smoke_mean": ev_smoke["reference_kl"]["kl_mean"],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_p = args.out_dir / "preflight_14b_passed.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] 14B PRE-FLIGHT GATE PASSED in {report['elapsed_seconds']}s -> {out_p}")


if __name__ == "__main__":
    main()
