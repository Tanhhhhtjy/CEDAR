#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 6 Baseline: CEDAR Same-Data Joint DPO (`Uniform-LoRA`).

Implements standard textbook Rafailov et al. (NeurIPS 2023) DPO on the EXACT same 6,000 training examples
(`ACL-exp-c3/data/c3_train_examples_nothink.jsonl`) and exact same `Uniform-LoRA` budget (`7,077,888` params on 8B),
atomic windowing (`grad_accum=20 -> 300` optimizer steps, 1 epoch), and `c3_dev` checkpoint selection as `u1_base` /
`u2_corridor`, transparently documenting every adaptation detail required by Astra's Stage 5 review:
  - Label: `CEDAR same-data Joint DPO adaptation (Uniform-LoRA)` (distinct from Ma et al.'s self-rollout DPO)
  - Preference pairs constructed directly from all 6,000 training examples:
      * Update pairs (`V, V_auth, V_hes`): `chosen = "Final: <target>"`, `rejected = "Final: <initial>"` (`1,800` pairs)
      * Stay pairs (`F, F_auth, F_hes, shell_only, AO, S, F_src_third_party, P0_confirm_a0, I0_confirm_a0, N`):
        `chosen = "Final: <initial>"`, `rejected = "Final: <target>"` (`4,200` pairs)
      * Symmetric core weighting matching `c3_train.py` (`w=1.0` on 6 core arms: 1,800 update + 1,800 stay; `w=0.5` on aux arms)
  - Reference model: frozen `M0` (precomputed exact full-completion sequence log-probs before step 1 — zero extra VRAM!)
  - Fair dev search over `beta in {0.05, 0.1, 0.2}` (each run saves checkpoints at steps `120, 200, 250, 300`).
  - Saves standard `adapter_state.pt` / `adapter_config.json` 100% compatible with `c35_dev_select.py`, `c35_export_peft_lora.py`,
    `c35_verify_vllm_parity.py`, `c35_eval.py`, `c1_eval_capabilities.py`, and `c35_eval_exfever.py`!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_train_dpo.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(C1 / "scripts"))

import c1_train as C1T  # noqa: E402
import c3_train as C3T  # noqa: E402
import c35_modules as M35  # noqa: E402

CORE_ARMS = set(C3T.CORE_ARMS)
UPDATE_ARMS = {"V", "V_auth", "V_hes", "V_src_third_party"}


def encode_dpo_pair(tok, row: dict[str, Any]) -> dict[str, Any]:
    """Encode `(prompt + chosen_completion, prompt + rejected_completion)` with exact completion token masks."""
    a0 = str(row["initial_answer"])
    at = str(row["target_answer"])
    gold = str(row["gold_answer"])
    arm = str(row["arm"])
    assert gold in (a0, at), (row.get("example_id"), gold, a0, at)
    rejected_letter = a0 if gold == at else at
    pair_direction = "update" if gold == at else "stay"
    if arm in UPDATE_ARMS:
        assert pair_direction == "update", (arm, gold, at)

    chosen_row = dict(row)
    chosen_row["gold_answer"] = gold
    chosen_row["assistant_target"] = f"Final: {gold}"
    rej_row = dict(row)
    rej_row["gold_answer"] = rejected_letter
    rej_row["assistant_target"] = f"Final: {rejected_letter}"

    enc_w = C1T.encode_sft_example(tok, chosen_row, enable_thinking=False, mask_rule_format_ce=False)
    enc_l = C1T.encode_sft_example(tok, rej_row, enable_thinking=False, mask_rule_format_ce=False)

    return {
        "example_id": row["example_id"],
        "pair_group_id": row["pair_group_id"],
        "base_id": row["base_id"],
        "arm": arm,
        "pair_direction": pair_direction,
        "is_core": arm in CORE_ARMS,
        "chosen_ids": enc_w["input_ids"],
        "chosen_labels": enc_w["labels"],
        "rejected_ids": enc_l["input_ids"],
        "rejected_labels": enc_l["labels"],
    }


def sequence_logprobs_batch(
    model,
    ids_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    pad_token_id: int,
) -> torch.Tensor:
    """Compute exact summed completion log-probabilities `sum_{t: label_t != -100} log p(y_t | x, y_{<t})`."""
    dev = C1T._input_device(model)
    bsz = len(ids_list)
    max_len = max(int(t.numel()) for t in ids_list)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
        L = int(ids.numel())
        input_ids[i, :L] = ids
        labels[i, :L] = lab
        attn_mask[i, :L] = 1

    logits = model(input_ids=input_ids.to(dev), attention_mask=attn_mask.to(dev), use_cache=False).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].to(logits.device)
    valid_mask = shift_labels != -100
    safe_labels = shift_labels.clamp_min(0)
    log_p = F.log_softmax(shift_logits, dim=-1)
    tok_lp = log_p.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_lp = (tok_lp * valid_mask.to(tok_lp.dtype)).sum(dim=-1)
    return seq_lp


@torch.no_grad()
def precompute_reference_logprobs(
    model,
    pairs: list[dict[str, Any]],
    pad_token_id: int,
    batch_size: int = 8,
) -> None:
    """Precompute exact frozen M0 `(ref_logp_w, ref_logp_l)` once before step 1 (zero extra GPU copy)."""
    was_training = model.training
    model.eval()
    try:
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            lp_w = sequence_logprobs_batch(
                model,
                [p["chosen_ids"] for p in chunk],
                [p["chosen_labels"] for p in chunk],
                pad_token_id,
            )
            lp_l = sequence_logprobs_batch(
                model,
                [p["rejected_ids"] for p in chunk],
                [p["rejected_labels"] for p in chunk],
                pad_token_id,
            )
            for p, wv, lv in zip(chunk, lp_w.tolist(), lp_l.tolist()):
                p["ref_logp_w"] = float(wv)
                p["ref_logp_l"] = float(lv)
    finally:
        if was_training:
            model.train()


def compute_dpo_microbatch(
    model,
    batch: list[dict[str, Any]],
    pad_token_id: int,
    *,
    beta: float,
    aux_weight: float,
    sft_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact textbook Rafailov et al. DPO loss with transparent core/aux weighting and reward-margin telemetry."""
    all_ids = [p["chosen_ids"] for p in batch] + [p["rejected_ids"] for p in batch]
    all_labs = [p["chosen_labels"] for p in batch] + [p["rejected_labels"] for p in batch]
    all_lp = sequence_logprobs_batch(model, all_ids, all_labs, pad_token_id)
    n = len(batch)
    pi_w = all_lp[:n]
    pi_l = all_lp[n:]

    ref_w = torch.tensor([p["ref_logp_w"] for p in batch], device=all_lp.device, dtype=torch.float32)
    ref_l = torch.tensor([p["ref_logp_l"] for p in batch], device=all_lp.device, dtype=torch.float32)
    weights = torch.tensor([1.0 if p["is_core"] else aux_weight for p in batch], device=all_lp.device, dtype=torch.float32)

    log_ratio_w = pi_w - ref_w
    log_ratio_l = pi_l - ref_l
    margin = beta * (log_ratio_w - log_ratio_l)
    per_ex_dpo = -F.logsigmoid(margin)
    if sft_weight > 0.0:
        # Optional chosen-NLL anchor (default 0.0 = pure DPO)
        chosen_tok_counts = torch.tensor(
            [max(1, int((p["chosen_labels"][1:] != -100).sum())) for p in batch],
            device=all_lp.device,
            dtype=torch.float32,
        )
        per_ex_loss = per_ex_dpo + sft_weight * (-pi_w / chosen_tok_counts)
    else:
        per_ex_loss = per_ex_dpo

    loss_sum = (per_ex_loss * weights).sum()
    stats = {
        "dpo_sum": float((per_ex_dpo.detach() * weights).sum()),
        "margin_sum": float(margin.detach().sum()),
        "acc_sum": float((margin.detach() > 0.0).float().sum()),
        "logp_w_sum": float(pi_w.detach().sum()),
        "logp_l_sum": float(pi_l.detach().sum()),
        "n": n,
    }
    return loss_sum, stats


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=2048, tie_word_embeddings=False,
    )
    tok = C1T._TinyTok()
    train_path = C3 / "data" / "c3_train_examples_nothink.jsonl"
    raw_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()[:20]]
    pairs = [encode_dpo_pair(tok, r) for r in raw_rows]
    dirs = Counter(p["pair_direction"] for p in pairs)
    assert dirs["update"] > 0 and dirs["stay"] > 0, dirs

    torch.manual_seed(11)
    model = Qwen3ForCausalLM(cfg).float()
    tiny_unif = {
        "kind": "uniform",
        "uniform_rank": 2,
        "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
        "zones": {"zone1": [0, 0], "zone2": [1, 2], "zone3": [3, 3]},
    }
    M35.apply_c35_adapters(model, tiny_unif)
    precompute_reference_logprobs(model, pairs, pad_token_id=tok.pad_token_id, batch_size=4)

    # At initialization (lora_B=0), pi == ref so DPO margin MUST equal 0.0 and loss MUST equal ln(2) * sum(weights)!
    beta = 0.1
    loss_init, st_init = compute_dpo_microbatch(model, pairs[:4], tok.pad_token_id, beta=beta, aux_weight=0.5)
    expected_w_sum = sum(1.0 if p["is_core"] else 0.5 for p in pairs[:4])
    expected_loss = math.log(2.0) * expected_w_sum
    assert abs(float(loss_init.detach()) - expected_loss) < 1e-5, (float(loss_init.detach()), expected_loss)
    assert abs(st_init["margin_sum"]) < 1e-5, st_init

    # Perturb lora_B and verify exact analytical DPO loss & finite nonzero gradients
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.05)
    loss_pert, _ = compute_dpo_microbatch(model, pairs[:4], tok.pad_token_id, beta=beta, aux_weight=0.5)
    loss_pert.backward()
    grad_norm = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.requires_grad and p.grad is not None) ** 0.5
    assert grad_norm > 1e-4 and math.isfinite(grad_norm), grad_norm

    with tempfile.TemporaryDirectory() as td:
        p_out = Path(td) / "adapter_state.pt"
        M35.save_c3_adapter_state(model, p_out)
        assert p_out.is_file()
    print(f"[OK] c35_train_dpo self-test PASSED: dirs={dict(dirs)}, init_loss_err={abs(float(loss_init.detach())-expected_loss):.2e}, grad_norm={grad_norm:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=("qwen3_8b", "qwen3_14b"), default="qwen3_8b")
    ap.add_argument("--arch", type=str, default="Uniform-LoRA")
    ap.add_argument("--components-file", type=Path, default=C35 / "data" / "c35_stage6_components_frozen.json")
    ap.add_argument("--train-file", type=Path, default=C3 / "data" / "c3_train_examples_nothink.jsonl")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.0, help="0.0 = pure textbook DPO")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=20)
    ap.add_argument("--microbatch", type=int, default=2)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--scaling-mode", choices=("rslora", "constant"), default="rslora")
    ap.add_argument("--lora-scale", type=float, default=2.0)
    ap.add_argument("--anchor-rank", type=int, default=8)
    ap.add_argument("--save-steps", type=str, default="120,200,250,300")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir, "--model and --out-dir are required"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    arch_spec = comp["models"][args.model_key]["architectures"][args.arch]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    raw_rows = [json.loads(l) for l in args.train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    pairs = [encode_dpo_pair(tok, r) for r in raw_rows]
    dir_counts = Counter(p["pair_direction"] for p in pairs)
    core_dir_counts = Counter(p["pair_direction"] for p in pairs if p["is_core"])

    print(
        f"[c35_train_dpo] model={args.model} arch={args.arch} beta={args.beta} sft_w={args.sft_weight} "
        f"pairs={len(pairs)} (update={dir_counts['update']}, stay={dir_counts['stay']}) lr={args.lr} seed={args.seed}"
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    adapter_info = M35.apply_c35_adapters(
        model, arch_spec,
        scaling_mode=args.scaling_mode, base_scale=args.lora_scale, anchor_rank=args.anchor_rank, dropout=0.0,
    )

    # Precompute exact frozen reference log-probs BEFORE any optimizer step (lora_B == 0 -> exact M0)
    t_ref = time.time()
    precompute_reference_logprobs(model, pairs, pad_token_id=pad_id, batch_size=max(4, args.microbatch * 2))
    print(f"[c35_train_dpo] precomputed exact M0 reference logprobs for {len(pairs)} pairs in {time.time() - t_ref:.1f}s")

    param_groups = M35.build_zone_optimizer_groups(model, lr=args.lr, lr_z1_mult=1.0, weight_decay=args.weight_decay)
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    save_steps = sorted({int(x.strip()) for x in args.save_steps.split(",") if x.strip()})
    all_windows: list[list[dict]] = []
    for ep in range(args.epochs):
        all_windows.extend(C1T.build_atomic_windows(pairs, args.grad_accum, seed=args.seed + ep * 1009))
    total_steps = min(len(all_windows), args.max_steps) if args.max_steps > 0 else len(all_windows)
    warmup_steps = max(1, int(0.10 * total_steps))
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    model.train()
    logs: list[dict[str, Any]] = []
    saved_checkpoints: dict[str, str] = {}
    zone_norm_trajectory: dict[str, dict[str, float]] = {}
    n_clipped = 0

    for step_idx in range(total_steps):
        win = all_windows[step_idx]
        cur_step = step_idx + 1
        lrf = C1T.lr_factor(step_idx, total_steps, warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lrf

        opt.zero_grad(set_to_none=True)
        win_n = len(win)
        w_dpo = w_marg = w_acc = w_lpw = w_lpl = 0.0
        for j in range(0, win_n, args.microbatch):
            mb = win[j : j + args.microbatch]
            loss_sum, st = compute_dpo_microbatch(
                model, mb, pad_id, beta=args.beta, aux_weight=args.aux_weight, sft_weight=args.sft_weight
            )
            (loss_sum / win_n).backward()
            w_dpo += st["dpo_sum"]
            w_marg += st["margin_sum"]
            w_acc += st["acc_sum"]
            w_lpw += st["logp_w_sum"]
            w_lpl += st["logp_l_sum"]

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm))
        n_clipped += int(grad_norm > args.max_grad_norm)
        opt.step()

        logs.append({
            "step": cur_step,
            "lr": round(opt.param_groups[-1]["lr"], 8),
            "dpo_mean": round(w_dpo / win_n, 4),
            "margin_mean": round(w_marg / win_n, 4),
            "pair_acc": round(w_acc / win_n, 3),
            "logp_w_mean": round(w_lpw / win_n, 3),
            "logp_l_mean": round(w_lpl / win_n, 3),
            "grad_norm": round(grad_norm, 4),
        })
        if cur_step % 25 == 0 or cur_step == total_steps:
            r = logs[-1]
            print(
                f"  step {cur_step:3d}/{total_steps} | dpo={r['dpo_mean']:.4f} marg={r['margin_mean']:+.3f} "
                f"acc={r['pair_acc']:.3f} gnorm={r['grad_norm']:.2f}"
            )
        if cur_step in save_steps:
            ckpt_p = args.out_dir / f"adapter_state_step{cur_step:04d}.pt"
            M35.save_c3_adapter_state(model, ckpt_p)
            saved_checkpoints[str(cur_step)] = C3T._file_sha256(ckpt_p)
            zone_norm_trajectory[str(cur_step)] = M35.zone_delta_norms(model)

    final_p = args.out_dir / "adapter_state.pt"
    M35.save_c3_adapter_state(model, final_p)
    final_sha = C3T._file_sha256(final_p)
    final_norms = M35.zone_delta_norms(model)
    scaling_dict = {"scaling_mode": args.scaling_mode, "base_scale": args.lora_scale, "anchor_rank": args.anchor_rank}

    adapter_cfg = {
        "schema": "cedar.acl_exp_c35.adapter_config.v1",
        "base_model": args.model,
        "model_key": args.model_key,
        "arch": args.arch,
        "loss_mode": f"joint_dpo_beta{args.beta}",
        "iit_target": "none",
        "arch_spec": arch_spec,
        "iit_spec": None,
        "scaling": scaling_dict,
        "trainable_params": adapter_info["trainable_params"],
        "adapter_state_sha256": final_sha,
    }
    (args.out_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "cedar.acl_exp_c35.dpo_train_meta.v1",
        "baseline_label": "CEDAR same-data Joint DPO adaptation (Uniform-LoRA)",
        "reference_model": f"frozen M0 ({args.model})",
        "model": args.model,
        "model_key": args.model_key,
        "arch": args.arch,
        "beta": args.beta,
        "sft_weight": args.sft_weight,
        "seed": args.seed,
        "train_file": str(args.train_file),
        "train_file_sha256": C3T._file_sha256(args.train_file),
        "n_pairs_total": len(pairs),
        "pair_counts_all": dict(dir_counts),
        "pair_counts_core": dict(core_dir_counts),
        "total_steps": total_steps,
        "clip_rate": round(n_clipped / max(1, total_steps), 4),
        "hyperparams": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "grad_accum": args.grad_accum,
            "microbatch": args.microbatch,
            "aux_weight": args.aux_weight,
            "beta": args.beta,
            "sft_weight": args.sft_weight,
        },
        "adapter_info": adapter_info,
        "final_zone_delta_norms": final_norms,
        "zone_delta_norm_trajectory": zone_norm_trajectory,
        "adapter_state_sha256": final_sha,
        "saved_checkpoints_sha256": saved_checkpoints,
        "elapsed_seconds": round(time.time() - t0, 1),
        "tail_logs": logs[-20:],
        "all_logs": logs,
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[c35_train_dpo] DONE in {meta['elapsed_seconds']}s | clip_rate={meta['clip_rate']:.2%} | sha={final_sha[:12]}")


if __name__ == "__main__":
    main()
