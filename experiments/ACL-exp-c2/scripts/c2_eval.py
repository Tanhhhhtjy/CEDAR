#!/usr/bin/env python3
"""ACL-exp-c2 evaluation wrapper around frozen `c1_eval.py` (HF lossless live / vLLM native LoRA).

Registers ACL-exp-c2 datasets (`c2_confirm_bidir`, `c2_stealth_stress`) alongside open C1 reference sets
(`c1_dev`, `c1_proxy_template_v2`, `c1_step2b_confirm_ce`) WITHOUT modifying any file under `experiments/ACL-exp-c1/`,
and PHYSICALLY BLOCKS sealed `c1_test_id` and `c1_test_ood` unless `--allow-sealed-final` is explicitly passed!

Usage (GPU):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c2/scripts/c2_eval.py \
      --model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm Targeted-Dual \
      --adapter-state <PATH>/adapter_state.pt --regime no_think \
      --datasets c2_confirm_bidir c2_stealth_stress c1_proxy_template_v2 c1_dev \
      --eligible-ids-in <M0_ELIG> --out-dir <OUT>

Usage (CPU self-test):
  python experiments/ACL-exp-c2/scripts/c2_eval.py --selftest-tiny
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C2 = HERE.parents[0]
REPO = C2.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"

sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import c1_eval as C1EV  # noqa: E402

SEALED_DATASETS = {"c1_test_id", "c1_test_ood"}

# Register C2 datasets cleanly into C1EV.DEFAULT_DATASETS (in-memory only; never edits C1 files!)
C1EV.DEFAULT_DATASETS["c2_confirm_bidir"] = C2 / "data" / "c2_confirm_bidir_bases.jsonl"
C1EV.DEFAULT_DATASETS["c2_stealth_stress"] = C2 / "data" / "c2_stealth_stress_bases.jsonl"


def _check_sealed_guard(argv: list[str]) -> list[str]:
    allow_sealed = "--allow-sealed-final" in argv
    cleaned = [a for a in argv if a != "--allow-sealed-final"]
    if not allow_sealed:
        for tok in cleaned:
            if tok in SEALED_DATASETS:
                raise RuntimeError(
                    f"SEALED TEST SET GUARD: Refusing to evaluate sealed dataset {tok!r} during ACL-exp-c2 development! "
                    "(Pass --allow-sealed-final only after final freeze approval.)"
                )
    return cleaned


def run_selftest_tiny() -> None:
    """Verify C2 datasets load cleanly through C1EV message rendering & tiny Qwen3 scoring + sealed guard works."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from render_prompts import build_messages, load_config
    import c1_train as C1T

    # 1. Verify sealed guard blocks c1_test_id / c1_test_ood
    for s in SEALED_DATASETS:
        blocked = False
        try:
            _check_sealed_guard(["--datasets", s])
        except RuntimeError:
            blocked = True
        assert blocked, f"Sealed guard failed to block {s}"

    # 2. Verify c2_confirm_bidir & c2_stealth_stress (both late-buried AND classical sophisms!) render & score cleanly
    cfg_a1 = load_config()
    tok = C1T._TinyTok()
    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    opt_ids = {L: tok.encode(L)[0] for L in ("A", "B", "C", "D")}

    for ds_name in ("c2_confirm_bidir", "c2_stealth_stress"):
        p = C1EV.DEFAULT_DATASETS[ds_name]
        assert p.is_file(), f"Missing dataset: {p}"
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        # Check first row + last row (covers both late-buried and classical sophism in c2_stealth_stress!)
        sample_rows = [rows[0], rows[-1]]
        for row in sample_rows:
            for arm in ("turn1_F", "turn1_V", "V", "V_hes", "V_auth", "F", "F_auth", "F_hes", "shell_only", "S", "AO"):
                msgs = build_messages(cfg_a1, row, arm, "no_think")
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Final:"
                ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
                with torch.no_grad():
                    logits = model(input_ids=ids).logits[0, -1]
                u = float(logits[opt_ids[row["target_answer"]]] - logits[opt_ids[row["initial_answer"]]])
                assert u == u  # finite
        print(f"[OK] c2_eval tiny smoke passed on {ds_name} (n={len(rows)}, first={rows[0]['base_id']}, last={rows[-1]['base_id']})")
    print("[OK] ALL c2_eval self-tests PASSED!")


def main() -> None:
    if "--selftest-tiny" in sys.argv[1:]:
        run_selftest_tiny()
        return
    sys.argv = [sys.argv[0]] + _check_sealed_guard(sys.argv[1:])
    C1EV.main()


if __name__ == "__main__":
    main()
