#!/usr/bin/env python3
"""ACL-exp-c3.5 Evaluation Wrapper registering `c3_blind_confirm_v2` & `c3_blind_think_v2` around frozen `c3_eval.py`.

Zero modifications to frozen `ACL-exp-c3/`; physically blocks `c1_test_id / c1_test_ood` unless `--allow-sealed-final`.

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_eval.py --selftest-tiny
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-c1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import c3_eval as C3EV  # noqa: E402

C3EV.C3_DATASETS["c3_blind_confirm_v2"] = C35 / "data" / "c3_blind_confirm_v2_bases.jsonl"
C3EV.C3_DATASETS["c3_blind_think_v2"] = C35 / "data" / "c3_blind_think_v2_bases.jsonl"


def run_selftest_tiny() -> None:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import b1_lib as B1L
    import c1_train as C1T
    from render_prompts import load_config

    for ds in ("c3_blind_confirm_v2", "c3_blind_think_v2"):
        p = C3EV.C3_DATASETS[ds]
        assert p.is_file(), p

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    tok = C1T._TinyTok()
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in C3EV.OPTION_LETTERS}
    model = Qwen3ForCausalLM(cfg).float().eval()
    rows = [json.loads(l) for l in C3EV.C3_DATASETS["c3_blind_confirm_v2"].read_text(encoding="utf-8").splitlines()[:2]]

    orig_enc = B1L.encode_prompt
    def _tiny_enc(t, msgs, pref, enable_thinking=False):
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
        return torch.tensor(t.encode(txt), dtype=torch.long)
    B1L.encode_prompt = _tiny_enc
    try:
        recs = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, rows, list(C3EV.CORE_4CELL_ARMS), opt_ids, batch_size=2)
    finally:
        B1L.encode_prompt = orig_enc
    assert len(recs) == 12
    print("[OK] c35_eval self-test PASSED on c3_blind_confirm_v2!")


def main() -> None:
    if "--selftest-tiny" in sys.argv[1:]:
        run_selftest_tiny()
        return
    C3EV.main()


if __name__ == "__main__":
    main()
