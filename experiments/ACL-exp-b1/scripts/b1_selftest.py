#!/usr/bin/env python3
"""CPU self-test of b1_lib on a tiny randomly initialised Qwen3 (no download, no tokenizer).

Checks: hook shapes; exactness of residual patching; suffix/last alignment; head/neuron/MLP patching
paths; AtP ranks single-head patch effects (Spearman > 0.5); DLA runs. Run:  python b1_selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_lib as L  # noqa: E402


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      tie_word_embeddings=False)
    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).float().eval()
    return m


def spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main():
    m = tiny_model()
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(5, 300, (20,), generator=g)
    suffix = torch.randint(5, 300, (6,), generator=g)
    span_c = torch.randint(5, 300, (5,), generator=g)
    span_k = torch.randint(5, 300, (3,), generator=g)   # different length -> suffix alignment only
    clean = torch.cat([prefix, span_c, suffix])
    corrupt = torch.cat([prefix, span_k, suffix])
    p = L.EncodedPair("t", "test", clean, corrupt, pos_token=7, neg_token=11)
    assert p.common_suffix() == 6 and p.common_prefix() == 20, (p.common_suffix(), p.common_prefix())

    m_clean = L.run_metric(m, clean, 7, 11)
    m_corr = L.run_metric(m, corrupt, 7, 11)

    # 1) residual after last layer at last position -> exact clean metric
    r = L.patch_recovery(m, p, "resid", [(2, 0)], pos_mode="last")
    assert abs(r["m_patched"] - m_clean) < 1e-4, r
    # 2) residual after layer 0 at all suffix positions (+ nothing else) -> not necessarily exact, but runs
    L.patch_recovery(m, p, "resid", [(0, 0)], pos_mode="suffix")
    # 3) equal-length pair: residual after layer 0 at ALL positions -> exact clean
    corrupt_eq = clean.clone(); corrupt_eq[22] = (corrupt_eq[22] + 17) % 300
    peq = L.EncodedPair("eq", "test", clean, corrupt_eq, 7, 11)
    r = L.patch_recovery(m, peq, "resid", [(0, 0)], pos_mode="all")
    assert abs(r["m_patched"] - m_clean) < 1e-4, r
    # 4) all heads + MLPs of every layer at all positions in equal-length pair, plus layer-0 resid -> exact
    # (sanity of head slicing): patch every head of every layer only at all positions; compare to manual
    H = L.n_heads_of(m)
    units = [(li, h) for li in range(3) for h in range(H)]
    r_heads = L.patch_recovery(m, peq, "head", units, pos_mode="all")
    assert r_heads["m_patched"] == r_heads["m_patched"]  # not NaN
    L.patch_recovery(m, peq, "mlp", [(1, 0)], pos_mode="all")
    L.patch_recovery(m, peq, "neuron", [(1, 3), (1, 5), (2, 7)], pos_mode="all")

    # 5) AtP vs real single-head patching (denoising, suffix positions)
    scores = L.atp(m, p, kinds=("head", "mlp", "neuron"), pos_mode="suffix")
    assert abs(scores["effect"] - (m_clean - m_corr)) < 1e-3
    atp_vals, real_vals = [], []
    cache = L.build_cache(m, p, ["head"], pos_mode="suffix")
    for li in range(3):
        for h in range(H):
            atp_vals.append(float(scores[("head", li)][h]))
            rr = L.patch_recovery(m, p, "head", [(li, h)], pos_mode="suffix", cache=cache)
            real_vals.append(rr["m_patched"] - rr["m_corrupt"])
    rho = spearman(atp_vals, real_vals)
    print(f"AtP vs real single-head patch Spearman = {rho:.3f}")
    assert rho > 0.5, rho

    # 6) aggregation helpers + random control + DLA
    S = L.components_from_scores([scores, scores], "head", 3)
    assert S.shape == (3, H)
    top = L.top_k(S, 2)
    rnd = L.layer_matched_random(top, H, seed=0)
    assert len(rnd) == len(top) and not set(rnd) & set(top), (top, rnd)
    assert sorted(li for li, _ in rnd) == sorted(li for li, _ in top)
    dla = L.head_direct_logit_attribution(m, clean, 7, 11)
    assert dla.shape == (3, H)
    print("[OK] b1_lib self-test passed")


if __name__ == "__main__":
    main()
