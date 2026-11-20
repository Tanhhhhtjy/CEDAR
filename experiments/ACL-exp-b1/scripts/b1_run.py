#!/usr/bin/env python3
"""ACL-exp-b1 runner: white-box localisation of the social / form / content pathways (Qwen3, no_think).

Stages (run in order; each writes JSON/PT under results/<model_alias>/):
  sanity       effect E = m(clean) - m(corrupt) for every pair -> per-contrast median, sign consistency, CI
  resid_sweep  denoising residual-stream patching per layer over the aligned suffix (+ per-position on a subset)
  atp          denoising AtP for heads and MLP layers (optionally MLP neurons) on every pair; per-half means
  dla          direct logit attribution of heads on C_soc (Pinpoint-style direct-path ranking)
  verify       real patching of top-k heads / MLP layers vs layer-matched random sets (5 seeds) per contrast
  reliability  split-half top-k Jaccard within contrast vs across contrasts
  dissociation recovery matrix: top-k heads of contrast X patched on pairs of contrast Y (+ random control)

Usage (GPU host):
  python experiments/ACL-exp-b1/scripts/b1_run.py --model /path/Qwen3-8B --alias qwen3_8b --stages all
  python experiments/ACL-exp-b1/scripts/b1_run.py --selftest-tiny          # CPU smoke test, no download
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
B1 = HERE.parents[0]
sys.path.insert(0, str(HERE))
import b1_lib as L  # noqa: E402

ALL_STAGES = ["sanity", "resid_sweep", "atp", "dla", "verify", "reliability", "dissociation"]


# ----------------------------------------------------------------------------------------------- io


def load_cfg() -> dict:
    return yaml.safe_load((B1 / "configs" / "b1_config.yaml").read_text(encoding="utf-8"))


def load_eligible_ids(path: Path, model_key: str) -> tuple[set[str], dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if model_key not in blob.get("models", {}):
        raise KeyError(f"model_key {model_key!r} not in {path}; available={sorted(blob.get('models', {}))}")
    info = blob["models"][model_key]
    ids = set(info["eligible_base_ids"])
    meta = {
        "eligible_file": str(path),
        "eligible_model": model_key,
        "source_commit": blob.get("source_commit"),
        "source_sha256": info.get("source_sha256"),
        "counts": info.get("counts"),
        "total_eligible_bases": len(ids),
    }
    return ids, meta


def load_pairs(
    path: Path,
    contrasts: list[str],
    include_deferred: bool,
    limit: int | None,
    eligible_ids: set[str] | None = None,
) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["contrast"] in contrasts and (include_deferred or not r["deferred"])]
    if eligible_ids is not None:
        rows = [r for r in rows if r["base_id"] in eligible_ids]
    if limit:
        by = {}
        for r in rows:
            by.setdefault(r["contrast"], []).append(r)
        rows = [r for c in by.values() for r in c[:limit]]
    return rows


def encode_all(tok, rows: list[dict]) -> list[L.EncodedPair]:
    out, cache = [], {}
    for r in rows:
        key = (r["readout_key"], r["pos_word"], r["neg_word"])
        if key not in cache:
            cache[key] = L.readout_token_ids(tok, *key)
        pos_t, neg_t = cache[key]
        c = L.encode_prompt(tok, r["clean_messages"], r["readout_key"])
        k = L.encode_prompt(tok, r["corrupt_messages"], r["readout_key"])
        key_ids = tok.encode(r["readout_key"], add_special_tokens=False)
        for ids in (c, k):  # the readout key must survive tokenisation in context (no merge with the preceding text)
            assert ids[-len(key_ids):].tolist() == key_ids, f"readout key not token-aligned in {r['pair_id']}"
        meta = {x: r[x] for x in ("dataset", "base_id", "family_id", "pair_type", "fallacy_class", "split_half")}
        p = L.EncodedPair(r["pair_id"], r["contrast"], c, k, pos_t, neg_t, meta)
        meta["common_suffix"] = p.common_suffix()
        out.append(p)
    return out


def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        torch.save(obj, path)
    else:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")


def by_contrast(pairs):
    d = {}
    for p in pairs:
        d.setdefault(p.contrast, []).append(p)
    return d


# ----------------------------------------------------------------------------------------------- stages


def stage_sanity(model, pairs, out_dir, cfg):
    res = {}
    per_pair = []
    min_e = cfg["min_effect_logits"]
    for c, ps in by_contrast(pairs).items():
        effects = []
        for p in ps:
            mc = L.run_metric(model, p.clean_ids, p.pos_token, p.neg_token)
            mk = L.run_metric(model, p.corrupt_ids, p.pos_token, p.neg_token)
            effects.append(mc - mk)
            per_pair.append({"pair_id": p.pair_id, "m_clean": mc, "m_corrupt": mk, **p.meta})
        res[c] = {"n": len(effects), "median_effect": statistics.median(effects),
                  "sign_consistency": sum(e > 0 for e in effects) / len(effects), "mean": L.mean_ci(effects),
                  "n_pos_ge_min": sum(e >= min_e for e in effects),
                  "n_neg_le_neg_min": sum(e <= -min_e for e in effects) if min_e > 0 else 0,
                  "median_common_suffix": statistics.median(p.meta["common_suffix"] for p in ps)}
    save({"summary": res, "pairs": per_pair}, out_dir / "stage_sanity.json")
    return res


def stage_resid_sweep(model, pairs, out_dir, cfg):
    n_layers = len(L.layers_of(model))
    min_e = cfg["min_effect_logits"]
    max_pos_suffix = cfg.get("resid_per_position_max_suffix", 48)
    res = {}
    for c, ps in by_contrast(pairs).items():
        curves = []
        valid_ps = []
        for p in ps:
            cache = L.build_cache(model, p, ["resid"], "suffix")
            E = cache["m_clean"] - cache["m_corrupt"]
            if E < min_e or abs(E) < 1e-6:
                continue
            valid_ps.append(p)
            curves.append([L.patch_recovery(model, p, "resid", [(li, 0)], "suffix", cache=cache)["recovery"]
                           for li in range(n_layers)])
        per_pos = []
        pos_ps = (valid_ps or ps)[: cfg["resid_per_position_pairs"]]
        for p in pos_ps:
            pc, pk = p.aligned_positions("suffix", max_suffix=max_pos_suffix)
            layers = list(range(0, n_layers, cfg["resid_per_position_layer_stride"]))
            _, acts_all = L.cache_run(model, p.clean_ids, ["resid"], pc)
            grid = []
            for li in layers:
                row = []
                for j in range(len(pc)):
                    spec = {("resid", li): [(None, acts_all[("resid", li)][j:j + 1])]}
                    with L.Patcher(model, spec, [pk[j]]):
                        row.append(L.run_metric(model, p.corrupt_ids, p.pos_token, p.neg_token))
                grid.append(row)
            per_pos.append({"pair_id": p.pair_id, "layers": layers, "n_suffix_positions": len(pc),
                            "m_patched": grid,
                            "m_corrupt": L.run_metric(model, p.corrupt_ids, p.pos_token, p.neg_token),
                            "m_clean": L.run_metric(model, p.clean_ids, p.pos_token, p.neg_token)})
        res[c] = {"n_pairs": len(curves), "n_excluded_low_or_neg": len(ps) - len(curves),
                  "mean_recovery_by_layer": [statistics.mean(x) for x in zip(*curves)] if curves else [],
                  "per_position_examples": per_pos}
    save(res, out_dir / "stage_resid_sweep.json")
    return res


def stage_atp(model, pairs, out_dir, cfg):
    kinds = tuple(cfg["atp_kinds"])
    n_layers = len(L.layers_of(model))
    min_e = cfg["min_effect_logits"]
    summary = {}
    for c, ps in by_contrast(pairs).items():
        maps = {"A": [], "B": [], "all": []}
        effects = []
        for p in ps:
            s = L.atp(model, p, kinds=kinds, pos_mode="suffix")
            effects.append(s["effect"])
            if s["effect"] < min_e or abs(s["effect"]) < 1e-6:
                continue
            # normalise per pair so that every positively oriented pair contributes its fraction of its own effect
            norm = {k: (v / s["effect"]) for k, v in s.items() if isinstance(k, tuple)}
            maps["all"].append(norm)
            maps[p.meta["split_half"]].append(norm)
        blob = {"effects": effects}
        for half, ms in maps.items():
            if not ms:
                continue
            for kind in kinds:
                blob[f"{kind}_{half}"] = L.components_from_scores(ms, kind, n_layers)
        save(blob, out_dir / f"atp_{c}.pt")
        summary[c] = {"n_total": len(ps), "n_used": len(maps["all"]), "n_A": len(maps["A"]), "n_B": len(maps["B"]),
                      "n_neg": sum(e < 0 for e in effects), "n_excluded": len(ps) - len(maps["all"]),
                      "top_heads": L.top_k(blob["head_all"], cfg["report_top"]) if "head_all" in blob else []}
    save(summary, out_dir / "stage_atp.json")
    return summary


def stage_dla(model, pairs, out_dir, cfg):
    groups = by_contrast(pairs)
    dla_cs = [c for c in cfg.get("dla_contrasts", ["C_pinpoint", "C_soc"]) if c in groups]
    summary = {}
    for c in dla_cs:
        by_half = {"A": [], "B": [], "all": []}
        for p in groups[c]:
            dc = L.head_direct_logit_attribution(model, p.clean_ids, p.pos_token, p.neg_token)
            dk = L.head_direct_logit_attribution(model, p.corrupt_ids, p.pos_token, p.neg_token)
            diff = dc - dk
            by_half["all"].append(diff)
            by_half[p.meta["split_half"]].append(diff)
        if by_half["all"]:
            blob = {f"dla_{h}": torch.stack(ms).mean(0) for h, ms in by_half.items() if ms}
            save(blob, out_dir / f"dla_{c}.pt")
            summary[c] = {"n": len(by_half["all"]), "n_A": len(by_half["A"]), "n_B": len(by_half["B"]),
                          "top_heads_direct": L.top_k(blob["dla_all"], cfg["report_top"])}
    save(summary, out_dir / "stage_dla.json")
    return summary


def _load_atp(out_dir, c, kind, half="all"):
    blob = torch.load(out_dir / f"atp_{c}.pt")
    return blob.get(f"{kind}_{half}")


def _load_dla(out_dir, c, half="all"):
    p = out_dir / f"dla_{c}.pt"
    if not p.exists():
        return None
    blob = torch.load(p)
    return blob.get(f"dla_{half}", blob.get("dla_all", blob.get(f"dla_{c}")))


def stage_verify(model, pairs, out_dir, cfg):
    H = L.n_heads_of(model)
    n_layers = len(L.layers_of(model))
    min_e = cfg["min_effect_logits"]
    res = {}
    for c, ps in by_contrast(pairs).items():
        S_head = _load_atp(out_dir, c, "head")
        S_mlp = _load_atp(out_dir, c, "mlp")
        if S_head is None:
            res[c] = {"n_pairs": 0, "heads": {}, "mlp_layers": {}}
            continue
        ps_eval = [p for p in ps if p.meta["split_half"] == cfg["verify_on_half"]] or ps
        caches = {p.pair_id: L.build_cache(model, p, ["head", "mlp"], "suffix") for p in ps_eval}
        ps_eval = [p for p in ps_eval
                   if (caches[p.pair_id]["m_clean"] - caches[p.pair_id]["m_corrupt"]) >= min_e
                   and abs(caches[p.pair_id]["m_clean"] - caches[p.pair_id]["m_corrupt"]) >= 1e-6]
        if not ps_eval:
            res[c] = {"n_pairs": 0, "heads": {}, "mlp_layers": {}}
            continue
        # selection on the OTHER half to avoid double dipping (falls back to 'all' if a half is empty)
        sel_half = "B" if cfg["verify_on_half"] == "A" else "A"
        S_sel = _load_atp(out_dir, c, "head", sel_half)
        S_sel = S_sel if S_sel is not None else S_head
        out = {"n_pairs": len(ps_eval), "selection_half": sel_half, "eval_half": cfg["verify_on_half"],
               "heads": {}, "mlp_layers": {}}
        for k in cfg["verify_k_heads"]:
            top = L.top_k(S_sel, k)
            rec = [L.patch_recovery(model, p, "head", top, "suffix", cache=caches[p.pair_id])["recovery"] for p in ps_eval]
            rnd = []
            for seed in range(cfg["n_random_seeds"]):
                rs = L.layer_matched_random(top, H, seed)
                rnd.append(statistics.mean(
                    L.patch_recovery(model, p, "head", rs, "suffix", cache=caches[p.pair_id])["recovery"] for p in ps_eval))
            out["heads"][k] = {"top": top, "recovery": L.mean_ci(rec), "random_layer_matched": rnd}
        S_mlp_sel = _load_atp(out_dir, c, "mlp", sel_half)
        S_mlp_sel = S_mlp_sel if S_mlp_sel is not None else S_mlp
        if S_mlp_sel is not None:
            for k in cfg["verify_k_mlp_layers"]:
                top = L.top_k(S_mlp_sel.view(n_layers, 1), k)
                rec = [L.patch_recovery(model, p, "mlp", top, "suffix", cache=caches[p.pair_id])["recovery"] for p in ps_eval]
                pool = [l for l in range(n_layers) if (l, 0) not in top]
                rnd_list = []
                for seed in range(cfg["n_random_seeds"]):
                    rng = random.Random(seed)
                    rnd_layers = [(li, 0) for li in rng.sample(pool, min(k, len(pool)))]
                    rnd_list.append(statistics.mean(
                        L.patch_recovery(model, p, "mlp", rnd_layers, "suffix", cache=caches[p.pair_id])["recovery"]
                        for p in ps_eval))
                out["mlp_layers"][k] = {"top": top, "recovery": L.mean_ci(rec), "random_layers": rnd_list}
        res[c] = out
    save(res, out_dir / "stage_verify.json")
    return res


def _weighted_overlap(top_x: list[tuple[int, int]], S_y: torch.Tensor, top_y: list[tuple[int, int]]) -> float:
    """Fraction of Y's top-k positive AtP mass carried by heads shared with X's top-k."""
    shared = set(top_x) & set(top_y)
    denom = sum(max(0.0, float(S_y[li, h])) for li, h in top_y)
    if denom <= 1e-12:
        return 0.0
    num = sum(max(0.0, float(S_y[li, h])) for li, h in shared)
    return num / denom


def stage_reliability(model, pairs, out_dir, cfg):
    cs = [c for c in by_contrast(pairs) if (out_dir / f"atp_{c}.pt").exists() and _load_atp(out_dir, c, "head") is not None]
    res = {"within_split_half": {}, "across_contrasts": {}, "weighted_overlap_top_k": {}}
    tops = {}
    scores = {}
    for c in cs:
        A, B, ALL = (_load_atp(out_dir, c, "head", h) for h in ("A", "B", "all"))
        scores[c] = ALL
        for k in cfg["reliability_k"]:
            if A is not None and B is not None:
                res["within_split_half"].setdefault(c, {})[k] = L.jaccard(L.top_k(A, k), L.top_k(B, k))
            tops[(c, k)] = L.top_k(ALL, k)
    for k in cfg["reliability_k"]:
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                res["across_contrasts"].setdefault(f"{a}~{b}", {})[k] = L.jaccard(tops[(a, k)], tops[(b, k)])
        for a in cs:
            for b in cs:
                if a != b:
                    res["weighted_overlap_top_k"].setdefault(f"{a}->{b}", {})[k] = _weighted_overlap(
                        tops[(a, k)], scores[b], tops[(b, k)]
                    )
    for dc in cfg.get("dla_contrasts", ["C_pinpoint", "C_soc"]):
        M = _load_dla(out_dir, dc, "all")
        if M is None:
            continue
        for c in cs:
            for k in cfg["reliability_k"]:
                td = L.top_k(M, k)
                res["across_contrasts"].setdefault(f"DLA({dc})~{c}", {})[k] = L.jaccard(td, tops[(c, k)])
                res["weighted_overlap_top_k"].setdefault(f"DLA({dc})->{c}", {})[k] = _weighted_overlap(
                    td, scores[c], tops[(c, k)]
                )
    save(res, out_dir / "stage_reliability.json")
    return res


def stage_dissociation(model, pairs, out_dir, cfg):
    H = L.n_heads_of(model)
    min_e = cfg["min_effect_logits"]
    eval_half = cfg["verify_on_half"]
    sel_half = "B" if eval_half == "A" else "A"
    groups = by_contrast(pairs)
    cs = [c for c in cfg["dissociation_contrasts"]
          if c in groups and (out_dir / f"atp_{c}.pt").exists() and _load_atp(out_dir, c, "head") is not None]
    k = cfg["dissociation_k"]
    tops = {}
    for c in cs:
        S_sel = _load_atp(out_dir, c, "head", sel_half)
        if S_sel is None:
            S_sel = _load_atp(out_dir, c, "head", "all")
        tops[c] = L.top_k(S_sel, k)
    for dc in cfg.get("dla_contrasts", ["C_pinpoint", "C_soc"]):
        M_sel = _load_dla(out_dir, dc, sel_half)
        if M_sel is not None:
            tops[f"DLA({dc})"] = L.top_k(M_sel, k)
    res = {"k": k, "selection_half": sel_half, "eval_half": eval_half,
           "n_pairs_by_target": {}, "matrix": {}, "normalized_matrix": {},
           "random": {}, "random_layer_matched_seeds": {},
           "random_by_source": {}, "random_by_source_seeds": {}}
    for y in cs:
        ps_half = [p for p in groups[y] if p.meta["split_half"] == eval_half] or groups[y]
        ps = ps_half[: cfg["dissociation_max_pairs"]]
        caches = {p.pair_id: L.build_cache(model, p, ["head"], "suffix") for p in ps}
        ps = [p for p in ps
              if (caches[p.pair_id]["m_clean"] - caches[p.pair_id]["m_corrupt"]) >= min_e
              and abs(caches[p.pair_id]["m_clean"] - caches[p.pair_id]["m_corrupt"]) >= 1e-6]
        res["n_pairs_by_target"][y] = len(ps)
        if not ps:
            continue
        for x, top in tops.items():
            outs = [L.patch_recovery(model, p, "head", top, "suffix", cache=caches[p.pair_id]) for p in ps]
            rec = [o["recovery"] for o in outs]
            clamped = [max(-1.0, min(2.0, r)) for r in rec]
            dm = [o["m_patched"] - o["m_corrupt"] for o in outs]
            cell = L.mean_ci(rec)
            cell["median"] = statistics.median(rec)
            cell["clamped_mean"] = statistics.mean(clamped)
            cell["delta_m_mean"] = statistics.mean(dm)
            cell["delta_m_median"] = statistics.median(dm)
            res["matrix"].setdefault(x, {})[y] = cell
        # Target-Y-matched random control (5 seeds)
        rnd_seeds = []
        for seed in range(cfg["n_random_seeds"]):
            rs = L.layer_matched_random(tops[y], H, seed=seed)
            rnd_seeds.append(statistics.mean(
                L.patch_recovery(model, p, "head", rs, "suffix", cache=caches[p.pair_id])["recovery"] for p in ps))
        res["random_layer_matched_seeds"][y] = rnd_seeds
        res["random"][y] = {"n_seeds": len(rnd_seeds), "mean": statistics.mean(rnd_seeds), "max": max(rnd_seeds)}
        # Source-X-matched random control (5 seeds) — checks off-diagonal & late-layer DLA rows against X's layer distribution
        for x, top_x in tops.items():
            if x == y:
                res["random_by_source_seeds"].setdefault(x, {})[y] = rnd_seeds
                res["random_by_source"].setdefault(x, {})[y] = res["random"][y]
                continue
            sx_seeds = []
            for seed in range(cfg["n_random_seeds"]):
                rs = L.layer_matched_random(top_x, H, seed=seed)
                sx_seeds.append(statistics.mean(
                    L.patch_recovery(model, p, "head", rs, "suffix", cache=caches[p.pair_id])["recovery"] for p in ps))
            res["random_by_source_seeds"].setdefault(x, {})[y] = sx_seeds
            res["random_by_source"].setdefault(x, {})[y] = {
                "n_seeds": len(sx_seeds), "mean": statistics.mean(sx_seeds), "max": max(sx_seeds)
            }
    # Normalised dissociation index D[x->y] = R[x][y] / R[y][y] (immune to over-recovery scaling)
    for x, row in res["matrix"].items():
        for y, cell in row.items():
            diag = res["matrix"].get(y, {}).get(y, {}).get("mean")
            if diag and abs(diag) > 1e-6 and "mean" in cell:
                res["normalized_matrix"].setdefault(x, {})[y] = cell["mean"] / diag
    save(res, out_dir / "stage_dissociation.json")
    return res


STAGE_FN = {"sanity": stage_sanity, "resid_sweep": stage_resid_sweep, "atp": stage_atp, "dla": stage_dla,
            "verify": stage_verify, "reliability": stage_reliability, "dissociation": stage_dissociation}


# ----------------------------------------------------------------------------------------------- tiny self-test


class FakeTok:
    """Word-level hashing tokenizer with a chat template (CPU smoke test only)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        s = "".join(f"<|{m['role']}|> {m['content']} <|end|> " for m in messages)
        return s + ("<|assistant|> <think> </think> " if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [5 + (sum(ord(ch) * 31 ** i for i, ch in enumerate(w)) % 290) for w in text.split()]


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).float().eval()


# ----------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--alias")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--contrasts", nargs="*")
    ap.add_argument("--dissociation-contrasts", nargs="*", help="override cfg['dissociation_contrasts']")
    ap.add_argument("--eligible-file", type=Path, help="JSON from b1_export_eligible.py to restrict base_ids")
    ap.add_argument("--eligible-model", help="model key inside --eligible-file (e.g. qwen3_14b or qwen3_8b)")
    ap.add_argument("--include-deferred", action="store_true", help="also run C_content_CF / C_ctx_shell (only after the A1 2x2 gate)")
    ap.add_argument("--limit", type=int, help="max pairs per contrast (debug)")
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")
    contrasts = args.contrasts or cfg["contrasts"]
    if args.dissociation_contrasts:
        cfg = {**cfg, "dissociation_contrasts": args.dissociation_contrasts}
    eligible_ids, eligible_meta = None, None
    if args.eligible_file:
        if not args.eligible_model:
            ap.error("--eligible-model is required when --eligible-file is set")
        elig_path = args.eligible_file if args.eligible_file.exists() else (B1 / args.eligible_file)
        eligible_ids, eligible_meta = load_eligible_ids(elig_path, args.eligible_model)
    pairs_path = B1 / cfg["pairs_file"]

    if args.selftest_tiny:
        model, tok, alias = tiny_model(), FakeTok(), args.alias or "selftest_tiny"
        cfg = {**cfg, "verify_k_heads": [2], "verify_k_mlp_layers": [1], "n_random_seeds": 2, "reliability_k": [2],
               "dissociation_k": 2, "report_top": 3, "resid_per_position_pairs": 1, "resid_per_position_layer_stride": 2,
               "resid_per_position_max_suffix": 8, "dissociation_max_pairs": 6, "min_effect_logits": -1e9}
        args.limit = args.limit or 6
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     device_map=cfg.get("device_map", "auto"))
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        alias = args.alias
    out_dir = B1 / "results" / alias
    rows = load_pairs(pairs_path, contrasts, args.include_deferred, args.limit, eligible_ids=eligible_ids)
    pairs = encode_all(tok, rows)
    print(f"[b1] {len(pairs)} pairs, contrasts={sorted(by_contrast(pairs))}, stages={stages}")
    meta = {"model": args.model, "alias": alias, "stages": stages, "n_pairs": len(pairs), "started": time.time(),
            "eligible_filter": eligible_meta, "config": cfg, "torch": torch.__version__}
    for st in stages:
        t0 = time.time()
        STAGE_FN[st](model, pairs, out_dir, cfg)
        meta[f"seconds_{st}"] = time.time() - t0
        print(f"[b1] stage {st} done in {meta[f'seconds_{st}']:.1f}s")
    save(meta, out_dir / "run_meta.json")


if __name__ == "__main__":
    main()
