#!/usr/bin/env python3
"""ACL-exp-b1 core library: readout, activation caching, patching and attribution for Qwen3 (HF transformers).

Design (see experiments/ACL-exp-b1/PROTOCOL.md):
* Every contrast is a (clean, corrupt) prompt pair that shares the Turn-1 prefix and the generation suffix
  (end of user turn + assistant header + empty think block + readout key "Final:" or "ArgumentCheck:").
* Metric m(x) = logit[pos_token] - logit[neg_token] at the LAST position (the token right after the readout key):
    dialog contrasts:  pos = a_t letter, neg = a0 letter  (this is u_theta of protocol A1)
    standalone verdict: pos = VALID, neg = INVALID
  Effect of a contrast on one pair: E = m(clean) - m(corrupt).
* Components: per-head attention outputs (= o_proj input slices), per-layer MLP outputs, MLP neurons
  (= down_proj input), residual stream after each decoder layer.
* Positions: the aligned common SUFFIX (from the end) by default - where information is read out; if clean and
  corrupt have equal length, all positions can be aligned ("all").
* Patching direction: denoising (run corrupt, insert clean activations) -> recovery = (m_patched - m_corrupt) / E.
* AtP: attribution(c) = sum_{pos} (a_clean - a_corrupt) . d m / d a  evaluated on the corrupt run (denoising AtP).

Works with transformers 4.5x and 5.x (decoder layers may return a tensor or a tuple).
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

# ----------------------------------------------------------------------------------------------- model access


def layers_of(model) -> list:
    return list(model.model.layers)


def n_heads_of(model) -> int:
    return model.config.num_attention_heads


def head_dim_of(model) -> int:
    hd = getattr(model.config, "head_dim", None)
    return hd if hd else model.config.hidden_size // model.config.num_attention_heads


def input_device_of(model) -> torch.device:
    """Device of the input embedding layer (safe under HF device_map='auto' multi-GPU splits)."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


def _out_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _with_tensor(output, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    return new


# ----------------------------------------------------------------------------------------------- pairs


@dataclass
class EncodedPair:
    pair_id: str
    contrast: str
    clean_ids: torch.Tensor  # [T_c]
    corrupt_ids: torch.Tensor  # [T_k]
    pos_token: int
    neg_token: int
    meta: dict = field(default_factory=dict)

    def common_suffix(self) -> int:
        a, b = self.clean_ids.tolist(), self.corrupt_ids.tolist()
        n = 0
        while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
            n += 1
        return n

    def common_prefix(self) -> int:
        a, b = self.clean_ids.tolist(), self.corrupt_ids.tolist()
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    def aligned_positions(self, mode: str = "suffix", max_suffix: int | None = None) -> tuple[list[int], list[int]]:
        """Return (clean_positions, corrupt_positions) that are aligned 1:1."""
        Tc, Tk = len(self.clean_ids), len(self.corrupt_ids)
        if mode == "all":
            assert Tc == Tk, "mode='all' requires equal-length clean/corrupt"
            return list(range(Tc)), list(range(Tk))
        if mode == "suffix":
            s = self.common_suffix()
            if max_suffix is not None:
                s = min(s, max_suffix)
            return [Tc - s + i for i in range(s)], [Tk - s + i for i in range(s)]
        if mode == "last":
            return [Tc - 1], [Tk - 1]
        raise ValueError(mode)


def readout_token_ids(tok, key: str, pos_word: str, neg_word: str) -> tuple[int, int]:
    """First token generated after `key` for each word, e.g. key='Final:', words 'B'/'A'."""
    base = tok.encode(key, add_special_tokens=False)
    ids = []
    for w in (pos_word, neg_word):
        full = tok.encode(f"{key} {w}", add_special_tokens=False)
        assert full[: len(base)] == base, f"tokenisation of {key!r} not prefix-stable"
        ids.append(full[len(base)])
    assert ids[0] != ids[1], f"readout tokens collide for {pos_word}/{neg_word}"
    return ids[0], ids[1]


def encode_prompt(tok, messages: list[dict], readout_key: str, enable_thinking: bool = False) -> torch.Tensor:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    text = text + readout_key
    return torch.tensor(tok.encode(text, add_special_tokens=False), dtype=torch.long)


# ----------------------------------------------------------------------------------------------- hooks


class Recorder:
    """Caches (and optionally keeps grads of) component activations during one forward pass."""

    def __init__(self, model, want: Iterable[str], retain_grad: bool = False):
        self.model, self.want, self.retain_grad = model, set(want), retain_grad
        self.acts: dict[tuple[str, int], torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        for li, layer in enumerate(layers_of(self.model)):
            if "head" in self.want:
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._pre("head", li)))
            if "neuron" in self.want:
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._pre("neuron", li)))
            if "mlp" in self.want:
                self.handles.append(layer.mlp.register_forward_hook(self._post("mlp", li)))
            if "resid" in self.want:
                self.handles.append(layer.register_forward_hook(self._post("resid", li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _store(self, kind, li, t):
        if self.retain_grad and t.requires_grad:
            t.retain_grad()
        self.acts[(kind, li)] = t

    def _pre(self, kind, li):
        def hook(mod, args):
            self._store(kind, li, args[0])
        return hook

    def _post(self, kind, li):
        def hook(mod, args, output):
            self._store(kind, li, _out_tensor(output))
        return hook


class Patcher:
    """Overwrites selected component activations at selected positions with donor values.

    spec: dict[(kind, layer)] -> list of (unit_index or None, donor_tensor[n_pos, width]) applied at `positions`
      kind 'head'  : unit = head index  (slice of o_proj input)
      kind 'neuron': unit = LongTensor of neuron indices (columns of down_proj input) or None (all)
      kind 'mlp'   : unit = None (whole MLP output)
      kind 'resid' : unit = None (whole residual stream after the layer)
    """

    def __init__(self, model, spec: dict, positions: list[int]):
        self.model, self.spec, self.positions = model, spec, positions
        self.hd = head_dim_of(model)
        self.handles = []

    def __enter__(self):
        for (kind, li), items in self.spec.items():
            layer = layers_of(self.model)[li]
            if kind == "head":
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._pre(items, "head")))
            elif kind == "neuron":
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._pre(items, "neuron")))
            elif kind == "mlp":
                self.handles.append(layer.mlp.register_forward_hook(self._post(items)))
            elif kind == "resid":
                self.handles.append(layer.register_forward_hook(self._post(items)))
            else:
                raise ValueError(kind)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _apply(self, x, items, kind):
        x = x.clone()
        pos = torch.tensor(self.positions, device=x.device)
        for unit, donor in items:
            donor = donor.to(x.device, x.dtype)
            if kind == "head":
                sl = slice(unit * self.hd, (unit + 1) * self.hd)
                x[0, pos, sl] = donor[:, sl]
            elif kind == "neuron" and unit is not None:
                idx = unit.to(x.device)
                x[0, pos[:, None], idx[None, :]] = donor[:, idx]
            else:
                x[0, pos, :] = donor
        return x

    def _pre(self, items, kind):
        def hook(mod, args):
            return (self._apply(args[0], items, kind),) + tuple(args[1:])
        return hook

    def _post(self, items):
        def hook(mod, args, output):
            return _with_tensor(output, self._apply(_out_tensor(output), items, "whole"))
        return hook


# ----------------------------------------------------------------------------------------------- core ops


def metric_from_logits(logits: torch.Tensor, pos_token: int, neg_token: int) -> torch.Tensor:
    last = logits[0, -1].float()
    return last[pos_token] - last[neg_token]


@torch.no_grad()
def run_metric(model, ids: torch.Tensor, pos_token: int, neg_token: int) -> float:
    out = model(input_ids=ids[None].to(input_device_of(model)))
    return float(metric_from_logits(out.logits, pos_token, neg_token))


@torch.no_grad()
def cache_run(model, ids: torch.Tensor, want: Iterable[str], positions: list[int]) -> tuple[float | None, dict]:
    """Forward once, return activations restricted to `positions` (cpu, float32)."""
    with Recorder(model, want) as rec:
        out = model(input_ids=ids[None].to(input_device_of(model)))
    pos = torch.tensor(positions)
    acts = {k: v[0, pos.to(v.device)].float().cpu() for k, v in rec.acts.items()}
    return out.logits, acts


def atp(model, pair: EncodedPair, kinds=("head", "mlp"), pos_mode: str = "suffix", max_suffix: int | None = None) -> dict:
    """Denoising AtP scores on one pair. Returns {'effect': E, 'm_clean', 'm_corrupt', ('head', l): [H], ...}."""
    pc, pk = pair.aligned_positions(pos_mode, max_suffix)
    clean_logits, clean_acts = cache_run(model, pair.clean_ids, kinds, pc)
    m_clean = float(metric_from_logits(clean_logits, pair.pos_token, pair.neg_token))

    emb = model.get_input_embeddings()(pair.corrupt_ids[None].to(input_device_of(model))).detach().requires_grad_(True)
    with torch.enable_grad(), Recorder(model, kinds, retain_grad=True) as rec:
        out = model(inputs_embeds=emb)
        m = metric_from_logits(out.logits, pair.pos_token, pair.neg_token)
        m.backward()
    m_corrupt = float(m.detach())
    pos = torch.tensor(pk)
    res: dict[Any, Any] = {"m_clean": m_clean, "m_corrupt": m_corrupt, "effect": m_clean - m_corrupt}
    H, hd = n_heads_of(model), head_dim_of(model)
    for (kind, li), a in rec.acts.items():
        g = a.grad[0, pos.to(a.device)].float().cpu()
        delta = clean_acts[(kind, li)] - a.detach()[0, pos.to(a.device)].float().cpu()
        prod = delta * g  # [n_pos, width]
        if kind == "head":
            res[(kind, li)] = prod.view(len(pk), H, hd).sum(dim=(0, 2))
        elif kind == "neuron":
            res[(kind, li)] = prod.sum(dim=0)
        else:
            res[(kind, li)] = prod.sum()
    model.zero_grad(set_to_none=True)
    return res


def components_from_scores(score_maps: list[dict], kind: str, n_layers: int) -> torch.Tensor:
    """Average per-pair score maps into a [n_layers, n_units] tensor for one component kind."""
    rows = []
    for li in range(n_layers):
        vals = [s[(kind, li)] for s in score_maps if (kind, li) in s]
        rows.append(torch.stack([v if v.dim() else v[None] for v in vals]).float().mean(0))
    return torch.stack(rows)


def top_k(scores: torch.Tensor, k: int, absolute: bool = False) -> list[tuple[int, int]]:
    s = scores.abs() if absolute else scores
    flat = s.flatten()
    idx = torch.topk(flat, min(k, flat.numel())).indices.tolist()
    W = scores.shape[1]
    return [(i // W, i % W) for i in idx]


def layer_matched_random(selected: list[tuple[int, int]], n_units: int, seed: int,
                         exclude: set | None = None) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    exclude = set(exclude or selected)
    per_layer = {}
    for li, _ in selected:
        per_layer[li] = per_layer.get(li, 0) + 1
    out = []
    for li, n in per_layer.items():
        pool = [u for u in range(n_units) if (li, u) not in exclude]
        out += [(li, u) for u in rng.sample(pool, min(n, len(pool)))]
    return out


@torch.no_grad()
def patch_recovery(model, pair: EncodedPair, kind: str, units: list[tuple[int, int]] | None,
                   pos_mode: str = "suffix", max_suffix: int | None = None,
                   cache: dict | None = None) -> dict:
    """Denoise the corrupt run by inserting clean activations of `units` (kind 'head'/'mlp'/'resid')."""
    pc, pk = pair.aligned_positions(pos_mode, max_suffix)
    if cache is None:
        _, clean_acts = cache_run(model, pair.clean_ids, [kind], pc)
        m_clean = run_metric(model, pair.clean_ids, pair.pos_token, pair.neg_token)
        m_corrupt = run_metric(model, pair.corrupt_ids, pair.pos_token, pair.neg_token)
    else:
        clean_acts, m_clean, m_corrupt = cache["acts"], cache["m_clean"], cache["m_corrupt"]
    spec: dict = {}
    if kind == "head":
        for li, h in units:
            spec.setdefault(("head", li), []).append((h, clean_acts[("head", li)]))
    elif kind in ("mlp", "resid"):
        for li, _ in units:
            spec[(kind, li)] = [(None, clean_acts[(kind, li)])]
    elif kind == "neuron":
        by_layer: dict[int, list[int]] = {}
        for li, n in units:
            by_layer.setdefault(li, []).append(n)
        for li, ns in by_layer.items():
            spec[("neuron", li)] = [(torch.tensor(ns), clean_acts[("neuron", li)])]
    with Patcher(model, spec, pk):
        m_patched = run_metric(model, pair.corrupt_ids, pair.pos_token, pair.neg_token)
    E = m_clean - m_corrupt
    return {"m_clean": m_clean, "m_corrupt": m_corrupt, "m_patched": m_patched,
            "recovery": (m_patched - m_corrupt) / E if abs(E) > 1e-6 else float("nan")}


def build_cache(model, pair: EncodedPair, kinds: Iterable[str], pos_mode: str = "suffix",
                max_suffix: int | None = None) -> dict:
    pc, _ = pair.aligned_positions(pos_mode, max_suffix)
    logits, acts = cache_run(model, pair.clean_ids, kinds, pc)
    return {"acts": acts, "m_clean": float(metric_from_logits(logits, pair.pos_token, pair.neg_token)),
            "m_corrupt": run_metric(model, pair.corrupt_ids, pair.pos_token, pair.neg_token)}


def jaccard(a: Iterable, b: Iterable) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def mean_ci(xs: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0}
    rng = random.Random(seed)
    mu = sum(xs) / len(xs)
    boots = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return {"n": len(xs), "mean": mu, "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]]}


@contextlib.contextmanager
def eval_mode(model):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


# ----------------------------------------------------------------------------------------------- direct logit attribution


@torch.no_grad()
def head_direct_logit_attribution(model, ids: torch.Tensor, pos_token: int, neg_token: int) -> torch.Tensor:
    """[n_layers, n_heads] direct contribution of each head (via o_proj, final RMSNorm scale frozen) to the metric
    at the last position. Multi-GPU safe under device_map='auto'."""
    with Recorder(model, ["head", "resid"]) as rec:
        model(input_ids=ids[None].to(input_device_of(model)))
    n_layers = len(layers_of(model))
    final_resid = rec.acts[("resid", n_layers - 1)][0, -1].float().cpu()  # pre-final-norm residual at last position
    norm = model.model.norm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    rms = torch.sqrt(final_resid.pow(2).mean() + eps)
    w_u = model.lm_head.weight
    direction_cpu = (w_u[pos_token].float().cpu() - w_u[neg_token].float().cpu()) * norm.weight.float().cpu() / rms
    H, hd = n_heads_of(model), head_dim_of(model)
    rows = []
    for li, layer in enumerate(layers_of(model)):
        a = rec.acts[("head", li)][0, -1].float()  # [H*hd] on layer's device
        W_o = layer.self_attn.o_proj.weight.float()  # [d_model, H*hd] on layer's device
        # (W_o_h @ a_h) . d == (W_o_h^T @ d) . a_h; stays on layer's device, then moves scalar vector to CPU
        d_dev = direction_cpu.to(W_o.device)
        proj = (W_o.t() @ d_dev) * a.to(W_o.device)  # [H*hd]
        rows.append(proj.view(H, hd).sum(dim=1).cpu())
    return torch.stack(rows)
