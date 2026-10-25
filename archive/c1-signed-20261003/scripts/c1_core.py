"""Signed C1: fixed random vectors follow dose sign; actual BF16 norm matching."""
from __future__ import annotations
import hashlib
import numpy as np
import torch

NORM_TOL = 0.005

def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("nonfinite or zero direction")
    return x / n

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))

def seed_for(family, j):
    return int(hashlib.sha256(f"20260928:{family}:{j}".encode()).hexdigest()[:16], 16)

def bf16_add(h, d, amplitude):
    return (h.float() + float(amplitude) * d.float()).to(torch.bfloat16)

def realized(h, d, amplitude):
    y = bf16_add(h, d, amplitude)
    return y, float(torch.linalg.vector_norm(y.float() - h.float()))

def matched_random(h, direction, target_norm):
    """Match on states only; never use model outputs or replace a direction."""
    d = torch.as_tensor(direction, dtype=torch.float32)
    if target_norm == 0:
        return h.clone(), 0.0, 0.0
    low, high = 0.0, max(float(target_norm) * 2, 1e-6)
    y, norm = realized(h, d, high)
    for _ in range(32):
        if norm >= target_norm:
            break
        high *= 2
        y, norm = realized(h, d, high)
    else:
        raise RuntimeError("cannot bracket realized random norm")
    best = (abs(norm-target_norm), y, high, norm)
    for _ in range(64):
        mid = (low+high)/2
        y, norm = realized(h, d, mid)
        if abs(norm-target_norm) < best[0]:
            best = (abs(norm-target_norm), y, mid, norm)
        if norm < target_norm:
            low = mid
        else:
            high = mid
    error, y, amp, norm = best
    if error / target_norm > NORM_TOL:
        raise RuntimeError(f"BF16 realized norm mismatch {error/target_norm:.6g}")
    return y, float(amp), float(norm)

def make_patch(target, donor, direction, sigma, operation, dose, random_dir=None):
    h = torch.as_tensor(target, dtype=torch.bfloat16)
    t = h.float().numpy().astype(np.float64)
    donor = np.asarray(donor, dtype=np.float64)
    d = unit(direction)
    rt, rd = rms(t), rms(donor)
    if rt <= 0 or rd <= 0:
        raise ValueError("zero RMS")
    component = float(np.dot(donor/rd - t/rt, d))
    amplitude = rt * (component if operation == "component" else float(dose)*sigma)
    true, true_norm = realized(h, torch.tensor(d, dtype=torch.float32), amplitude)
    if operation == "noop":
        y, applied_amp = h.clone(), 0.0
        reference_norm = 0.0
    elif operation == "replace":
        y, applied_amp = torch.as_tensor(donor, dtype=torch.bfloat16), None
        reference_norm = None
    elif operation in ("component", "dose"):
        reference_norm = true_norm
        if random_dir is None:
            y, applied_amp = true, amplitude
        else:
            sign = -1.0 if operation == 'dose' and dose < 0 else 1.0
            y, applied_amp, _ = matched_random(h, sign * unit(random_dir), true_norm)
    else:
        raise ValueError(operation)
    after = y.float().numpy()
    delta = after.astype(np.float64)-t
    norm = float(np.linalg.norm(delta))
    relative = None if reference_norm is None else (
        abs(norm-reference_norm)/reference_norm if reference_norm else abs(norm))
    if random_dir is not None and relative > NORM_TOL:
        raise RuntimeError("random norm check failed")
    along = float(delta @ d)
    stats = dict(target_rms=rt, donor_rms=rd, paired_unit_component=component,
        requested_true_raw_amplitude=amplitude, actual_raw_projection_change=along,
        actual_unit_rms_projection_change=float((after/rms(after)-t/rt)@d),
        reference_true_norm=reference_norm, realized_norm=norm,
        random_norm_relative_error=relative, applied_scalar=applied_amp,
        cosine_to_validity=float(along/norm) if norm else 0.0,
        sigma_train=sigma, operation=operation, dose=dose,
        random_direction_sign=(-1 if operation=='dose' and dose<0 else 1) if random_dir is not None else None,
        realized_over_target_norm=norm/float(np.linalg.norm(t)),
        realized_over_intended_norm=norm/abs(amplitude) if amplitude else None)
    return after, stats

def hook_inventory(model):
    return {name: [len(m._forward_hooks),len(m._forward_pre_hooks)]
            for name,m in model.named_modules()
            if m._forward_hooks or m._forward_pre_hooks}

class PromptPatch:
    def __init__(self, model, expected, replacement, prompt_tokens, prefix_tokens):
        self.model, self.expected, self.replacement = model, expected, replacement
        self.prompt_tokens, self.prefix_tokens = prompt_tokens, prefix_tokens
        self.calls = self.applied = 0
        self.handle = None
        self.before = self.after = None

    def __enter__(self):
        self.handle = self.model.model.layers[54].register_forward_pre_hook(self.hook)
        return self

    def hook(self, module, inputs):
        h = inputs[0]
        self.calls += 1
        if self.calls > 1:
            if h.shape != (1,1,len(self.expected)):
                raise RuntimeError("unexpected decode input shape")
            return None
        if h.shape != (1,self.prompt_tokens,len(self.expected)):
            raise RuntimeError(f"unexpected prefill shape {h.shape}")
        if h.dtype != torch.bfloat16:
            raise RuntimeError("model precision changed")
        self.before = h[0,-1].detach().float().cpu().numpy().copy()
        if np.max(np.abs(self.before-self.expected)) > 1e-4:
            raise RuntimeError("cached prompt state mismatch")
        out = h.clone()
        out[0,-1] = torch.as_tensor(self.replacement,device=h.device,dtype=h.dtype)
        self.after = out[0,-1].detach().float().cpu().numpy().copy()
        if not np.array_equal(self.after,self.replacement):
            raise RuntimeError("replacement state changed in transit")
        self.applied += 1
        return (out,*inputs[1:])

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if exc[0] is None and (self.applied != 1 or self.calls != self.prefix_tokens+1):
            raise RuntimeError(f"wrong hook schedule: calls={self.calls}, applied={self.applied}")

