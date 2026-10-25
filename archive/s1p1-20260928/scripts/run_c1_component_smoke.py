#!/usr/bin/env python3
"""Bounded C1 component-transfer smoke on one held-out family.

This is an engineering gate for the frozen C1 design. It is not a scientific
confirmation result. The script reuses the historical forced-replay runner and
adds one prompt-boundary layer-53 component patch per forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "CEDAR_SUPPORTING_DATA"
CODE = PACKAGE / "code"
for path in (CODE / "src", CODE / "scripts"):
    sys.path.insert(0, str(path))

import run_e2b_fixed_trace_cached as historical  # noqa: E402
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


class PromptBoundaryPatch:
    """Patch only the final prompt position at layer-54 input once."""

    def __init__(self, model, mode: str, vector: np.ndarray | None = None,
                 donor: np.ndarray | None = None, scale: float = 0.0):
        self.model = model
        self.mode = mode
        self.vector = vector
        self.donor = donor
        self.scale = float(scale)
        self.calls = 0
        self.modified = 0
        self.before = None
        self.after = None
        self.expected_norm = None
        self.handle = None

    def __enter__(self):
        layer = self.model.model.layers[54]
        self.handle = layer.register_forward_pre_hook(self._hook)
        return self

    def _hook(self, _module, inputs):
        if not inputs:
            raise RuntimeError("layer-54 pre-hook received no inputs")
        hidden = inputs[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"unexpected layer-54 input shape: {type(hidden)}")
        self.calls += 1
        if self.calls != 1:
            return inputs
        current = hidden[:, -1, :]
        self.before = current.detach().float().cpu().numpy().reshape(-1)
        if self.mode == "noop":
            updated = current
        elif self.mode == "replace":
            if self.donor is None:
                raise RuntimeError("replace requires donor state")
            updated = torch.as_tensor(self.donor, device=current.device,
                                      dtype=current.dtype).reshape(1, -1)
        elif self.mode in {"true", "random"}:
            if self.vector is None:
                raise RuntimeError(f"{self.mode} requires a vector")
            direction = torch.as_tensor(self.vector, device=current.device,
                                        dtype=torch.float32).reshape(1, -1)
            direction = direction / torch.linalg.vector_norm(direction)
            updated = current + self.scale * direction.to(current.dtype)
        else:
            raise ValueError(self.mode)
        out = hidden.clone()
        out[:, -1, :] = updated
        self.after = updated.detach().float().cpu().numpy().reshape(-1)
        self.modified = 1
        self.expected_norm = float(np.linalg.norm(self.after - self.before))
        return (out, *inputs[1:])

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def direction_path(family: str, source: str) -> Path:
    key = hashlib.sha256((family + source).encode()).hexdigest()[:16]
    return ROOT / "analysis-output/formal/p1/directions" / f"{key}.npy"


def choose_rows() -> tuple[list[dict], str]:
    records = json.loads((ROOT / "runs/formal/results.json").read_text())
    rows = [r for r in records if r["trace_source"] == "e2a_uncapped"
            and r["condition"] == "uncapped" and r["arm"] in {"V", "F"}]
    families = sorted({r["family_id"] for r in rows})
    if not families:
        raise RuntimeError("no e2a_uncapped V/F rows in formal results")
    family = families[0]
    selected = [r for r in rows if r["family_id"] == family]
    if len(selected) != 4:
        raise RuntimeError(f"expected 4 rows in first smoke family, got {len(selected)}")
    return selected, family


def main(args: argparse.Namespace) -> None:
    out = ROOT / "runs" / "c1_smoke"
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    assets = json.loads((ROOT / "explore_outputs/assets_status.json").read_text())
    if assets.get("status") != "ready":
        raise RuntimeError("verified assets are not ready")
    traces = {r["trace_id"]: r for r in read_jsonl(ROOT / "inputs/frozen_traces192.jsonl")}
    prompt_rows = {r["id"]: r for r in read_jsonl(PACKAGE / "data/phase_validity_inputs/answer_inputs96.jsonl")}
    records = json.loads((ROOT / "runs/formal/results.json").read_text())
    uncapped_records = [r for r in records if r["condition"] == "uncapped"]
    index_by_trace = {r["trace_id"]: i for i, r in enumerate(uncapped_records)}
    rows, family = choose_rows()
    write(out / "manifest.json", {
        "status": "engineering_smoke_only",
        "family": family,
        "rows": [r["trace_id"] for r in rows],
        "directions": {source: sha(direction_path(family, source))
                       for source in ("U", "P")},
        "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)),
        "model_revision": assets["revision"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "controls": ["noop", "true", "random", "replace"],
        "transfers": ["U->P", "P->U"],
    })
    write(status_path, {"stage": "loading_model", "family": family, "completed": 0})

    caps, cap_metadata = historical.e2run.load_p25_assets(ROOT / "cache/hub")
    model, tokenizer = load_qwen_model(
        QwenModelSpec(repo_id=assets["model"]),
        max_memory={i: "18GiB" for i in range(torch.cuda.device_count())},
        local_files_only=True,
    )
    if any(str(d) in {"cpu", "disk"} for d in model.hf_device_map.values()):
        raise RuntimeError(f"unexpected model offload: {model.hf_device_map}")
    if len(model.model.layers) != 64:
        raise RuntimeError("expected 64 decoder layers")

    rng = np.random.default_rng(20260928)
    results = []
    total = len(rows) * 2 * 4
    for row in rows:
        trace = traces[row["trace_id"]]
        idx = index_by_trace[trace["trace_id"]]
        target_states = {
            "U": dict(np.load(ROOT / "runs/formal" / f"{idx:04d}_uncapped.npz")),
            "P": dict(np.load(ROOT / "runs/formal" / f"{idx:04d}_full_p25.npz")),
        }
        for transfer in ("U->P", "P->U"):
            donor_condition, target_condition = transfer.split("->")
            donor = target_states[donor_condition]["prompt_layer_53"]
            target = target_states[target_condition]["prompt_layer_53"]
            source = donor_condition
            dpath = direction_path(family, source)
            direction = np.load(dpath).astype(np.float32)
            if direction.shape != target.shape:
                raise RuntimeError(f"direction shape mismatch: {direction.shape} vs {target.shape}")
            diff = donor.astype(np.float32) - target.astype(np.float32)
            true_scalar = float(np.dot(diff, direction))
            true_delta_norm = abs(true_scalar)
            random_direction = rng.normal(size=direction.shape).astype(np.float32)
            random_direction /= np.linalg.norm(random_direction)
            cells = [("noop", None, 0.0), ("true", direction, true_scalar),
                     ("random", random_direction, true_delta_norm), ("replace", None, 0.0)]
            for control, vector, scale in cells:
                name = f"{row['trace_id']}__{transfer.replace('>', '')}__{control}"
                result_path = out / f"{len(results):03d}_{name}.json"
                if result_path.exists():
                    results.append(json.loads(result_path.read_text()))
                    continue
                hook = PromptBoundaryPatch(model, control, vector=vector,
                                           donor=donor, scale=scale)
                started = time.monotonic()
                with hook:
                    replay = historical.run_one(model, tokenizer, __import__("transformers"),
                                                caps, prompt_rows, trace, "full_p25" if target_condition == "P" else "uncapped")
                elapsed = time.monotonic() - started
                payload = {
                    "schema": "cedar.c1.component_smoke.v1",
                    "status": "engineering_smoke_only",
                    "trace_id": trace["trace_id"], "family_id": family,
                    "arm": trace["arm"], "base_id": trace["base_id"],
                    "transfer": transfer, "donor_condition": donor_condition,
                    "target_condition": target_condition, "control": control,
                    "direction_source": source,
                    "target_forward_condition": "full_p25" if target_condition == "P" else "uncapped",
                    "hook_calls": hook.calls, "modified_positions": hook.modified,
                    "patch_norm": hook.expected_norm,
                    "true_projection_scalar": true_scalar,
                    "true_delta_norm": true_delta_norm,
                    "seconds": elapsed,
                    "logit_diagnostics": replay["logit_diagnostics"],
                    "historical_option_max_abs": replay.get("historical_option_max_abs"),
                    "hook_counts_before": replay["hook_counts_before"],
                    "hook_counts_after": replay["hook_counts_after"],
                    "cap_stats": replay.get("full_cap_stats"),
                }
                write(result_path, payload)
                results.append(payload)
                write(status_path, {"stage": "running", "family": family,
                                    "completed": len(results), "total": total,
                                    "last_cell": name, "last_seconds": elapsed})
                print(json.dumps({"completed": len(results), "total": total,
                                  "cell": name, "seconds": elapsed,
                                  "patch_norm": hook.expected_norm}), flush=True)
    write(out / "results.json", {"status": "engineering_smoke_only", "family": family,
                                  "n": len(results), "results": results,
                                  "cap_metadata": cap_metadata})
    failures = [r for r in results if r["hook_calls"] != 1 or r["modified_positions"] != 1
                or r["hook_counts_before"] != r["hook_counts_after"]]
    gate = {"status": "passed" if not failures else "failed",
            "technical_pass": not failures, "n": len(results),
            "failure_cells": [r["trace_id"] + ":" + r["transfer"] + ":" + r["control"] for r in failures],
            "note": "Engineering smoke only; no C1 scientific claim."}
    write(out / "gate.json", gate)
    write(status_path, {"stage": "complete", "family": family, "completed": len(results),
                        "total": total, "gate": gate})
    if failures:
        raise RuntimeError(f"C1 smoke gate failed: {gate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        write(ROOT / "runs/c1_smoke/failure.json", {"error": repr(exc),
             "traceback": traceback.format_exc(), "time": time.time()})
        raise
