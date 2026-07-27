#!/usr/bin/env python3
"""Run RG1-P Stage A cache-faithful propagation and random references."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "datasets/m2_2b_formal/rg1p_stagea"
E2B_DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cached"
E2B_V1_DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
E2A_DATA = ROOT / "datasets/m2_2b_formal/e2_phase_validity_choice"
for path in (SRC, DATA, E2B_DATA, E2B_V1_DATA, E2A_DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_e2_phase_validity_choice as e2run  # noqa: E402
import run_e2b_fixed_trace_cached as e2b_run  # noqa: E402
import run_e2b_fixed_trace_cross as e2b_v1_run  # noqa: E402
import run_prover_generator_step5_uncapped as base_run  # noqa: E402
from rg1p_common import (  # noqa: E402
    ATTENTION_IMPLEMENTATION,
    BASE_RESULT_COMMIT,
    BRANCH,
    CAP_LAYERS,
    CAPTURE_LAYERS,
    ENGINEERING_CORRECTION_02_ID,
    ENGINEERING_CORRECTION_ID,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_SMOKE_CELLS,
    INITIAL_CPU_GATE_COMMIT,
    MODEL_DTYPE,
    MODEL_REPO,
    MODEL_REVISION,
    NOOP_CONDITION,
    OFFICIAL_CONDITIONS,
    RANDOM_CONDITIONS,
    RANDOM_NORM_SEARCH_EXPANSIONS,
    RANDOM_NORM_SEARCH_ITERATIONS,
    REALIZED_NORM_ABS_TOLERANCE,
    REALIZED_NORM_REL_TOLERANCE,
    SOURCE_RESULTS,
    SOURCE_TRACES,
    canonical_text_sha256,
    direction_map,
    file_sha256,
    object_sha256,
    parse_random_condition,
    read_json,
    read_jsonl,
    require,
    resolve_matrix,
    tensor_sha256,
)

from cedar_m2.checkpoint import RecordStore, atomic_write_json, initialize_run  # noqa: E402
from cedar_m2.interventions.capping import ActivationCapper  # noqa: E402
from cedar_m2.interventions.phase_capping import PhaseActivationCapper  # noqa: E402
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/rg1p_stagea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME")),
    )
    parser.add_argument("--asset-cache-dir", type=Path)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def _command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True, text=True)


def git_provenance(*, require_synced: bool) -> dict[str, Any]:
    head = _command("rev-parse", "HEAD").stdout.strip()
    branch = _command("branch", "--show-current").stdout.strip()
    status = _command("status", "--porcelain").stdout.rstrip()
    upstream_result = _command("rev-parse", "@{u}", check=False)
    upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
    ancestor = _command("merge-base", "--is-ancestor", BASE_RESULT_COMMIT, head, check=False)
    require(ancestor.returncode == 0, "RG1-P HEAD must descend from frozen E2B result")
    require(branch == BRANCH, f"RG1-P requires branch {BRANCH}, found {branch}")
    if require_synced:
        require(not status, "RG1-P GPU runner requires a clean worktree")
        require(upstream == head, "RG1-P GPU runner requires HEAD == upstream")
    return {
        "commit": head,
        "branch": branch,
        "upstream_commit": upstream,
        "head_matches_upstream": upstream == head if upstream else False,
        "dirty": bool(status),
        "e2b_result_is_ancestor": True,
    }


def initialize_stagea_run(
    run_dir: Path, configuration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Initialize a run or resume the single documented smoke correction.

    The first 40 smoke forwards are immutable and remain bound to the initial
    CPU Gate fingerprint. Only the derived native-lens implementation changed.
    Reuse that fingerprint after proving that every scientific configuration
    field is identical and that the corrected commit descends from the initial
    Gate. Formal starts in a fresh directory and never uses this exception.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return initialize_run(run_dir, configuration), None
    existing = read_json(manifest_path)
    previous = existing.get("configuration")
    require(isinstance(previous, dict), "Existing run manifest is malformed")
    if previous == configuration:
        return initialize_run(run_dir, configuration), None

    require(configuration.get("scope") == "smoke", "Correction resume is smoke-only")
    previous_git = previous.get("git")
    current_git = configuration.get("git")
    require(
        isinstance(previous_git, dict) and isinstance(current_git, dict),
        "Correction resume Git provenance is malformed",
    )
    require(
        previous_git.get("commit") == INITIAL_CPU_GATE_COMMIT,
        "Correction resume requires the initial RG1-P CPU Gate run",
    )
    require(
        previous_git.get("branch") == current_git.get("branch") == BRANCH,
        "Correction resume branch changed",
    )
    require(
        _command(
            "merge-base",
            "--is-ancestor",
            INITIAL_CPU_GATE_COMMIT,
            str(current_git.get("commit")),
            check=False,
        ).returncode
        == 0,
        "Corrected execution commit does not descend from the initial CPU Gate",
    )
    previous_science = {key: value for key, value in previous.items() if key != "git"}
    current_science = {key: value for key, value in configuration.items() if key != "git"}
    # Correction 02 changes only the numerical solver used to realize the
    # already-frozen random schedule in BF16. The 40 imported official/no-op
    # records do not depend on this field.
    for science in (previous_science, current_science):
        random_reference = science.get("random_reference")
        if isinstance(random_reference, dict):
            random_reference.pop("bf16_norm_solver", None)
    require(previous_science == current_science, "Correction resume changed run configuration")
    initialized = initialize_run(run_dir, previous)
    return initialized, {
        "reason": ENGINEERING_CORRECTION_02_ID,
        "run_fingerprint_commit": INITIAL_CPU_GATE_COMMIT,
        "execution_commit": current_git["commit"],
    }


def validate_release(
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gate = read_json(DATA / "CPU_GATE.json")
    manifest = read_json(DATA / "manifest.json")
    require(
        gate.get("gpu_authorized") is True
        and gate.get("authorized_scope") == "RG1P_StageA_72_smoke_then_864_formal_only",
        "RG1-P CPU Gate does not authorize this run",
    )
    require(manifest.get("status") == "cpu_gate_ready", "RG1-P manifest is not ready")
    require(
        canonical_text_sha256(DATA / "manifest.json") == gate["manifest_canonical_lf_sha256"],
        "RG1-P manifest hash changed",
    )
    for relative, expected in manifest["release_files"].items():
        path = ROOT / relative
        require(path.is_file(), f"Missing release file: {relative}")
        observed = (
            file_sha256(path) if path.suffix == ".safetensors" else canonical_text_sha256(path)
        )
        require(observed == expected, f"Release file hash changed: {relative}")
    require(
        canonical_text_sha256(SOURCE_TRACES) == manifest["source_hashes"]["frozen_traces192.jsonl"],
        "Frozen E2B traces changed",
    )
    require(
        canonical_text_sha256(SOURCE_RESULTS) == manifest["source_hashes"]["e2b_results.jsonl"],
        "Frozen E2B results changed",
    )
    traces = read_jsonl(SOURCE_TRACES)
    matrix_path = DATA / ("smoke_matrix72.jsonl" if scope == "smoke" else "formal_matrix864.jsonl")
    matrix = read_jsonl(matrix_path)
    expected = EXPECTED_SMOKE_CELLS if scope == "smoke" else EXPECTED_FORMAL_CELLS
    require(len(matrix) == expected, f"Unexpected {scope} matrix size")
    return traces, matrix, manifest


def _tensor_from_output(output: object, layer: int) -> tuple[Tensor, str]:
    if isinstance(output, Tensor):
        return output, "tensor"
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0], "tuple"
    if isinstance(output, list) and output and isinstance(output[0], Tensor):
        return output[0], "list"
    raise TypeError(f"Unsupported decoder output at layer {layer}: {type(output)}")


def _replace_output(output: object, modified: Tensor, container: str) -> object:
    if container == "tensor":
        return modified
    if container == "tuple":
        assert isinstance(output, tuple)
        return (modified, *output[1:])
    assert isinstance(output, list)
    return [modified, *output[1:]]


def decoder_layers(model: Any) -> Sequence[nn.Module]:
    require(hasattr(model, "model") and hasattr(model.model, "layers"), "Qwen layers unavailable")
    layers = model.model.layers
    require(len(layers) > max(CAPTURE_LAYERS), "Decoder layer count changed")
    return layers


def hook_counts(model: Any) -> dict[str, int]:
    layers = decoder_layers(model)
    return {str(layer): len(layers[layer]._forward_hooks) for layer in CAPTURE_LAYERS}


class TerminalBlockCollector:
    """Capture post-intervention block outputs at the E2B terminal probe call."""

    def __init__(self, model: Any, terminal_call_index: int) -> None:
        self.model = model
        self.terminal_call_index = terminal_call_index
        self.calls = {layer: 0 for layer in CAPTURE_LAYERS}
        self.states: dict[int, Tensor] = {}
        self._handles: list[Any] = []

    def _hook(self, layer: int):
        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            tensor, _container = _tensor_from_output(output, layer)
            call = self.calls[layer]
            if call == self.terminal_call_index:
                require(tensor.ndim == 3 and tensor.shape[0] == 1, "Unexpected block tensor shape")
                self.states[layer] = tensor[0, -1].detach().to("cpu", dtype=torch.bfloat16)
            self.calls[layer] += 1
            return None

        return capture

    def __enter__(self) -> TerminalBlockCollector:
        layers = decoder_layers(self.model)
        self._handles = [
            layers[layer].register_forward_hook(self._hook(layer)) for layer in CAPTURE_LAYERS
        ]
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finalize(self) -> Tensor:
        expected_calls = self.terminal_call_index + 1
        require(
            all(value == expected_calls for value in self.calls.values()),
            f"Block call counts changed: {self.calls}",
        )
        require(set(self.states) == set(CAPTURE_LAYERS), "Terminal block states missing")
        return torch.stack([self.states[layer] for layer in CAPTURE_LAYERS])


class OfficialScheduleCapture:
    """Capture the exact official p25 correction norm at every causal position."""

    def __init__(self, model: Any, terminal_call_index: int) -> None:
        self.model = model
        self.terminal_call_index = terminal_call_index
        self.calls = {layer: 0 for layer in CAP_LAYERS}
        self._pending: dict[int, Tensor] = {}
        self._chunks: dict[int, list[Tensor]] = {layer: [] for layer in CAP_LAYERS}
        self.terminal_input: dict[int, Tensor] = {}
        self.terminal_output: dict[int, Tensor] = {}

    def _before(self, layer: int):
        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            tensor, _container = _tensor_from_output(output, layer)
            require(layer not in self._pending, f"Unconsumed cap input at layer {layer}")
            self._pending[layer] = tensor.detach()
            return None

        return capture

    def _after(self, layer: int):
        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            tensor, _container = _tensor_from_output(output, layer)
            before = self._pending.pop(layer)
            require(before.shape == tensor.shape, "Cap input/output shape mismatch")
            correction = tensor.float() - before.float()
            norms = torch.linalg.vector_norm(correction, dim=-1).reshape(-1).detach()
            self._chunks[layer].append(norms)
            call = self.calls[layer]
            if call == self.terminal_call_index:
                self.terminal_input[layer] = before[0, -1].to("cpu", dtype=torch.bfloat16)
                self.terminal_output[layer] = tensor[0, -1].detach().to("cpu", dtype=torch.bfloat16)
            self.calls[layer] += 1
            return None

        return capture

    @contextlib.contextmanager
    def before_context(self):
        layers = decoder_layers(self.model)
        handles = [layers[layer].register_forward_hook(self._before(layer)) for layer in CAP_LAYERS]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    @contextlib.contextmanager
    def after_context(self):
        layers = decoder_layers(self.model)
        handles = [layers[layer].register_forward_hook(self._after(layer)) for layer in CAP_LAYERS]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def finalize(self) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        expected_calls = self.terminal_call_index + 1
        require(not self._pending, "Unconsumed official cap inputs")
        require(
            all(value == expected_calls for value in self.calls.values()),
            f"Official schedule call counts changed: {self.calls}",
        )
        require(set(self.terminal_input) == set(CAP_LAYERS), "Terminal cap input missing")
        schedules = [torch.cat(self._chunks[layer]).float().cpu() for layer in CAP_LAYERS]
        lengths = {value.numel() for value in schedules}
        require(len(lengths) == 1, "Official schedules have unequal layer lengths")
        schedule = torch.stack(schedules)
        cap_input = torch.stack([self.terminal_input[layer] for layer in CAP_LAYERS])
        cap_output = torch.stack([self.terminal_output[layer] for layer in CAP_LAYERS])
        telemetry = {
            "positions_per_layer": schedule.shape[1],
            "triggered_by_layer": {
                str(layer): int(torch.count_nonzero(schedule[index] > 0).item())
                for index, layer in enumerate(CAP_LAYERS)
            },
            "correction_norm_sum_by_layer": {
                str(layer): float(schedule[index].sum().item())
                for index, layer in enumerate(CAP_LAYERS)
            },
            "schedule_sha256": tensor_sha256(schedule),
        }
        return schedule, cap_input, cap_output, telemetry


class ScheduleMatchedRandomInjector:
    """Inject fixed directions using an immutable official p25 norm schedule."""

    def __init__(
        self,
        model: Any,
        schedule: Tensor,
        directions: Mapping[int, Tensor],
        *,
        sign: int,
        terminal_call_index: int,
    ) -> None:
        require(tuple(schedule.shape[:1]) == (len(CAP_LAYERS),), "Schedule layer shape changed")
        require(sign in {-1, 1}, "Random sign must be +/-1")
        self.model = model
        self.schedule = schedule.float().cpu()
        self.directions = {layer: directions[layer].float().cpu() for layer in CAP_LAYERS}
        self.sign = sign
        self.terminal_call_index = terminal_call_index
        self.offsets = {layer: 0 for layer in CAP_LAYERS}
        self.calls = {layer: 0 for layer in CAP_LAYERS}
        self.requested_errors: dict[int, list[Tensor]] = {layer: [] for layer in CAP_LAYERS}
        self.realized_norms: dict[int, list[Tensor]] = {layer: [] for layer in CAP_LAYERS}
        self.realized_errors: dict[int, list[Tensor]] = {layer: [] for layer in CAP_LAYERS}
        self.realized_relative_errors: dict[int, list[Tensor]] = {layer: [] for layer in CAP_LAYERS}
        self.realized_direction_cosines: dict[int, list[Tensor]] = {
            layer: [] for layer in CAP_LAYERS
        }
        self.terminal_input: dict[int, Tensor] = {}
        self.terminal_output: dict[int, Tensor] = {}
        self._handles: list[Any] = []
        self._direction_cache: dict[tuple[int, torch.device], Tensor] = {}
        self._schedule_cache: dict[tuple[int, torch.device], Tensor] = {}

    def _hook(self, layer: int):
        layer_index = CAP_LAYERS.index(layer)

        def inject(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            tensor, container = _tensor_from_output(output, layer)
            require(tensor.ndim == 3 and tensor.shape[0] == 1, "Unexpected injection tensor")
            positions = tensor.shape[1]
            start = self.offsets[layer]
            stop = start + positions
            require(stop <= self.schedule.shape[1], "Random schedule overrun")
            cache_key = (layer, tensor.device)
            schedule_device = self._schedule_cache.get(cache_key)
            if schedule_device is None:
                schedule_device = self.schedule[layer_index].to(tensor.device)
                self._schedule_cache[cache_key] = schedule_device
            desired = schedule_device[start:stop]
            direction = self._direction_cache.get(cache_key)
            if direction is None:
                direction = self.directions[layer].to(tensor.device)
                direction = direction / torch.linalg.vector_norm(direction)
                direction = direction * self.sign
                self._direction_cache[cache_key] = direction

            before = tensor
            modified, actual = _bf16_norm_matched_update(before, desired, direction)
            # The ideal requested vector is c * unit(r). Compute its norm in
            # float64 so the engineering check does not mistake float32
            # reduction error at large c for a changed requested dose.
            unit64 = direction.detach().double()
            unit64 = unit64 / torch.linalg.vector_norm(unit64)
            commanded = desired.double().abs() * torch.linalg.vector_norm(unit64)
            requested_error = torch.abs(commanded - desired.double()).float()
            realized_error = torch.abs(actual - desired)
            relative = torch.zeros_like(realized_error)
            nonzero = desired > 0
            relative[nonzero] = realized_error[nonzero] / desired[nonzero]
            self.requested_errors[layer].append(requested_error.detach())
            self.realized_norms[layer].append(actual.detach())
            self.realized_errors[layer].append(realized_error.detach())
            self.realized_relative_errors[layer].append(relative.detach())
            realized_delta = (modified.float() - before.float()).reshape(-1, before.shape[-1])
            cosine = torch.zeros_like(actual)
            direction_nonzero = (desired > 0) & (actual > 0)
            cosine[direction_nonzero] = (
                realized_delta[direction_nonzero] @ direction
            ) / actual[direction_nonzero]
            self.realized_direction_cosines[layer].append(cosine[direction_nonzero].detach())

            call = self.calls[layer]
            if call == self.terminal_call_index:
                self.terminal_input[layer] = before[0, -1].detach().to("cpu", dtype=torch.bfloat16)
                self.terminal_output[layer] = (
                    modified[0, -1].detach().to("cpu", dtype=torch.bfloat16)
                )
            self.calls[layer] += 1
            self.offsets[layer] = stop
            return _replace_output(output, modified, container)

        return inject

    def __enter__(self) -> ScheduleMatchedRandomInjector:
        layers = decoder_layers(self.model)
        self._handles = [
            layers[layer].register_forward_hook(self._hook(layer)) for layer in CAP_LAYERS
        ]
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._direction_cache.clear()
        self._schedule_cache.clear()

    def finalize(self) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        expected_calls = self.terminal_call_index + 1
        require(
            all(value == expected_calls for value in self.calls.values()),
            f"Random injection call counts changed: {self.calls}",
        )
        require(
            all(value == self.schedule.shape[1] for value in self.offsets.values()),
            f"Random injection did not consume the full schedule: {self.offsets}",
        )
        require(set(self.terminal_input) == set(CAP_LAYERS), "Terminal injection input missing")
        requested = torch.cat(
            [torch.cat(self.requested_errors[layer]).float().cpu() for layer in CAP_LAYERS]
        )
        realized_norms = torch.stack(
            [torch.cat(self.realized_norms[layer]).float().cpu() for layer in CAP_LAYERS]
        )
        realized = torch.abs(realized_norms - self.schedule)
        relative = torch.cat(
            [torch.cat(self.realized_relative_errors[layer]).float().cpu() for layer in CAP_LAYERS]
        )
        cosine_chunks = [
            torch.cat(self.realized_direction_cosines[layer]).float().cpu()
            for layer in CAP_LAYERS
            if self.realized_direction_cosines[layer]
            and any(chunk.numel() for chunk in self.realized_direction_cosines[layer])
        ]
        direction_cosines = torch.cat(cosine_chunks) if cosine_chunks else torch.ones(1)
        max_abs = float(realized.max().item()) if realized.numel() else 0.0
        max_rel = float(relative.max().item()) if relative.numel() else 0.0
        input_tensor = torch.stack([self.terminal_input[layer] for layer in CAP_LAYERS])
        output_tensor = torch.stack([self.terminal_output[layer] for layer in CAP_LAYERS])
        return (
            input_tensor,
            output_tensor,
            realized_norms,
            {
                "schedule_sha256": tensor_sha256(self.schedule),
                "realized_norms_sha256": tensor_sha256(realized_norms),
                "positions_per_layer": self.schedule.shape[1],
                "triggered_by_layer": {
                    str(layer): int(torch.count_nonzero(self.schedule[index] > 0).item())
                    for index, layer in enumerate(CAP_LAYERS)
                },
                "requested_norm_max_abs_error": float(requested.max().item()),
                "realized_norm_max_abs_error": max_abs,
                "realized_norm_max_relative_error": max_rel,
                "realized_norm_within_frozen_bf16_tolerance": (
                    max_abs <= REALIZED_NORM_ABS_TOLERANCE or max_rel <= REALIZED_NORM_REL_TOLERANCE
                ),
                "bf16_norm_solver": (
                    f"monotone_bisection_e{RANDOM_NORM_SEARCH_EXPANSIONS}"
                    f"_i{RANDOM_NORM_SEARCH_ITERATIONS}"
                ),
                "realized_direction_cosine_min": float(direction_cosines.min().item()),
                "realized_direction_cosine_mean": float(direction_cosines.mean().item()),
            },
        )


def _bf16_norm_matched_update(
    before: Tensor, desired: Tensor, direction: Tensor
) -> tuple[Tensor, Tensor]:
    """Find the closest BF16-visible dose along a fixed continuous direction.

    For a fixed direction each coordinate's rounded displacement magnitude is
    monotone in the scalar dose. A per-position bracket and bisection therefore
    finds a representable residual update closer to the frozen official norm
    than the previous four-step ratio iteration, without changing the layer,
    position, direction, target norm, or model dtype.
    """
    require(before.ndim == 3 and before.shape[0] == 1, "Unexpected BF16 solver tensor")
    require(desired.ndim == 1 and desired.shape[0] == before.shape[1], "Dose shape changed")
    before_float = before.float()

    def evaluate(scale: Tensor) -> tuple[Tensor, Tensor]:
        delta = scale.view(1, -1, 1) * direction.view(1, 1, -1)
        candidate = (before_float + delta).to(before.dtype)
        norm = torch.linalg.vector_norm(candidate.float() - before_float, dim=-1).reshape(-1)
        return candidate, norm

    low = torch.zeros_like(desired)
    high = desired.clone()
    best_scale = low.clone()
    best_error = desired.clone()

    def consider(scale: Tensor, norm: Tensor) -> None:
        nonlocal best_scale, best_error
        error = torch.abs(norm - desired)
        improved = error < best_error
        best_scale = torch.where(improved, scale, best_scale)
        best_error = torch.where(improved, error, best_error)

    for _ in range(RANDOM_NORM_SEARCH_EXPANSIONS):
        _candidate, norm = evaluate(high)
        consider(high, norm)
        high = torch.where((desired > 0) & (norm < desired), high * 2, high)

    for _ in range(RANDOM_NORM_SEARCH_ITERATIONS):
        midpoint = (low + high) / 2
        _candidate, norm = evaluate(midpoint)
        consider(midpoint, norm)
        below = norm < desired
        low = torch.where(below, midpoint, low)
        high = torch.where(below, high, midpoint)

    for endpoint in (low, high):
        _candidate, norm = evaluate(endpoint)
        consider(endpoint, norm)
    modified, actual = evaluate(best_scale)
    return modified, actual


def _atomic_save_tensors(path: Path, tensors: Mapping[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()}, temporary
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_filename(cell_id: str) -> str:
    return f"{object_sha256(cell_id)}.safetensors"


def _anchor_key(trace: Mapping[str, Any], condition: str) -> tuple[str, str, str]:
    anchor_condition = "uncapped" if condition == NOOP_CONDITION else condition
    return str(trace["source_id"]), str(trace["trace_source"]), anchor_condition


def run_one(
    model: Any,
    tokenizer: Any,
    transformers_module: Any,
    caps: Mapping[int, Any],
    random_directions: Mapping[tuple[int, int], Tensor],
    prompt_rows: Mapping[str, Mapping[str, Any]],
    anchor_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    trace: Mapping[str, Any],
    condition: str,
    run_dir: Path,
) -> dict[str, Any]:
    prompt_row = prompt_rows[str(trace["source_id"])]
    prompt = base_run.render_prompt(tokenizer, prompt_row, base_run.MODES["thinking"])
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    prefix_ids = [int(value) for value in trace["trace_prefix_token_ids"]]
    require(object_sha256(prompt) == trace["prompt_sha256"], "Prompt hash drift")
    require(object_sha256(prompt_ids) == trace["prompt_token_ids_sha256"], "Prompt-token drift")
    require(object_sha256(prefix_ids) == trace["trace_prefix_token_ids_sha256"], "Trace drift")
    require(
        object_sha256([*prompt_ids, *prefix_ids]) == trace["teacher_forced_input_token_ids_sha256"],
        "Prompt-plus-trace drift",
    )
    encoded = {name: value.to(model_input_device(model)) for name, value in encoded.items()}
    state = e2b_run.ProcessorOrderState(prompt_tokens=len(prompt_ids), prefix_ids=prefix_ids)
    capture = e2b_run.RawAnswerCapture(
        state, lambda scores: e2b_v1_run.diagnostics(scores, trace, tokenizer)
    )
    forcer = e2b_run.FrozenTokenForcer(
        state, terminal_probe_token_id=int(trace["generated_label_token_id"])
    )
    before_hooks = hook_counts(model)
    monitor = PhaseActivationCapper(
        model,
        caps,
        enabled_phases=("prefill", "decode"),
        apply_changes=False,
        defer_stats_sync=True,
    )
    block_collector = TerminalBlockCollector(model, len(prefix_ids))
    schedule_capture: OfficialScheduleCapture | None = None
    capper: ActivationCapper | None = None
    random_injector: ScheduleMatchedRandomInjector | None = None
    tensors: dict[str, Tensor] = {}
    schedule_source_cell: str | None = None

    if condition == "full_p25":
        schedule_capture = OfficialScheduleCapture(model, len(prefix_ids))
        capper = ActivationCapper(
            model, caps, positions="all", track_stats=True, defer_stats_sync=True
        )
    elif condition in RANDOM_CONDITIONS:
        require(trace["trace_source"] == "e2a_uncapped", "Random reference on post-treatment trace")
        seed, sign = parse_random_condition(condition)
        schedule_source_cell = f"{trace['trace_id']}::full_p25"
        schedule_path = run_dir / "tensors" / tensor_filename(schedule_source_cell)
        require(schedule_path.is_file(), "Official p25 schedule missing before random reference")
        source_tensors = load_file(schedule_path, device="cpu")
        require("official_schedule_norms" in source_tensors, "Official schedule tensor missing")
        schedule = source_tensors["official_schedule_norms"].float()
        directions = {layer: random_directions[(seed, layer)] for layer in CAP_LAYERS}
        random_injector = ScheduleMatchedRandomInjector(
            model,
            schedule,
            directions,
            sign=sign,
            terminal_call_index=len(prefix_ids),
        )
    elif condition not in {"uncapped", NOOP_CONDITION}:
        raise ValueError(f"Unknown RG1-P condition: {condition}")

    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": len(prefix_ids) + 1,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "output_logits": False,
        "logits_processor": transformers_module.LogitsProcessorList([capture, forcer]),
    }
    transformers_module.set_seed(int(trace["anchor_record_seed"]))
    with torch.inference_mode(), ExitStack() as stack:
        stack.enter_context(monitor)
        if schedule_capture is not None:
            stack.enter_context(schedule_capture.before_context())
            assert capper is not None
            stack.enter_context(capper)
            stack.enter_context(schedule_capture.after_context())
        if random_injector is not None:
            stack.enter_context(random_injector)
        stack.enter_context(block_collector)
        generated = model.generate(**encoded, **generation_kwargs)

    replay_ids = [int(value) for value in generated.sequences[0, len(prompt_ids) :].tolist()]
    require(replay_ids[:-1] == prefix_ids, "Forced replay emitted a changed trace")
    require(replay_ids[-1] == int(trace["generated_label_token_id"]), "Terminal label changed")
    require(capture.observed is not None, "Missing terminal raw logits")
    require(state.capture_calls == len(prefix_ids) + 1, "Capture call count changed")
    require(state.force_calls == len(prefix_ids) + 1, "Force call count changed")
    require(state.capture_call_index == len(prefix_ids), "Terminal probe boundary changed")
    require(not state.order_violations, "Capture-before-force ordering failed")
    tensors["block_outputs"] = block_collector.finalize()

    schedule_telemetry: dict[str, Any] | None = None
    injection_telemetry: dict[str, Any] | None = None
    if schedule_capture is not None:
        schedule, cap_input, cap_output, schedule_telemetry = schedule_capture.finalize()
        tensors["official_schedule_norms"] = schedule
        tensors["cap_input_terminal"] = cap_input
        tensors["cap_output_terminal"] = cap_output
    if random_injector is not None:
        (
            injection_input,
            injection_output,
            realized_norms,
            injection_telemetry,
        ) = random_injector.finalize()
        tensors["injection_input_terminal"] = injection_input
        tensors["injection_output_terminal"] = injection_output
        tensors["random_realized_norms"] = realized_norms

    after_hooks = hook_counts(model)
    require(before_hooks == after_hooks, f"Hook leak: before={before_hooks}, after={after_hooks}")
    cell_id = f"{trace['trace_id']}::{condition}"
    tensor_path = run_dir / "tensors" / tensor_filename(cell_id)
    _atomic_save_tensors(tensor_path, tensors)
    anchor = anchor_rows.get(_anchor_key(trace, condition))
    if condition in {*OFFICIAL_CONDITIONS, NOOP_CONDITION}:
        require(anchor is not None, "Frozen E2B anchor row missing")
    if condition in OFFICIAL_CONDITIONS:
        cell_kind = "official"
    elif condition == NOOP_CONDITION:
        cell_kind = "engineering_noop"
    else:
        cell_kind = "random_reference"
    result = {
        "schema": "cedar.rg1p_stagea.result.v1",
        "cell_id": cell_id,
        "cell_kind": cell_kind,
        "source_id": trace["source_id"],
        "trace_id": trace["trace_id"],
        "trace_source": trace["trace_source"],
        "source_condition": trace["source_condition"],
        "forward_condition": condition,
        "family_id": trace["family_id"],
        "base_id": trace["base_id"],
        "structure_class": trace["structure_class"],
        "variant": trace["variant"],
        "arm": trace["arm"],
        "initial_answer": trace["initial_answer"],
        "target_answer": trace["target_answer"],
        "gold_answer": trace["gold_answer"],
        "prompt_sha256": trace["prompt_sha256"],
        "prompt_token_ids_sha256": trace["prompt_token_ids_sha256"],
        "trace_prefix_text_sha256": trace["trace_prefix_text_sha256"],
        "trace_prefix_token_ids_sha256": trace["trace_prefix_token_ids_sha256"],
        "forced_trace_token_ids_sha256": object_sha256(replay_ids[:-1]),
        "prompt_tokens": len(prompt_ids),
        "trace_prefix_tokens": len(prefix_ids),
        "causal_schedule_positions": len(prompt_ids) + len(prefix_ids),
        "candidate_token_ids": trace["candidate_token_ids"],
        "terminal_probe_token_id": replay_ids[-1],
        "terminal_probe_label": trace["generated_label"],
        "terminal_probe": {
            "capture_call_index": state.capture_call_index,
            "capture_sequence_position": len(prompt_ids) + len(prefix_ids) - 1,
            "context_token_ids_sha256": trace["teacher_forced_input_token_ids_sha256"],
        },
        "processor_order": {
            "capture_calls": state.capture_calls,
            "force_calls": state.force_calls,
            "capture_before_force_all": not state.order_violations,
            "violations": state.order_violations,
        },
        "replay_engine": "model.generate_kv_cache_forced_tokens",
        "model_generate_used": True,
        "natural_generation": False,
        "generated_text": False,
        "logit_diagnostics": capture.observed,
        "e2b_anchor_logits": anchor["logit_diagnostics"] if anchor is not None else None,
        "monitor_telemetry": monitor.telemetry(),
        "full_cap_stats": e2b_v1_run.full_stats(capper.stats) if capper else None,
        "official_schedule": schedule_telemetry,
        "random_injection": injection_telemetry,
        "random_schedule_source_cell": schedule_source_cell,
        "axis_definition": {
            "layer45": "reporting_layer_without_authorized_axis_no_projection",
            "layers46_53": "official_layer_specific_assistant_axis",
            "layers54_63": "fixed_layer53_exit_axis",
        },
        "tensor_file": str(tensor_path.relative_to(run_dir)),
        "tensor_file_sha256": file_sha256(tensor_path),
        "tensor_keys": sorted(tensors),
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "hook_counts_before": before_hooks,
        "hook_counts_after": after_hooks,
        "retry_count": 0,
        "answer_repair_performed": False,
    }
    del generated, encoded, tensors
    return result


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_key(index: int, trace: Mapping[str, Any], condition: str) -> dict[str, Any]:
    return {
        "index": index,
        "trace_id": trace["trace_id"],
        "forward_condition": condition,
        "assay": "rg1p_stagea",
    }


def materialize(
    store: RecordStore,
    matrix: Sequence[tuple[dict[str, Any], str, str]],
    run_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for index, (trace, condition, _kind) in enumerate(matrix):
        payload = store.read(record_key(index, trace, condition))
        if payload is None:
            continue
        require(payload.get("index") == index, f"Checkpoint index mismatch: {index}")
        result = payload["result"]
        tensor_path = run_dir / result["tensor_file"]
        require(tensor_path.is_file(), f"Checkpoint tensor missing: {tensor_path}")
        require(file_sha256(tensor_path) == result["tensor_file_sha256"], "Tensor hash drift")
        results.append(result)
    atomic_write_jsonl(run_dir / "results.jsonl", results)
    return results


def _native_lens_logits(model: Any, states: Tensor) -> Tensor:
    """Apply the native lens with the terminal decode call's tensor shape.

    The checkpoint's BF16 LM head is shape-sensitive: applying it once to a
    ``[layers, hidden]`` matrix can select a different GEMM path from the
    terminal ``[1, 1, hidden]`` decode call.  Evaluate each captured layer as
    its own singleton sequence so the pre-registered final-layer anchor tests
    the same numerical path as the archived answer logits.
    """
    require(states.ndim == 2, "Native lens expects [layers, hidden] states")
    norm = model.model.norm
    head = model.lm_head
    norm_device = norm.weight.device
    head_device = head.weight.device
    outputs = []
    for state in states.unbind(dim=0):
        values = state.reshape(1, 1, -1).to(
            device=norm_device, dtype=norm.weight.dtype
        )
        normalized = norm(values)
        if normalized.device != head_device:
            normalized = normalized.to(head_device)
        outputs.append(head(normalized)[0, 0].float().cpu())
    return torch.stack(outputs)


def restricted_option_log_probs(logits: Tensor, candidate_ids: Sequence[int]) -> Tensor:
    require(logits.ndim == 1, "Restricted option softmax expects one logit vector")
    require(len(candidate_ids) == 4 and len(set(candidate_ids)) == 4, "Candidate IDs changed")
    selected = logits[list(candidate_ids)].float()
    return torch.log_softmax(selected, dim=-1)


def _lens_record_path(run_dir: Path, pair_id: str) -> Path:
    # Keep the failed batched-BF16 v1 readouts immutable as engineering
    # evidence. Shape-faithful records live in a distinct cache namespace.
    return run_dir / "lens_records_shape_faithful_v2" / f"{object_sha256(pair_id)}.json"


def _compute_lens_pair(
    model: Any,
    run_dir: Path,
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    pair_kind: str,
) -> dict[str, Any]:
    baseline_tensors = load_file(run_dir / str(baseline["tensor_file"]), device="cpu")
    treatment_tensors = load_file(run_dir / str(treatment["tensor_file"]), device="cpu")
    baseline_states = baseline_tensors["block_outputs"].float()
    treatment_states = treatment_tensors["block_outputs"].float()
    require(
        list(baseline_states.shape) == [len(CAPTURE_LAYERS), 5120]
        and treatment_states.shape == baseline_states.shape,
        "Full-depth lens state shape changed",
    )
    with torch.inference_mode():
        logits_baseline = _native_lens_logits(model, baseline_states)
        logits_treatment = _native_lens_logits(model, treatment_states)
    candidate_ids = {key: int(value) for key, value in baseline["candidate_token_ids"].items()}
    require(candidate_ids == {"A": 32, "B": 33, "C": 34, "D": 35}, "Candidate IDs changed")
    ordered_ids = [candidate_ids[label] for label in ("A", "B", "C", "D")]
    initial_index = ("A", "B", "C", "D").index(str(baseline["initial_answer"]))
    noninitial = [index for index in range(4) if index != initial_index]
    layer_metrics = []
    for layer in CAPTURE_LAYERS:
        base_logits = logits_baseline[layer]
        treated_logits = logits_treatment[layer]
        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        treated_log_probs = torch.log_softmax(treated_logits, dim=-1)
        base_probs = torch.exp(base_log_probs)
        kl_vocab = float(torch.sum(base_probs * (base_log_probs - treated_log_probs)).item())
        base_options = base_logits[ordered_ids]
        treated_options = treated_logits[ordered_ids]
        base_option_log_probs = restricted_option_log_probs(base_logits, ordered_ids)
        treated_option_log_probs = restricted_option_log_probs(treated_logits, ordered_ids)
        base_option_probs = torch.exp(base_option_log_probs)
        kl_opt = float(
            torch.sum(base_option_probs * (base_option_log_probs - treated_option_log_probs)).item()
        )
        base_margin = float(
            torch.mean(base_options[noninitial]).item() - base_options[initial_index].item()
        )
        treated_margin = float(
            torch.mean(treated_options[noninitial]).item() - treated_options[initial_index].item()
        )
        base_state = baseline_states[layer]
        treated_state = treatment_states[layer]
        base_norm = float(torch.linalg.vector_norm(base_state).item())
        treated_norm = float(torch.linalg.vector_norm(treated_state).item())
        delta_norm = float(torch.linalg.vector_norm(treated_state - base_state).item())
        cosine = float(
            torch.dot(base_state, treated_state).item() / (base_norm * treated_norm + 1e-12)
        )
        layer_metrics.append(
            {
                "layer": layer,
                "kl_opt": kl_opt,
                "kl_vocab": kl_vocab,
                "m_general_mean_noninitial_uncapped": base_margin,
                "m_general_mean_noninitial_treatment": treated_margin,
                "delta_m_general_mean_noninitial": treated_margin - base_margin,
                "delta_h_l2": delta_norm,
                "state_cosine": cosine,
                "uncapped_state_l2": base_norm,
                "treatment_state_l2": treated_norm,
            }
        )
    final_base = {
        label: float(logits_baseline[63, token_id].item())
        for label, token_id in candidate_ids.items()
    }
    final_treatment = {
        label: float(logits_treatment[63, token_id].item())
        for label, token_id in candidate_ids.items()
    }
    base_anchor = baseline["logit_diagnostics"]["logits"]
    treatment_anchor = treatment["logit_diagnostics"]["logits"]
    final_errors = {
        "uncapped": max(
            abs(final_base[label] - float(base_anchor[label])) for label in candidate_ids
        ),
        "treatment": max(
            abs(final_treatment[label] - float(treatment_anchor[label])) for label in candidate_ids
        ),
    }
    return {
        "schema": "cedar.rg1p_stagea.layer_lens_pair.v2",
        "pair_id": f"{treatment['trace_id']}::{pair_kind}",
        "pair_kind": pair_kind,
        "source_id": treatment["source_id"],
        "trace_id": treatment["trace_id"],
        "trace_source": treatment["trace_source"],
        "treatment_condition": treatment["forward_condition"],
        "family_id": treatment["family_id"],
        "base_id": treatment["base_id"],
        "arm": treatment["arm"],
        "variant": treatment["variant"],
        "candidate_token_ids": candidate_ids,
        "mapping": "frozen_final_norm_then_lm_head_singleton_decode_shape",
        "translator_fitted": False,
        "final_layer_candidate_logit_max_abs_error": final_errors,
        "layers": layer_metrics,
    }


def materialize_layer_lens(
    model: Any, run_dir: Path, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_cell = {str(row["cell_id"]): row for row in rows}
    pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for treatment in rows:
        condition = treatment["forward_condition"]
        if condition == "full_p25":
            pair_kind = "official_p25"
        elif condition == NOOP_CONDITION:
            pair_kind = "engineering_noop"
        else:
            continue
        baseline = by_cell[f"{treatment['trace_id']}::uncapped"]
        pairs.append((baseline, treatment, pair_kind))

    records = []
    for index, (baseline, treatment, pair_kind) in enumerate(pairs):
        pair_id = f"{treatment['trace_id']}::{pair_kind}"
        path = _lens_record_path(run_dir, pair_id)
        if path.is_file():
            record = read_json(path)
        else:
            record = _compute_lens_pair(model, run_dir, baseline, treatment, pair_kind)
            atomic_write_json(path, record)
        records.append(record)
        print(
            json.dumps(
                {
                    "stage": "layer_lens",
                    "complete": index + 1,
                    "expected": len(pairs),
                    "pair_id": pair_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    csv_path = run_dir / "layer_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    fieldnames = [
        "pair_id",
        "pair_kind",
        "source_id",
        "trace_id",
        "trace_source",
        "treatment_condition",
        "family_id",
        "base_id",
        "arm",
        "variant",
        "layer",
        "kl_opt",
        "kl_vocab",
        "m_general_mean_noninitial_uncapped",
        "m_general_mean_noninitial_treatment",
        "delta_m_general_mean_noninitial",
        "delta_h_l2",
        "state_cosine",
        "uncapped_state_l2",
        "treatment_state_l2",
        "final_layer_uncapped_anchor_error",
        "final_layer_treatment_anchor_error",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            errors = record["final_layer_candidate_logit_max_abs_error"]
            for layer in record["layers"]:
                writer.writerow(
                    {
                        **{
                            key: record[key]
                            for key in (
                                "pair_id",
                                "pair_kind",
                                "source_id",
                                "trace_id",
                                "trace_source",
                                "treatment_condition",
                                "family_id",
                                "base_id",
                                "arm",
                                "variant",
                            )
                        },
                        **layer,
                        "final_layer_uncapped_anchor_error": errors["uncapped"],
                        "final_layer_treatment_anchor_error": errors["treatment"],
                    }
                )
    os.replace(temporary, csv_path)
    return records


def enforce_layer_lens_gate(
    records: Sequence[Mapping[str, Any]], *, expected_pairs: int
) -> None:
    """Fail before random references when native-lens or no-op controls drift."""
    require(len(records) == expected_pairs, "Layer-wise lens gate pair count changed")
    for record in records:
        errors = record["final_layer_candidate_logit_max_abs_error"]
        require(
            max(float(errors["uncapped"]), float(errors["treatment"])) < 1e-3,
            f"Final-layer native-lens anchor failed: {record['pair_id']}",
        )
        if record["pair_kind"] != "engineering_noop":
            continue
        for layer in record["layers"]:
            require(
                float(layer["kl_opt"]) < 1e-6 and float(layer["delta_h_l2"]) < 1e-4,
                f"Engineering no-op failed: {record['pair_id']} L{layer['layer']}",
            )


def main() -> None:
    args = parse_args()
    require(args.gpu_memory_gib == 18, "RG1-P requires 18 GiB per visible GPU")
    traces, matrix_rows, manifest = validate_release(args.scope)
    matrix = resolve_matrix(traces, matrix_rows)
    if args.scope == "formal" and args.stage in {"generate", "all"}:
        smoke_path = args.output_root / "SMOKE_AUDIT.json"
        require(smoke_path.is_file(), "Formal blocked until smoke audit exists")
        require(read_json(smoke_path).get("status") == "passed", "Formal blocked by smoke audit")
    git = git_provenance(require_synced=args.stage in {"generate", "all"})
    run_dir = args.output_root / f"{args.scope}-stagea-seed0"
    configuration = {
        "schema": "cedar.rg1p_stagea.run.v1",
        "scope": args.scope,
        "cells": len(matrix),
        "conditions": list(manifest["matrix"]["conditions"]),
        "model": manifest["model"],
        "p25": manifest["p25"],
        "replay_engine": manifest["replay_engine"],
        "random_reference": manifest["random_reference"],
        "git": git,
    }
    initialized, correction_resume = initialize_stagea_run(run_dir, configuration)
    store = RecordStore(run_dir / "records", run_fingerprint=str(initialized["fingerprint"]))
    results = materialize(store, matrix, run_dir)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "scope": args.scope,
                "cells": len(matrix),
                "existing": len(results),
                "gpu_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.stage == "prepare":
        return

    import transformers

    require(Path(sys.prefix).name == "cedar", f"GPU run requires cedar: {sys.prefix}")
    require(torch.cuda.device_count() == 4, "RG1-P requires exactly four visible GPUs")
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    cache = args.asset_cache_dir.resolve() if args.asset_cache_dir else args.hf_home.resolve()
    caps, cap_metadata = e2run.load_p25_assets(cache)
    require(cap_metadata["axis_sha256"] == EXPECTED_AXIS_SHA256, "Axis hash drift")
    require(
        cap_metadata["capping_config_sha256"] == EXPECTED_CAP_CONFIG_SHA256,
        "Capping config hash drift",
    )
    frozen_random = load_file(DATA / "random_directions.safetensors", device="cpu")["directions"]
    random_directions = direction_map(frozen_random)
    spec = QwenModelSpec(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )
    model, tokenizer = load_qwen_model(
        spec,
        max_memory={index: f"{args.gpu_memory_gib}GiB" for index in range(4)},
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_run.validate_label_tokens(tokenizer)
    prompt_rows = {row["id"]: row for row in read_jsonl(E2A_DATA / "answer_inputs96.jsonl")}
    anchor_rows = {
        (row["source_id"], row["trace_source"], row["forward_condition"]): row
        for row in read_jsonl(SOURCE_RESULTS)
    }
    runtime_path = run_dir / "runtime.json"
    if not runtime_path.is_file():
        atomic_write_json(
            runtime_path,
            {
                "generation_started_at": datetime.now().astimezone().isoformat(),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "visible_gpu_count": torch.cuda.device_count(),
                    "visible_gpu_names": [torch.cuda.get_device_name(index) for index in range(4)],
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                "official_p25": cap_metadata,
                "replay_engine": "model.generate_kv_cache_forced_tokens",
                "resume_events": [],
            },
        )
    if correction_resume is not None:
        runtime = read_json(runtime_path)
        events = list(runtime.get("resume_events", []))
        if not any(
            event.get("execution_commit") == correction_resume["execution_commit"]
            and event.get("reason") == correction_resume["reason"]
            for event in events
        ):
            events.append(
                {
                    **correction_resume,
                    "resumed_at": datetime.now().astimezone().isoformat(),
                    "completed_records_before_resume": len(results),
                    "completed_records_preserved": True,
                }
            )
            runtime["resume_events"] = events
            corrections = list(runtime.get("engineering_corrections", []))
            for correction in (ENGINEERING_CORRECTION_ID, correction_resume["reason"]):
                if correction not in corrections:
                    corrections.append(correction)
            runtime["engineering_corrections"] = corrections
            atomic_write_json(runtime_path, runtime)
    lens_gate_complete = False
    expected_lens_pairs = 24 if args.scope == "smoke" else 288
    for index, (trace, condition, kind) in enumerate(matrix):
        if kind == "random_reference" and not lens_gate_complete:
            results = materialize(store, matrix, run_dir)
            lens_records = materialize_layer_lens(model, run_dir, results)
            enforce_layer_lens_gate(lens_records, expected_pairs=expected_lens_pairs)
            lens_gate_complete = True
        key = record_key(index, trace, condition)
        if store.contains(key):
            continue
        result = run_one(
            model,
            tokenizer,
            transformers,
            caps,
            random_directions,
            prompt_rows,
            anchor_rows,
            trace,
            condition,
            run_dir,
        )
        store.write(key, {"index": index, "result": result})
        results = materialize(store, matrix, run_dir)
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "complete": len(results),
                    "expected": len(matrix),
                    "cell_kind": kind,
                    "source_id": trace["source_id"],
                    "trace_source": trace["trace_source"],
                    "forward_condition": condition,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    results = materialize(store, matrix, run_dir)
    lens_records = materialize_layer_lens(model, run_dir, results)
    enforce_layer_lens_gate(lens_records, expected_pairs=expected_lens_pairs)
    runtime = read_json(runtime_path)
    runtime.update(
        {
            "generation_completed_at": datetime.now().astimezone().isoformat(),
            "complete_results": len(results),
            "results_sha256": file_sha256(run_dir / "results.jsonl"),
            "tensor_files": len(list((run_dir / "tensors").glob("*.safetensors"))),
            "layer_lens_pairs": len(lens_records),
            "layer_metrics_sha256": file_sha256(run_dir / "layer_metrics.csv"),
            "nonfinite_logits": [
                row["cell_id"]
                for row in results
                if not all(
                    torch.isfinite(torch.tensor(value))
                    for value in row["logit_diagnostics"]["logits"].values()
                )
            ],
            "hook_leaks": [
                row["cell_id"]
                for row in results
                if row["hook_counts_before"] != row["hook_counts_after"]
            ],
        }
    )
    atomic_write_json(runtime_path, runtime)
    print(json.dumps({"complete": len(results), "expected": len(matrix)}), flush=True)


if __name__ == "__main__":
    main()
