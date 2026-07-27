"""Frozen contracts and deterministic helpers for RG1-P Stage A."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E2B_DATA = HERE.parent / "e2b_fixed_trace_cached"
E2B_V1_DATA = HERE.parent / "e2b_fixed_trace_cross"
E2A_DATA = HERE.parent / "e2_phase_validity_choice"
for _path in (E2B_DATA, E2B_V1_DATA, E2A_DATA):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import e2b_cached_common as e2b  # noqa: E402

SCHEMA = "cedar.rg1p_stagea.v1"
BRANCH = "agent/rg1p-stagea-fixed-trace-propagation"
BASE_RESULT_COMMIT = "6a7e286283cfc763b0dafec23d84fc91b031a99d"
INITIAL_CPU_GATE_COMMIT = "6f5f5dd643e774862a7185357b1d6027d5685c4f"
ENGINEERING_CORRECTION_ID = "RG1P_STAGEA_ENGINEERING_CORRECTION_01"
PROTOCOL = ROOT / "docs/RG1P_STAGEA_FIXED_TRACE_PROPAGATION_PROTOCOL.md"
SOURCE_TRACES = E2B_DATA.parent / "e2b_fixed_trace_cross/frozen_traces192.jsonl"
SOURCE_MATRIX = E2B_DATA.parent / "e2b_fixed_trace_cross/formal_matrix384.jsonl"
SOURCE_MANIFEST = E2B_DATA / "manifest.json"
SOURCE_RESULTS = (
    ROOT / "artifacts/m2/e2b_fixed_trace_cached_v1_1/formal-fixed-trace-seed0/results.jsonl"
)
SOURCE_AUDIT = ROOT / "artifacts/m2/e2b_fixed_trace_cached_v1_1/E2B_AUDIT.json"
SOURCE_ANALYSIS = ROOT / "artifacts/m2/e2b_fixed_trace_cached_v1_1/E2B_ANALYSIS.json"

MODEL_REPO = e2b.MODEL_REPO
MODEL_REVISION = e2b.MODEL_REVISION
MODEL_DTYPE = e2b.MODEL_DTYPE
ATTENTION_IMPLEMENTATION = e2b.ATTENTION_IMPLEMENTATION
CAP_EXPERIMENT = e2b.CAP_EXPERIMENT
CAP_LAYERS = tuple(e2b.CAP_LAYERS)
EXPECTED_AXIS_SHA256 = e2b.EXPECTED_AXIS_SHA256
EXPECTED_CAP_CONFIG_SHA256 = e2b.EXPECTED_CAP_CONFIG_SHA256
REPLAY_TOLERANCE = e2b.REPLAY_TOLERANCE

CAPTURE_LAYERS = tuple(range(64))
MEASURED_LAYERS = tuple(range(45, 64))
REPORTING_LAYERS = (45, 53, 54, 60, 63)
DOWNSTREAM_LAYERS = tuple(range(54, 64))
TRACE_SOURCES = tuple(e2b.TRACE_SOURCES)
OFFICIAL_CONDITIONS = tuple(e2b.FORWARD_CONDITIONS)
RANDOM_SEEDS = (20260727, 20260728)
RANDOM_SIGNS = ("pos", "neg")
RANDOM_CONDITIONS = tuple(
    f"random_s{seed}_{sign}" for seed in RANDOM_SEEDS for sign in RANDOM_SIGNS
)
NOOP_CONDITION = "uncapped_noop_repeat"
ALL_CONDITIONS = (*OFFICIAL_CONDITIONS, NOOP_CONDITION, *RANDOM_CONDITIONS)
PRIMARY_TRACE_SOURCE = "e2a_uncapped"

EXPECTED_PROMPTS = 96
EXPECTED_TRACES = 192
EXPECTED_OFFICIAL_FORMAL = 384
EXPECTED_RANDOM_FORMAL = 384
EXPECTED_NOOP_FORMAL = 96
EXPECTED_FORMAL_CELLS = 864
EXPECTED_SMOKE_PROMPTS = 8
EXPECTED_OFFICIAL_SMOKE = 32
EXPECTED_RANDOM_SMOKE = 32
EXPECTED_NOOP_SMOKE = 8
EXPECTED_SMOKE_CELLS = 72

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260727
LAYER_BOOTSTRAP_DRAWS = 10_000
PROBE_PERMUTATIONS = 4096
EPSILON = 1e-12
RANDOM_ORTHOGONAL_TOLERANCE = 1e-6
RANDOM_UNIT_TOLERANCE = 1e-6
# Requested correction vectors match the official norm in float32. The model-visible
# BF16 residual is additionally audited at its representable numerical resolution.
REALIZED_NORM_ABS_TOLERANCE = 0.03125
REALIZED_NORM_REL_TOLERANCE = 0.002
FINAL_LENS_LOGIT_TOLERANCE = 1e-3

canonical_text_sha256 = e2b.canonical_text_sha256
file_sha256 = e2b.file_sha256
object_sha256 = e2b.object_sha256
read_json = e2b.read_json
read_jsonl = e2b.read_jsonl
require = e2b.require
write_json = e2b.write_json


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def random_condition(seed: int, sign: str) -> str:
    require(seed in RANDOM_SEEDS, f"Unexpected random seed: {seed}")
    require(sign in RANDOM_SIGNS, f"Unexpected random sign: {sign}")
    return f"random_s{seed}_{sign}"


def parse_random_condition(condition: str) -> tuple[int, int]:
    require(condition in RANDOM_CONDITIONS, f"Not a frozen random condition: {condition}")
    _, seed_text, sign = condition.split("_")
    return int(seed_text.removeprefix("s")), 1 if sign == "pos" else -1


def build_random_directions(assistant_axis: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Build two fixed per-layer Gaussian directions orthogonal to official axes."""
    require(assistant_axis.ndim == 2, "Assistant Axis must be rank two")
    require(assistant_axis.shape[0] > max(CAP_LAYERS), "Assistant Axis layer coverage changed")
    rows: list[Tensor] = []
    metadata: dict[str, Any] = {"seeds": {}, "layers": list(CAP_LAYERS)}
    for seed in RANDOM_SEEDS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        layer_rows: list[Tensor] = []
        seed_meta: dict[str, Any] = {}
        for layer in CAP_LAYERS:
            axis = assistant_axis[layer].double()
            axis = axis / torch.linalg.vector_norm(axis)
            direction = torch.randn(axis.shape, generator=generator, dtype=torch.float64)
            direction = direction - torch.dot(direction, axis) * axis
            direction = direction / torch.linalg.vector_norm(direction)
            stored = direction.float()
            # Re-project after the float32 cast so the frozen representation itself
            # satisfies the preregistered orthogonality tolerance.
            stored64 = stored.double()
            stored64 = stored64 - torch.dot(stored64, axis) * axis
            stored64 = stored64 / torch.linalg.vector_norm(stored64)
            stored = stored64.float()
            dot = float(torch.dot(stored.double(), axis).item())
            norm = float(torch.linalg.vector_norm(stored.double()).item())
            require(abs(dot) <= RANDOM_ORTHOGONAL_TOLERANCE, "Random direction not orthogonal")
            require(abs(norm - 1.0) <= RANDOM_UNIT_TOLERANCE, "Random direction not unit norm")
            layer_rows.append(stored)
            seed_meta[str(layer)] = {
                "axis_dot": dot,
                "norm": norm,
                "sha256": tensor_sha256(stored),
            }
        packed = torch.stack(layer_rows)
        rows.append(packed)
        metadata["seeds"][str(seed)] = {
            "tensor_sha256": tensor_sha256(packed),
            "by_layer": seed_meta,
        }
    tensor = torch.stack(rows)
    metadata["tensor_sha256"] = tensor_sha256(tensor)
    metadata["shape"] = list(tensor.shape)
    metadata["construction"] = "gaussian_then_official_axis_orthogonalization"
    metadata["shared_across_items"] = True
    return tensor, metadata


def direction_map(directions: Tensor) -> dict[tuple[int, int], Tensor]:
    require(
        list(directions.shape) == [len(RANDOM_SEEDS), len(CAP_LAYERS), directions.shape[-1]],
        "Random direction tensor shape changed",
    )
    return {
        (seed, layer): directions[seed_index, layer_index]
        for seed_index, seed in enumerate(RANDOM_SEEDS)
        for layer_index, layer in enumerate(CAP_LAYERS)
    }


def smoke_prompt_ids(traces: Sequence[Mapping[str, Any]]) -> list[str]:
    # Reuse the already frozen E2B smoke selection exactly.
    manifest = read_json(SOURCE_MANIFEST)
    ids = list(manifest["smoke_prompt_ids"])
    require(len(ids) == EXPECTED_SMOKE_PROMPTS, "E2B smoke prompt count changed")
    available = {str(row["source_id"]) for row in traces}
    require(set(ids) <= available, "E2B smoke prompt IDs are missing")
    return ids


def matrix_cells(
    traces: Sequence[Mapping[str, Any]], *, scope: str, smoke_ids: Sequence[str]
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in traces]
    if scope == "smoke":
        allowed = set(smoke_ids)
        selected = [row for row in selected if row["source_id"] in allowed]
        require(len(selected) == 16, "Smoke must contain 8 prompts x 2 trace sources")
    elif scope != "formal":
        raise ValueError(f"Unknown scope: {scope}")

    official = [
        {
            "cell_kind": "official",
            "trace_id": trace["trace_id"],
            "source_id": trace["source_id"],
            "trace_source": trace["trace_source"],
            "forward_condition": condition,
        }
        for trace in selected
        for condition in OFFICIAL_CONDITIONS
    ]
    naive = [row for row in selected if row["trace_source"] == PRIMARY_TRACE_SOURCE]
    random_rows = [
        {
            "cell_kind": "random_reference",
            "trace_id": trace["trace_id"],
            "source_id": trace["source_id"],
            "trace_source": trace["trace_source"],
            "forward_condition": condition,
        }
        for trace in naive
        for condition in RANDOM_CONDITIONS
    ]
    noop_rows = [
        {
            "cell_kind": "engineering_noop",
            "trace_id": trace["trace_id"],
            "source_id": trace["source_id"],
            "trace_source": trace["trace_source"],
            "forward_condition": NOOP_CONDITION,
        }
        for trace in naive
    ]
    rows = [*official, *noop_rows, *random_rows]
    expected = EXPECTED_SMOKE_CELLS if scope == "smoke" else EXPECTED_FORMAL_CELLS
    require(len(rows) == expected, f"Unexpected {scope} cell count")
    return rows


def resolve_matrix(
    traces: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[tuple[dict[str, Any], str, str]]:
    by_id = {str(row["trace_id"]): dict(row) for row in traces}
    require(len(by_id) == len(traces), "Trace IDs are not unique")
    resolved = []
    for row in rows:
        trace = by_id[str(row["trace_id"])]
        require(trace["source_id"] == row["source_id"], "Matrix source ID drift")
        require(trace["trace_source"] == row["trace_source"], "Matrix trace-source drift")
        resolved.append((trace, str(row["forward_condition"]), str(row["cell_kind"])))
    return resolved


def family_bootstrap(
    scores: Mapping[str, float], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    families = sorted(scores)
    require(len(families) == 12, f"Expected 12 families, found {len(families)}")
    values = [float(scores[family]) for family in families]
    rng = random.Random(seed)
    samples = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(draws)
    ]
    ordered = sorted(samples)
    lower = ordered[math.floor(0.025 * draws)]
    upper = ordered[min(draws - 1, math.ceil(0.975 * draws) - 1)]
    return {
        "estimate": statistics.fmean(values),
        "family_cluster_bootstrap_95ci": [lower, upper],
        "families": len(families),
        "draws": draws,
        "seed": seed,
        "family_values": {family: float(scores[family]) for family in families},
        "probability_positive": sum(value > 0 for value in samples) / draws,
        "probability_negative": sum(value < 0 for value in samples) / draws,
    }


def paired_difference(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    require(set(left) == set(right), "Paired family sets differ")
    return {family: float(left[family]) - float(right[family]) for family in left}


def leave_one_family_out(scores: Mapping[str, float]) -> dict[str, float]:
    return {
        omitted: statistics.fmean(value for family, value in scores.items() if family != omitted)
        for omitted in sorted(scores)
    }
