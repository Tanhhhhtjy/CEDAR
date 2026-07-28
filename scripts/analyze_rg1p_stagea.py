#!/usr/bin/env python3
"""Frozen family-clustered analysis for RG1-P Stage A."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/rg1p_stagea"
E2A_DATA = ROOT / "datasets/m2_2b_formal/e2_phase_validity_choice"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from rg1p_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CAP_LAYERS,
    CAPTURE_LAYERS,
    EPSILON,
    LAYER_BOOTSTRAP_DRAWS,
    MEASURED_LAYERS,
    NOOP_CONDITION,
    PRIMARY_TRACE_SOURCE,
    PROBE_PERMUTATIONS,
    RANDOM_CONDITIONS,
    direction_map,
    family_bootstrap,
    file_sha256,
    leave_one_family_out,
    paired_difference,
    parse_random_condition,
    read_json,
    read_jsonl,
    require,
    write_json,
    write_jsonl,
)

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/rg1p_stagea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def geometry(row: Mapping[str, Any]) -> dict[str, float]:
    logits = {key: float(value) for key, value in row["logit_diagnostics"]["logits"].items()}
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    alternatives = [value for label, value in logits.items() if label != initial]
    background = [value for label, value in logits.items() if label not in {initial, target}]
    return {
        "m_general": max(alternatives) - logits[initial],
        "m_target": logits[target] - logits[initial],
        "m_background": max(background) - logits[initial],
        "m_specific": logits[target] - max(background),
    }


def family_scores(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(float(row[metric]))
    scores = {family: statistics.fmean(values) for family, values in grouped.items()}
    require(len(scores) == 12, f"Expected 12 family scores for {metric}, found {len(scores)}")
    return scores


def projection_metrics(
    delta: torch.Tensor, baseline: torch.Tensor, axis: torch.Tensor
) -> dict[str, float]:
    delta = delta.float()
    baseline = baseline.float()
    axis = axis.float() / torch.linalg.vector_norm(axis.float())
    signed = float(torch.dot(delta, axis).item())
    parallel_norm = abs(signed)
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    off = delta - signed * axis
    off_norm = float(torch.linalg.vector_norm(off).item())
    denominator = delta_norm * delta_norm + EPSILON
    parallel_share = parallel_norm * parallel_norm / denominator
    off_share = off_norm * off_norm / denominator
    return {
        "delta_norm": delta_norm,
        "signed_aligned_projection": signed,
        "aligned_norm": parallel_norm,
        "off_direction_norm": off_norm,
        "aligned_share": parallel_share,
        "off_direction_share": off_share,
        "share_sum": parallel_share + off_share,
        "relative_delta": delta_norm / (float(torch.linalg.vector_norm(baseline).item()) + EPSILON),
    }


def layer_axis(
    condition: str,
    layer: int,
    official_axes: torch.Tensor,
    random_directions: Mapping[tuple[int, int], torch.Tensor],
) -> tuple[torch.Tensor | None, str]:
    if layer == 45:
        return None, "reporting_layer_without_authorized_axis_no_projection"
    if condition == "full_p25":
        if layer <= 53:
            return official_axes[CAP_LAYERS.index(layer)], "official_layer_specific_axis"
        return official_axes[CAP_LAYERS.index(53)], "fixed_layer53_exit_axis"
    seed, sign = parse_random_condition(condition)
    if layer <= 53:
        axis = random_directions[(seed, layer)]
        name = "signed_layer_random_injection_direction"
    else:
        axis = random_directions[(seed, 53)]
        name = "fixed_signed_layer53_injection_direction"
    return axis * sign, name


def _tensor_path(run_dir: Path, row: Mapping[str, Any]) -> Path:
    return run_dir / str(row["tensor_file"])


def build_metrics(
    rows: Sequence[dict[str, Any]], run_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    official_axes = load_file(DATA / "official_axis_units.safetensors", device="cpu")[
        "official_axis_units"
    ].float()
    random_tensor = load_file(DATA / "random_directions.safetensors", device="cpu")[
        "directions"
    ].float()
    random_directions = direction_map(random_tensor)
    by_cell = {row["cell_id"]: row for row in rows}
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    ordered_blocks: list[torch.Tensor] = []
    for row in rows:
        payload = load_file(_tensor_path(run_dir, row), device="cpu")
        tensors[row["cell_id"]] = payload
        ordered_blocks.append(payload["block_outputs"].to(torch.bfloat16))
    aggregate = {"block_outputs": torch.stack(ordered_blocks)}

    contrast_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for treatment in rows:
        condition = treatment["forward_condition"]
        if condition in {"uncapped", NOOP_CONDITION}:
            continue
        baseline_id = f"{treatment['trace_id']}::uncapped"
        baseline = by_cell.get(baseline_id)
        require(baseline is not None, f"Missing uncapped baseline: {baseline_id}")
        baseline_states = tensors[baseline_id]["block_outputs"].float()
        treatment_states = tensors[treatment["cell_id"]]["block_outputs"].float()
        base_geometry = geometry(baseline)
        treated_geometry = geometry(treatment)
        contrast = {
            "contrast_id": treatment["cell_id"],
            "source_id": treatment["source_id"],
            "trace_id": treatment["trace_id"],
            "trace_source": treatment["trace_source"],
            "treatment_condition": condition,
            "cell_kind": treatment["cell_kind"],
            "family_id": treatment["family_id"],
            "base_id": treatment["base_id"],
            "arm": treatment["arm"],
            "variant": treatment["variant"],
            "delta_m_general": treated_geometry["m_general"] - base_geometry["m_general"],
            "delta_m_target": treated_geometry["m_target"] - base_geometry["m_target"],
            "delta_m_background": treated_geometry["m_background"] - base_geometry["m_background"],
            "delta_m_specific": treated_geometry["m_specific"] - base_geometry["m_specific"],
            "delta_option_mass": float(treatment["logit_diagnostics"]["option_mass"])
            - float(baseline["logit_diagnostics"]["option_mass"]),
            "delta_entropy": float(treatment["logit_diagnostics"]["full_vocab_entropy"])
            - float(baseline["logit_diagnostics"]["full_vocab_entropy"]),
            "top_token_is_option": bool(treatment["logit_diagnostics"]["top_token_is_option"]),
            "option_argmax_changed": treatment["logit_diagnostics"]["option_argmax"]
            != baseline["logit_diagnostics"]["option_argmax"],
        }
        by_layer: dict[int, dict[str, float | None]] = {}
        for layer in MEASURED_LAYERS:
            layer_index = CAPTURE_LAYERS.index(layer)
            delta = treatment_states[layer_index] - baseline_states[layer_index]
            axis, axis_name = layer_axis(condition, layer, official_axes, random_directions)
            if axis is None:
                delta_norm = float(torch.linalg.vector_norm(delta.float()).item())
                values: dict[str, float | None] = {
                    "delta_norm": delta_norm,
                    "signed_aligned_projection": None,
                    "aligned_norm": None,
                    "off_direction_norm": None,
                    "aligned_share": None,
                    "off_direction_share": None,
                    "share_sum": None,
                    "relative_delta": delta_norm
                    / (
                        float(
                            torch.linalg.vector_norm(
                                baseline_states[layer_index].float()
                            ).item()
                        )
                        + EPSILON
                    ),
                }
            else:
                values = projection_metrics(delta, baseline_states[layer_index], axis)
                if float(values["delta_norm"]) > 1e-8 and abs(
                    float(values["share_sum"]) - 1.0
                ) > 2e-4:
                    raise ValueError(
                        f"Projection decomposition failed: {treatment['cell_id']} L{layer}"
                    )
            by_layer[layer] = values
            metric_rows.append(
                {
                    **{
                        key: contrast[key]
                        for key in (
                            "contrast_id",
                            "source_id",
                            "trace_id",
                            "trace_source",
                            "treatment_condition",
                            "cell_kind",
                            "family_id",
                            "base_id",
                            "arm",
                            "variant",
                            "delta_m_general",
                        )
                    },
                    "layer": layer,
                    "axis_definition": axis_name,
                    **values,
                }
            )
        contrast["g_off"] = float(by_layer[60]["off_direction_share"]) - float(
            by_layer[53]["off_direction_share"]
        )
        contrast["off_share_53"] = by_layer[53]["off_direction_share"]
        contrast["off_share_54"] = by_layer[54]["off_direction_share"]
        contrast["off_share_60"] = by_layer[60]["off_direction_share"]
        contrast["off_share_63"] = by_layer[63]["off_direction_share"]
        contrast["relative_delta_53"] = by_layer[53]["relative_delta"]
        contrast["relative_delta_60"] = by_layer[60]["relative_delta"]
        contrast["layer45_delta_norm"] = by_layer[45]["delta_norm"]
        contrast_rows.append(contrast)
    return contrast_rows, metric_rows, aggregate


def condition_rows(
    contrasts: Sequence[dict[str, Any]], condition: str, *, trace_source: str = PRIMARY_TRACE_SOURCE
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in contrasts
        if row["treatment_condition"] == condition and row["trace_source"] == trace_source
    ]
    require(len(selected) == 96, f"Expected 96 rows for {trace_source}/{condition}")
    return selected


def integrity_summary(
    rows: Sequence[dict[str, Any]], contrasts: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_cell = {row["cell_id"]: row for row in rows}
    output: dict[str, Any] = {}
    for condition in ("full_p25", *RANDOM_CONDITIONS):
        selected = condition_rows(contrasts, condition)
        treated = [by_cell[row["contrast_id"]] for row in selected]
        baselines = [by_cell[f"{row['trace_id']}::uncapped"] for row in selected]
        mean_mass = statistics.fmean(
            float(row["logit_diagnostics"]["option_mass"]) for row in treated
        )
        base_mass = statistics.fmean(
            float(row["logit_diagnostics"]["option_mass"]) for row in baselines
        )
        top_rate = statistics.fmean(
            float(row["logit_diagnostics"]["top_token_is_option"]) for row in treated
        )
        output[condition] = {
            "rows": len(treated),
            "mean_option_mass": mean_mass,
            "baseline_mean_option_mass": base_mass,
            "option_mass_ratio": mean_mass / (base_mass + EPSILON),
            "mean_entropy": statistics.fmean(
                float(row["logit_diagnostics"]["full_vocab_entropy"]) for row in treated
            ),
            "top_token_is_option_rate": top_rate,
            "option_argmax_changes": sum(bool(row["option_argmax_changed"]) for row in selected),
            "collapsed": mean_mass / (base_mass + EPSILON) < 0.5 or top_rate < 0.9,
        }
    return output


def _lofo_transforms_hidden(
    features: np.ndarray, labels: np.ndarray, families: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    folds = []
    for family in sorted(set(families.tolist())):
        test = families == family
        train = ~test
        scaler = StandardScaler().fit(features[train])
        train_scaled = scaler.transform(features[train])
        test_scaled = scaler.transform(features[test])
        components = min(10, train_scaled.shape[0] - 1, train_scaled.shape[1])
        pca = PCA(n_components=components, svd_solver="full").fit(train_scaled)
        folds.append((pca.transform(train_scaled), pca.transform(test_scaled), train, test, family))
    return folds


def _evaluate_folds(
    folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    probabilities = np.zeros(labels.shape[0], dtype=float)
    predictions = np.zeros(labels.shape[0], dtype=int)
    per_family: dict[str, float] = {}
    for train_x, test_x, train, test, family in folds:
        model = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", random_state=0, max_iter=1000
        ).fit(train_x, labels[train])
        probabilities[test] = model.predict_proba(test_x)[:, 1]
        predictions[test] = (probabilities[test] >= 0.5).astype(int)
        per_family[family] = float(balanced_accuracy_score(labels[test], predictions[test]))
    return probabilities, predictions, per_family


def _probe_summary(
    folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    labels: np.ndarray,
    families: np.ndarray,
    flip_masks: np.ndarray,
) -> dict[str, Any]:
    probabilities, predictions, per_family = _evaluate_folds(folds, labels)
    observed_balanced = float(balanced_accuracy_score(labels, predictions))
    observed_auc = float(roc_auc_score(labels, probabilities))
    family_order = sorted(set(families.tolist()))
    family_index = {family: index for index, family in enumerate(family_order)}
    permutations = []
    for mask in flip_masks:
        permuted = np.asarray(
            [
                label ^ int(mask[family_index[family]])
                for label, family in zip(labels, families, strict=True)
            ],
            dtype=int,
        )
        _prob, pred, _per = _evaluate_folds(folds, permuted)
        permutations.append(float(balanced_accuracy_score(permuted, pred)))
    p_value = (1 + sum(value >= observed_balanced for value in permutations)) / (
        1 + len(permutations)
    )
    return {
        "balanced_accuracy": observed_balanced,
        "roc_auc": observed_auc,
        "per_family_balanced_accuracy": per_family,
        "family_label_flip_permutations": len(permutations),
        "family_label_flip_p_value_balanced_accuracy": p_value,
    }


def probe_analysis(
    rows: Sequence[dict[str, Any]], aggregate: torch.Tensor
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary = [
        (index, row)
        for index, row in enumerate(rows)
        if row["trace_source"] == PRIMARY_TRACE_SOURCE
        and row["forward_condition"] in {"uncapped", "full_p25"}
    ]
    by_source_condition = {
        (row["source_id"], row["forward_condition"]): index for index, row in primary
    }
    base_centers: dict[tuple[str, int], torch.Tensor] = {}
    for base_id in sorted({row["base_id"] for _index, row in primary}):
        base_uncapped = [
            (index, row)
            for index, row in primary
            if row["base_id"] == base_id and row["forward_condition"] == "uncapped"
        ]
        require(len(base_uncapped) == 4, f"Expected four uncapped arms for {base_id}")
        for layer in (54, 60, 63):
            layer_index = CAPTURE_LAYERS.index(layer)
            base_centers[(base_id, layer)] = torch.stack(
                [aggregate[index, layer_index].float() for index, _row in base_uncapped]
            ).mean(dim=0)

    vf_sources = sorted({row["source_id"] for _index, row in primary if row["arm"] in {"V", "F"}})
    source_meta = {
        row["source_id"]: row
        for _index, row in primary
        if row["forward_condition"] == "uncapped" and row["arm"] in {"V", "F"}
    }
    labels = np.asarray([1 if source_meta[source]["arm"] == "V" else 0 for source in vf_sources])
    families = np.asarray([source_meta[source]["family_id"] for source in vf_sources])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    flip_masks = rng.integers(0, 2, size=(PROBE_PERMUTATIONS, 12), dtype=np.int8)
    hidden_results: dict[str, Any] = {}
    pca_rows: list[dict[str, Any]] = []
    pca_input: list[np.ndarray] = []
    pca_meta: list[tuple[str, str, str, str, str]] = []
    for condition in ("uncapped", "full_p25"):
        for layer in (54, 60, 63):
            layer_index = CAPTURE_LAYERS.index(layer)
            features = []
            for source in vf_sources:
                meta = source_meta[source]
                index = by_source_condition[(source, condition)]
                residual = (
                    aggregate[index, layer_index].float() - base_centers[(meta["base_id"], layer)]
                )
                features.append(residual.numpy())
                if layer == 60:
                    pca_input.append(residual.numpy())
                    pca_meta.append(
                        (
                            source,
                            meta["base_id"],
                            meta["family_id"],
                            meta["arm"],
                            condition,
                        )
                    )
            features_np = np.stack(features)
            folds = _lofo_transforms_hidden(features_np, labels, families)
            hidden_results[f"layer{layer}_{condition}"] = _probe_summary(
                folds, labels, families, flip_masks
            )

    prompt_rows = {row["id"]: row for row in read_jsonl(E2A_DATA / "answer_inputs96.jsonl")}
    texts = [
        "\n".join(message["content"] for message in prompt_rows[source]["messages"])
        for source in vf_sources
    ]
    text_folds = []
    for family in sorted(set(families.tolist())):
        test = families == family
        train = ~test
        vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=6000)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char", ngram_range=(3, 5), min_df=1, max_features=6000
                    ),
                ),
            ]
        )
        train_sparse = vectorizer.fit_transform([texts[index] for index in np.flatnonzero(train)])
        test_sparse = vectorizer.transform([texts[index] for index in np.flatnonzero(test)])
        components = min(10, train_sparse.shape[0] - 1, train_sparse.shape[1] - 1)
        svd = TruncatedSVD(n_components=components, random_state=0).fit(train_sparse)
        text_folds.append(
            (svd.transform(train_sparse), svd.transform(test_sparse), train, test, family)
        )
    text_result = _probe_summary(text_folds, labels, families, flip_masks)

    pca_matrix = np.stack(pca_input)
    pca = PCA(n_components=2, svd_solver="full").fit(pca_matrix)
    coordinates = pca.transform(pca_matrix)
    for meta, coordinate in zip(pca_meta, coordinates, strict=True):
        source, base_id, family_id, arm, condition = meta
        pca_rows.append(
            {
                "source_id": source,
                "base_id": base_id,
                "family_id": family_id,
                "arm": arm,
                "condition": condition,
                "pc1": float(coordinate[0]),
                "pc2": float(coordinate[1]),
            }
        )
    return {
        "status": "complete",
        "claim": "V/F condition information remains linearly decodable",
        "forbidden_claims": [
            "the model understands validity",
            "the model knows F is false",
            "validity is causally available",
        ],
        "hidden_state_lofo": hidden_results,
        "text_only_tfidf_lofo": text_result,
        "pca_layer60": {
            "descriptive_only": True,
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
            "move_to_appendix_if_low_variance": bool(
                sum(pca.explained_variance_ratio_) < 0.1
            ),
        },
    }, pca_rows


def layerwise_output_profile(run_dir: Path, output_root: Path) -> dict[str, Any]:
    source = run_dir / "layer_metrics.csv"
    require(source.is_file(), "Layer-wise lens metrics are missing")
    shutil.copyfile(source, output_root / "layer_metrics.csv")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 288 * len(CAPTURE_LAYERS), "Layer-wise metric row count changed")
    official = [row for row in rows if row["pair_kind"] == "official_p25"]
    noop = [row for row in rows if row["pair_kind"] == "engineering_noop"]
    require(len(official) == 192 * 64 and len(noop) == 96 * 64, "Layer pair counts changed")

    traces: dict[str, Any] = {}
    for trace_source in ("e2a_uncapped", "e2a_full_p25"):
        selected = [row for row in official if row["trace_source"] == trace_source]
        require(len(selected) == 96 * 64, f"Layer profile count changed: {trace_source}")
        metrics: dict[str, Any] = {}
        for metric in (
            "delta_m_general_mean_noninitial",
            "kl_opt",
            "kl_vocab",
            "delta_h_l2",
            "state_cosine",
            "uncapped_state_l2",
            "treatment_state_l2",
        ):
            by_layer: dict[str, Any] = {}
            lofo: dict[str, Any] = {}
            for layer in CAPTURE_LAYERS:
                layer_rows = [row for row in selected if int(row["layer"]) == layer]
                scores = family_scores(layer_rows, metric)
                by_layer[str(layer)] = family_bootstrap(
                    scores, draws=LAYER_BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
                )
                lofo[str(layer)] = leave_one_family_out(scores)
            metrics[metric] = {
                "layers": by_layer,
                "leave_one_family_out": lofo,
                "descriptive_exploratory": True,
                "per_layer_p_values": False,
            }
        traces[trace_source] = metrics

    arms: dict[str, Any] = {}
    primary = [row for row in official if row["trace_source"] == PRIMARY_TRACE_SOURCE]
    for arm in "NRVF":
        arm_rows = [row for row in primary if row["arm"] == arm]
        arms[arm] = {
            str(layer): family_bootstrap(
                family_scores(
                    [row for row in arm_rows if int(row["layer"]) == layer],
                    "delta_m_general_mean_noninitial",
                ),
                draws=LAYER_BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
            for layer in CAPTURE_LAYERS
        }

    noop_max_kl = max(float(row["kl_opt"]) for row in noop)
    noop_max_delta = max(float(row["delta_h_l2"]) for row in noop)
    layer63 = [row for row in rows if int(row["layer"]) == 63]
    final_anchor_error = max(
        max(
            float(row["final_layer_uncapped_anchor_error"]),
            float(row["final_layer_treatment_anchor_error"]),
        )
        for row in layer63
    )
    return {
        "mapping": "frozen_checkpoint_final_norm_then_lm_head",
        "translator_fitted": False,
        "capture_layers": list(CAPTURE_LAYERS),
        "trace_sources": traces,
        "primary_arm_curves": arms,
        "engineering_noop": {
            "rows": len(noop),
            "max_kl_opt": noop_max_kl,
            "max_delta_h_l2": noop_max_delta,
            "passed": noop_max_kl < 1e-6 and noop_max_delta < 1e-4,
        },
        "final_layer_anchor": {
            "max_abs_candidate_logit_error": final_anchor_error,
            "tolerance": 1e-3,
            "passed": final_anchor_error < 1e-3,
        },
        "limitations": [
            "Divergence below layer 46 can reflect cache-propagated effects.",
            "Low early-layer values can reflect native logit-lens readability.",
            "The curves are descriptive and do not localize an origin layer.",
        ],
    }


def write_plot_data(
    output_root: Path,
    contrasts: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    layer_profile: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "panel_a": {},
        "panel_b": {},
        "panel_c": {},
        "panel_d": {"trace_sources": {}, "arm_curves": {}},
    }
    for condition in ("full_p25", *RANDOM_CONDITIONS):
        label = "official_p25" if condition == "full_p25" else condition
        selected = [
            row
            for row in metric_rows
            if row["treatment_condition"] == condition
            and row["trace_source"] == PRIMARY_TRACE_SOURCE
            and row["layer"] != 45
        ]
        stats["panel_a"][label] = {}
        for layer in range(46, 64):
            layer_rows = [row for row in selected if row["layer"] == layer]
            scores = family_scores(layer_rows, "off_direction_share")
            summary = family_bootstrap(scores)
            stats["panel_a"][label][str(layer)] = summary
            rows.append(
                {
                    "panel": "A",
                    "series": label,
                    "layer": layer,
                    "estimate": summary["estimate"],
                    "ci_low": summary["family_cluster_bootstrap_95ci"][0],
                    "ci_high": summary["family_cluster_bootstrap_95ci"][1],
                }
            )
    primary_p25 = condition_rows(contrasts, "full_p25")
    for arm in "NRVF":
        stats["panel_b"][arm] = {}
        for layer in (53, 60, 63):
            layer_rows = [
                row
                for row in metric_rows
                if row["treatment_condition"] == "full_p25"
                and row["trace_source"] == PRIMARY_TRACE_SOURCE
                and row["arm"] == arm
                and row["layer"] == layer
            ]
            summary = family_bootstrap(family_scores(layer_rows, "off_direction_share"))
            stats["panel_b"][arm][str(layer)] = summary
            rows.append(
                {
                    "panel": "B",
                    "series": arm,
                    "layer": layer,
                    "estimate": summary["estimate"],
                    "ci_low": summary["family_cluster_bootstrap_95ci"][0],
                    "ci_high": summary["family_cluster_bootstrap_95ci"][1],
                }
            )
    family_g = family_scores(primary_p25, "g_off")
    family_m = family_scores(primary_p25, "delta_m_general")
    rho = float(spearmanr(list(family_g.values()), list(family_m.values())).statistic)
    stats["panel_c"] = {
        "descriptive_only": True,
        "spearman_rho": rho,
        "families": 12,
        "regression_line_forbidden": True,
    }
    for family in sorted(family_g):
        rows.append(
            {
                "panel": "C",
                "series": family,
                "x_g_off": family_g[family],
                "y_delta_m_general": family_m[family],
            }
        )
    for trace_source, label in ((PRIMARY_TRACE_SOURCE, "U"), ("e2a_full_p25", "P")):
        stats["panel_d"]["trace_sources"][label] = {}
        for metric in ("delta_m_general_mean_noninitial", "kl_opt"):
            profile = layer_profile["trace_sources"][trace_source][metric]["layers"]
            stats["panel_d"]["trace_sources"][label][metric] = profile
            for layer in CAPTURE_LAYERS:
                summary = profile[str(layer)]
                rows.append(
                    {
                        "panel": "D",
                        "series": label,
                        "metric": metric,
                        "layer": layer,
                        "estimate": summary["estimate"],
                        "ci_low": summary["family_cluster_bootstrap_95ci"][0],
                        "ci_high": summary["family_cluster_bootstrap_95ci"][1],
                    }
                )
    stats["panel_d"]["arm_curves"] = layer_profile["primary_arm_curves"]
    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "plot_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / "plot_stats.json", stats)
    return stats


def save_official_hidden_shards(
    output_root: Path, rows: Sequence[Mapping[str, Any]], block_outputs: torch.Tensor
) -> dict[str, Any]:
    """Store the 384 official rows in four GitHub-safe condition shards."""
    require(block_outputs.shape[0] == len(rows), "Hidden-state row mapping changed")
    shard_root = output_root / "hidden_states"
    shard_root.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, Any]] = []
    shard_hashes: dict[str, str] = {}
    for trace_source in (PRIMARY_TRACE_SOURCE, "e2a_full_p25"):
        for condition in ("uncapped", "full_p25"):
            indices = [
                index
                for index, row in enumerate(rows)
                if row["trace_source"] == trace_source
                and row["forward_condition"] == condition
            ]
            require(len(indices) == 96, f"Official shard count changed: {trace_source}/{condition}")
            filename = f"{trace_source}-{condition}.safetensors"
            path = shard_root / filename
            save_file(
                {"block_outputs": block_outputs[indices].to(torch.bfloat16).contiguous()},
                path,
            )
            relative = str(path.relative_to(output_root))
            shard_hashes[relative] = file_sha256(path)
            for shard_index, result_index in enumerate(indices):
                mapping_rows.append(
                    {
                        "cell_id": rows[result_index]["cell_id"],
                        "result_index": result_index,
                        "shard": relative,
                        "shard_index": shard_index,
                        "layers": list(CAPTURE_LAYERS),
                    }
                )
    mapping_path = output_root / "hidden_states.index.json"
    write_json(
        mapping_path,
        {
            "schema": "cedar.rg1p_stagea.hidden_state_index.v1",
            "storage_dtype": "bfloat16",
            "official_rows": len(mapping_rows),
            "shards": shard_hashes,
            "rows": mapping_rows,
            "random_and_noop_raw_states": "formal-stagea-seed0/tensors",
        },
    )
    return {
        "index": file_sha256(mapping_path),
        "shards": shard_hashes,
        "official_rows": len(mapping_rows),
    }


def analyze(output_root: Path) -> dict[str, Any]:
    audit = read_json(output_root / "FORMAL_AUDIT.json")
    require(audit.get("status") == "passed", "Analysis requires passed formal audit")
    run_dir = output_root / "formal-stagea-seed0"
    rows = read_jsonl(run_dir / "results.jsonl")
    require(len(rows) == 864, "RG1-P analysis requires 864 results")
    contrasts, metric_rows, aggregate_payload = build_metrics(rows, run_dir)
    # build_metrics uses the default only for its temporary records target; rewrite
    # explicitly for non-default output roots.
    write_jsonl(output_root / "propagation_metrics.jsonl", metric_rows)
    hidden_delivery = save_official_hidden_shards(
        output_root, rows, aggregate_payload["block_outputs"]
    )
    record_rows = []
    for tensor_index, row in enumerate(rows):
        record_rows.append(
            {
                "tensor_index": tensor_index,
                "cell_id": row["cell_id"],
                "source_id": row["source_id"],
                "trace_id": row["trace_id"],
                "trace_source": row["trace_source"],
                "forward_condition": row["forward_condition"],
                "family_id": row["family_id"],
                "base_id": row["base_id"],
                "arm": row["arm"],
                "variant": row["variant"],
                "prompt_sha256": row["prompt_sha256"],
                "trace_sha256": row["trace_prefix_token_ids_sha256"],
                "layer_mapping": list(CAPTURE_LAYERS),
                "terminal_probe": row["terminal_probe"],
                "e2b_anchor_logits": row["e2b_anchor_logits"],
                "current_logits": row["logit_diagnostics"],
                "cap_telemetry": row["full_cap_stats"],
            }
        )
    write_jsonl(output_root / "records.jsonl", record_rows)

    primary = condition_rows(contrasts, "full_p25")
    secondary = condition_rows(contrasts, "full_p25", trace_source="e2a_full_p25")
    primary_g_scores = family_scores(primary, "g_off")
    primary_margin_scores = family_scores(primary, "delta_m_general")
    primary_g = family_bootstrap(primary_g_scores)
    secondary_g = family_bootstrap(family_scores(secondary, "g_off"))
    margin = family_bootstrap(primary_margin_scores)

    random_summaries: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for condition in RANDOM_CONDITIONS:
        selected = condition_rows(contrasts, condition)
        g_scores = family_scores(selected, "g_off")
        m_scores = family_scores(selected, "delta_m_general")
        random_summaries[condition] = {
            "g_off": family_bootstrap(g_scores),
            "delta_m_general": family_bootstrap(m_scores),
            "delta_m_target": family_bootstrap(family_scores(selected, "delta_m_target")),
            "delta_m_background": family_bootstrap(
                family_scores(selected, "delta_m_background")
            ),
            "delta_m_specific": family_bootstrap(
                family_scores(selected, "delta_m_specific")
            ),
            "relative_delta_layer60": family_bootstrap(
                family_scores(selected, "relative_delta_60")
            ),
        }
        comparisons[condition] = {
            "p25_minus_random_g_off": family_bootstrap(
                paired_difference(primary_g_scores, g_scores)
            ),
            "p25_minus_random_delta_m_general": family_bootstrap(
                paired_difference(primary_margin_scores, m_scores)
            ),
        }

    arm_results: dict[str, Any] = {}
    arm_positive = 0
    for arm in "NRVF":
        selected = [row for row in primary if row["arm"] == arm]
        summary = family_bootstrap(family_scores(selected, "g_off"))
        arm_results[arm] = {
            "g_off": summary,
            "delta_m_general": family_bootstrap(family_scores(selected, "delta_m_general")),
            "delta_m_target": family_bootstrap(family_scores(selected, "delta_m_target")),
            "delta_m_background": family_bootstrap(
                family_scores(selected, "delta_m_background")
            ),
            "delta_m_specific": family_bootstrap(
                family_scores(selected, "delta_m_specific")
            ),
            "layers": {
                str(layer): {
                    metric: family_bootstrap(
                        family_scores(
                            [
                                row
                                for row in metric_rows
                                if row["treatment_condition"] == "full_p25"
                                and row["trace_source"] == PRIMARY_TRACE_SOURCE
                                and row["arm"] == arm
                                and row["layer"] == layer
                            ],
                            metric,
                        )
                    )
                    for metric in (
                        "aligned_norm",
                        "off_direction_norm",
                        "off_direction_share",
                        "relative_delta",
                    )
                }
                for layer in (53, 54, 60, 63)
            },
        }
        arm_positive += summary["estimate"] > 0

    integrity = integrity_summary(rows, contrasts)
    primary_p25_rows = [
        row
        for row in rows
        if row["trace_source"] == PRIMARY_TRACE_SOURCE
        and row["forward_condition"] == "full_p25"
    ]
    require(len(primary_p25_rows) == 96, "Primary p25 telemetry row count changed")
    cap_telemetry: dict[str, Any] = {}
    for layer in CAP_LAYERS:
        layer_stats = [row["full_cap_stats"]["by_layer"][str(layer)] for row in primary_p25_rows]
        total_positions = sum(int(item["total_positions"]) for item in layer_stats)
        triggered_positions = sum(int(item["triggered_positions"]) for item in layer_stats)
        correction_sum = sum(float(item["projection_correction_sum"]) for item in layer_stats)
        cap_telemetry[str(layer)] = {
            "total_positions": total_positions,
            "triggered_positions": triggered_positions,
            "trigger_rate": triggered_positions / total_positions,
            "projection_correction_sum": correction_sum,
            "mean_projection_correction_per_position": correction_sum / total_positions,
        }
    random_comparable = not any(
        integrity[condition]["collapsed"] for condition in RANDOM_CONDITIONS
    )
    positive_families = sum(value > 0 for value in primary_g_scores.values())
    stable_primary = (
        primary_g["family_cluster_bootstrap_95ci"][0] > 0
        and positive_families >= 7
        and arm_positive >= 3
    )
    g_exceeds_all = all(
        comparisons[condition]["p25_minus_random_g_off"]["family_cluster_bootstrap_95ci"][0] > 0
        for condition in RANDOM_CONDITIONS
    )
    margin_differences = [
        comparisons[condition]["p25_minus_random_delta_m_general"]
        for condition in RANDOM_CONDITIONS
    ]
    margin_signs = [math.copysign(1, item["estimate"]) for item in margin_differences]
    margin_differs_all = len(set(margin_signs)) == 1 and all(
        item["family_cluster_bootstrap_95ci"][0] > 0 or item["family_cluster_bootstrap_95ci"][1] < 0
        for item in margin_differences
    )
    generic_rotation = all(
        random_summaries[condition]["g_off"]["family_cluster_bootstrap_95ci"][0] > 0
        for condition in RANDOM_CONDITIONS
    )
    p25_margin_sign = math.copysign(1, margin["estimate"]) if margin["estimate"] else 0
    random_margin_signs = [
        math.copysign(1, random_summaries[condition]["delta_m_general"]["estimate"])
        if random_summaries[condition]["delta_m_general"]["estimate"]
        else 0
        for condition in RANDOM_CONDITIONS
    ]
    opposite_margins = p25_margin_sign != 0 and all(
        value == -p25_margin_sign for value in random_margin_signs
    )
    if stable_primary and (g_exceeds_all or margin_differs_all) and random_comparable:
        route = "A"
        route_text = "off_axis_propagation_exceeded_preregistered_random_references"
    elif generic_rotation and opposite_margins and random_comparable:
        route = "C"
        route_text = "generic_rotation_but_direction_dependent_answer_geometry"
    elif stable_primary and random_comparable:
        route = "B"
        route_text = "rotation_overlapped_schedule_matched_random_references"
    else:
        route = "D"
        route_text = "unstable_or_noncomparable_exploratory_result"

    family_g = primary_g_scores
    family_m = primary_margin_scores
    descriptive_rho = float(spearmanr(list(family_g.values()), list(family_m.values())).statistic)
    layer_profile = layerwise_output_profile(run_dir, output_root)
    probe, pca_rows = probe_analysis(rows, aggregate_payload["block_outputs"])
    write_jsonl(output_root / "pca_points.jsonl", pca_rows)
    plot_stats = write_plot_data(output_root, contrasts, metric_rows, layer_profile)
    analysis = {
        "schema": "cedar.rg1p_stagea.analysis.v1",
        "status": "complete",
        "scope": "controlled_fixed_trace_representation_audit",
        "not_causal_mediation": True,
        "rows": len(rows),
        "contrasts": len(contrasts),
        "families": 12,
        "bootstrap": {
            "unit": "family",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "interval": "two_sided_percentile_95ci",
        },
        "axis_definition": {
            "layer45": "reporting layer without an authorized axis; no projection",
            "layers46_53": "official layer-specific Assistant Axis",
            "layers54_63": "fixed layer-53 exit direction",
            "required_caption": (
                "Downstream projections use the fixed layer-53 cap direction, not a "
                "layer-specific Assistant Axis."
            ),
        },
        "primary": {
            "trace_source": PRIMARY_TRACE_SOURCE,
            "g_off_layer60_minus_layer53": primary_g,
            "delta_m_general": margin,
            "delta_m_target": family_bootstrap(family_scores(primary, "delta_m_target")),
            "delta_m_background": family_bootstrap(
                family_scores(primary, "delta_m_background")
            ),
            "delta_m_specific": family_bootstrap(
                family_scores(primary, "delta_m_specific")
            ),
            "positive_families": positive_families,
            "leave_one_family_out_g_off": leave_one_family_out(primary_g_scores),
            "arm_breakdown": arm_results,
        },
        "secondary_p25_source": {
            "g_off_layer60_minus_layer53": secondary_g,
            "delta_m_general": family_bootstrap(
                family_scores(secondary, "delta_m_general")
            ),
            "leave_one_family_out_g_off": leave_one_family_out(family_scores(secondary, "g_off")),
        },
        "random_direction_reference_battery": {
            "conditions": random_summaries,
            "p25_pairwise_comparisons": comparisons,
            "only_four_references_not_a_null_distribution": True,
            "output_space_integrity": integrity,
            "random_references_comparable": random_comparable,
        },
        "official_p25_correction_telemetry": cap_telemetry,
        "descriptive_family_association": {
            "spearman_rho": descriptive_rho,
            "families": 12,
            "no_regression_line": True,
            "not_a_route_gate": True,
        },
        "layerwise_output_divergence": layer_profile,
        "vf_probe_and_text_control": probe,
        "plot_stats": plot_stats,
        "route_gate": {
            "route": route,
            "classification": route_text,
            "stable_primary": stable_primary,
            "g_off_exceeds_all_random_references": g_exceeds_all,
            "margin_differs_from_all_random_references": margin_differs_all,
            "generic_random_rotation": generic_rotation,
            "opposite_margin_signs": opposite_margins,
            "random_output_comparable": random_comparable,
        },
        "claim_contract": {
            "allowed": [
                "fixed-trace representation audit",
                "exit-axis-aligned component",
                "off-exit-axis component",
                "four preregistered schedule- and norm-matched random references",
            ],
            "forbidden": [
                "causal mediation",
                "complete neural circuit",
                "layer-specific downstream Assistant Axis",
                "unique against all random perturbations",
                "natural-answer behavior proven by Stage A alone",
            ],
        },
        "artifact_hashes": {
            "hidden_states.index.json": file_sha256(output_root / "hidden_states.index.json"),
            "hidden_state_shards": hidden_delivery,
            "propagation_metrics.jsonl": file_sha256(output_root / "propagation_metrics.jsonl"),
            "records.jsonl": file_sha256(output_root / "records.jsonl"),
            "layer_metrics.csv": file_sha256(output_root / "layer_metrics.csv"),
        },
    }
    write_json(output_root / "ANALYSIS.json", analysis)
    ci = primary_g["family_cluster_bootstrap_95ci"]
    markdown = "\n".join(
        [
            "# RG1-P Stage A Analysis",
            "",
            "Controlled fixed-trace representation audit; not a causal-mediation result.",
            "",
            f"- G_off (layer 60 - layer 53): {primary_g['estimate']:.6f} "
            f"[{ci[0]:.6f}, {ci[1]:.6f}]",
            f"- positive families: {positive_families}/12",
            f"- delta m_general: {margin['estimate']:.6f}",
            f"- route: {route} ({route_text})",
            "- downstream projections use the fixed layer-53 cap direction, not a "
            "layer-specific Assistant Axis.",
            "",
        ]
    )
    (output_root / "ANALYSIS.md").write_text(markdown, encoding="utf-8", newline="\n")
    summary_zh = "\n".join(
        [
            "# RG1-P Stage A 结果摘要",
            "",
            "本实验是固定轨迹表征审计，不是因果中介实验。",
            "",
            f"- G_off（layer 60 - layer 53）：{primary_g['estimate']:.6f}，"
            f"95% family-bootstrap CI [{ci[0]:.6f}, {ci[1]:.6f}]",
            f"- 同向 family：{positive_families}/12",
            f"- 路由：{route}（{route_text}）",
            "- 下游投影统一使用固定 layer-53 exit direction。",
            "- 四个随机条件只是预注册方向参考，不是完整随机零分布。",
            "",
        ]
    )
    (output_root / "RESULT_SUMMARY_ZH.md").write_text(summary_zh, encoding="utf-8", newline="\n")
    return analysis


def main() -> None:
    args = parse_args()
    result = analyze(args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
