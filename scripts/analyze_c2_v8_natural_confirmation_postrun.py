#!/usr/bin/env python3
"""Disclosed post-run correction for the C2/V8 hierarchical sensitivity model."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts import analyze_c2_v8_natural_confirmation as frozen
except ModuleNotFoundError:
    import analyze_c2_v8_natural_confirmation as frozen

CORRECTION_ID = "C2_V8_POST_RUN_ANALYSIS_CORRECTION_01"
CORRECTION_DOC = Path(
    "docs/C2_V8_POST_RUN_ANALYSIS_CORRECTION_01_HIERARCHICAL_MODEL.md"
)


def fit_hierarchical_model(
    direct_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit domain and nested-base random intercepts without collinear columns."""
    rows = []
    for pair in direct_pairs:
        for condition in frozen.CONDITIONS:
            rows.append(
                {
                    "margin": frozen.direct_margin(pair, condition),
                    "cap": 1 if condition == "p25" else 0,
                    "arm": pair["arm"],
                    "target_position": pair["target_answer"],
                    "difficulty": pair["difficulty"],
                    "template": str(pair["template_id"]),
                    "base": pair["base_id"],
                    "domain": pair["domain"],
                }
            )
    frame = frozen.pd.DataFrame(rows)
    formula = (
        "margin ~ cap * C(arm, Treatment(reference='AO')) + C(target_position) "
        "+ C(difficulty) + C(template)"
    )
    try:
        model = frozen.smf.mixedlm(
            formula,
            frame,
            groups=frame["domain"],
            re_formula="1",
            vc_formula={"base": "0 + C(base)"},
        )
        fitted = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        interaction = next(
            name
            for name in fitted.params.index
            if name.startswith("cap:C(arm") and name.endswith("[T.S]")
        )
        return {
            "status": "fitted",
            "formula": formula,
            "n_observations": len(frame),
            "random_effects": ["(1 | domain)", "(1 | base)"],
            "interaction_term": interaction,
            "interaction_estimate": float(fitted.params[interaction]),
            "interaction_standard_error": float(fitted.bse[interaction]),
            "interaction_p_value": float(fitted.pvalues[interaction]),
            "converged": bool(fitted.converged),
            "all_parameters": {
                name: float(value) for name, value in fitted.params.items()
            },
        }
    except Exception as error:
        return {
            "status": "fit_failed",
            "formula": formula,
            "n_observations": len(frame),
            "random_effects": ["(1 | domain)", "(1 | base)"],
            "error": f"{type(error).__name__}: {error}",
        }


def add_correction_provenance(output_root: Path) -> None:
    analysis_path = output_root / "C2_V8_ANALYSIS.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["post_run_analysis_correction"] = {
        "id": CORRECTION_ID,
        "gpu_outputs_rerun": False,
        "frozen_analyzer_modified": False,
        "correction_document": str(CORRECTION_DOC),
        "correction_document_canonical_text_sha256": frozen.canonical_text_sha256(
            CORRECTION_DOC
        ),
        "correction_script": str(Path(__file__).relative_to(frozen.ROOT)),
        "correction_script_canonical_text_sha256": frozen.canonical_text_sha256(
            Path(__file__)
        ),
        "correction_script_raw_sha256_diagnostic": frozen.file_sha256(Path(__file__)),
        "correction_git_commit": frozen.git_value("rev-parse", "HEAD"),
    }
    frozen.atomic_write_json(analysis_path, analysis)

    model = analysis["direct"]["hierarchical_model"]
    summary_path = output_root / "C2_V8_SUMMARY_ZH.md"
    summary = summary_path.read_text(encoding="utf-8").rstrip()
    summary += (
        "\n\n## 层级模型敏感性分析（post-run correction 01）\n\n"
        f"- 状态：`{model['status']}`；随机效应：domain 与 base 随机截距。\n"
        f"- `cap × S` 交互：{model['interaction_estimate']:+.4f} logit。\n"
        f"- 标准误：{model['interaction_standard_error']:.4f}；"
        f"p={model['interaction_p_value']:.3g}。\n"
        "- 修正只处理首次分析中的结构性奇异，不修改主 bootstrap、原始输出或判定规则。\n"
    )
    summary_path.write_text(summary, encoding="utf-8", newline="\n")
    frozen.atomic_write_json(
        output_root / "delivery_manifest.json",
        frozen.delivery_manifest(output_root),
    )


def main() -> None:
    args = frozen.parse_args()
    frozen.fit_hierarchical_model = fit_hierarchical_model
    frozen.main()
    add_correction_provenance(args.output_root)


if __name__ == "__main__":
    main()
