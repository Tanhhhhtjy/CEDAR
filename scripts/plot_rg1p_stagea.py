#!/usr/bin/env python3
"""Generate the preregistered RG1-P four-panel figure without new Python deps."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/rg1p_stagea"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from rg1p_common import file_sha256, read_json, require, write_json  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/rg1p_stagea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def scale(value: float, low: float, high: float, start: float, stop: float) -> float:
    if high == low:
        return (start + stop) / 2
    return start + (value - low) * (stop - start) / (high - low)


def polyline(points: list[tuple[float, float]], **attrs: Any) -> str:
    body = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    properties = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<polyline points="{body}" {properties}/>'


def generate_svg(stats: dict[str, Any]) -> str:
    width, height = 1500, 1120
    panel_width = 430
    lefts = [55, 535, 1015]
    top, bottom = 65, 425
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}"
        ".title{font-size:17px;font-weight:700}.axis{font-size:12px}.label{font-size:11px}"
        ".caption{font-size:10px;fill:#333}</style>",
    ]

    # Panel A: official, four references, and their observed range.
    x0 = lefts[0]
    all_a = stats["panel_a"]
    values = [float(item["estimate"]) for series in all_a.values() for item in series.values()]
    y_low, y_high = min(values), max(values)
    padding = max(0.03, 0.1 * (y_high - y_low))
    y_low, y_high = max(0.0, y_low - padding), min(1.0, y_high + padding)
    parts.append(f'<text x="{x0}" y="28" class="title">A  Downstream off-direction share</text>')
    parts.append(
        f'<line x1="{x0}" y1="{bottom}" x2="{x0 + panel_width}" y2="{bottom}" stroke="#222"/>'
    )
    parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="#222"/>')
    for layer in (45, 53, 60, 63):
        x = scale(layer, 45, 63, x0, x0 + panel_width)
        parts.append(f'<line x1="{x}" y1="{bottom}" x2="{x}" y2="{bottom + 5}" stroke="#222"/>')
        parts.append(
            f'<text x="{x}" y="{bottom + 20}" text-anchor="middle" class="axis">{layer}</text>'
        )
        if layer in {53, 60}:
            parts.append(
                f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" '
                'stroke="#4b6a88" stroke-dasharray="4,4" opacity="0.55"/>'
            )
    for fraction in (0.0, 0.5, 1.0):
        value = y_low + fraction * (y_high - y_low)
        y = scale(value, y_low, y_high, bottom, top)
        parts.append(f'<line x1="{x0 - 5}" y1="{y}" x2="{x0}" y2="{y}" stroke="#222"/>')
        parts.append(
            f'<text x="{x0 - 9}" y="{y + 4}" text-anchor="end" class="axis">{value:.2f}</text>'
        )
    random_names = [name for name in all_a if name != "official_p25"]
    upper_points, lower_points = [], []
    for layer in range(46, 64):
        refs = [float(all_a[name][str(layer)]["estimate"]) for name in random_names]
        x = scale(layer, 45, 63, x0, x0 + panel_width)
        upper_points.append((x, scale(max(refs), y_low, y_high, bottom, top)))
        lower_points.append((x, scale(min(refs), y_low, y_high, bottom, top)))
    band = upper_points + list(reversed(lower_points))
    band_body = " ".join(f"{x:.2f},{y:.2f}" for x, y in band)
    parts.append(f'<polygon points="{band_body}" fill="#b8b8b8" opacity="0.22"/>')
    for name in random_names:
        points = [
            (
                scale(layer, 45, 63, x0, x0 + panel_width),
                scale(float(all_a[name][str(layer)]["estimate"]), y_low, y_high, bottom, top),
            )
            for layer in range(46, 64)
        ]
        parts.append(
            polyline(points, fill="none", stroke="#999", stroke_width="1.2", opacity="0.75")
        )
    official_points = [
        (
            scale(layer, 45, 63, x0, x0 + panel_width),
            scale(
                float(all_a["official_p25"][str(layer)]["estimate"]),
                y_low,
                y_high,
                bottom,
                top,
            ),
        )
        for layer in range(46, 64)
    ]
    parts.append(polyline(official_points, fill="none", stroke="#1764ab", stroke_width="4"))
    parts.append(
        f'<text x="{x0 + panel_width / 2}" y="{bottom + 43}" '
        'text-anchor="middle" class="axis">decoder layer</text>'
    )
    parts.append(
        f'<text x="{x0 + 8}" y="{top + 17}" class="label" fill="#1764ab">official p25</text>'
    )
    parts.append(
        f'<text x="{x0 + 8}" y="{top + 34}" class="label" fill="#777">4 random references</text>'
    )

    # Panel B: arm breakdown at preregistered layers.
    x0 = lefts[1]
    all_b = stats["panel_b"]
    b_values = [
        float(all_b[arm][str(layer)]["estimate"]) for arm in "NRVF" for layer in (53, 60, 63)
    ]
    b_low, b_high = max(0.0, min(b_values) - 0.05), min(1.0, max(b_values) + 0.05)
    parts.append(f'<text x="{x0}" y="28" class="title">B  Arm breakdown</text>')
    parts.append(
        f'<line x1="{x0}" y1="{bottom}" x2="{x0 + panel_width}" y2="{bottom}" stroke="#222"/>'
    )
    parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="#222"/>')
    colors = {"N": "#4c78a8", "R": "#f58518", "V": "#54a24b", "F": "#e45756"}
    layers = (53, 60, 63)
    for layer_index, layer in enumerate(layers):
        center = x0 + 90 + layer_index * 140
        parts.append(
            f'<text x="{center}" y="{bottom + 20}" '
            f'text-anchor="middle" class="axis">L{layer}</text>'
        )
        for arm_index, arm in enumerate("NRVF"):
            summary = all_b[arm][str(layer)]
            x = center + (arm_index - 1.5) * 22
            estimate = float(summary["estimate"])
            low, high = map(float, summary["family_cluster_bootstrap_95ci"])
            y = scale(estimate, b_low, b_high, bottom, top)
            y1 = scale(low, b_low, b_high, bottom, top)
            y2 = scale(high, b_low, b_high, bottom, top)
            parts.append(
                f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                f'stroke="{colors[arm]}" stroke-width="2"/>'
            )
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[arm]}"/>')
    for index, arm in enumerate("NRVF"):
        x = x0 + 65 + index * 85
        parts.append(f'<circle cx="{x}" cy="{top + 12}" r="5" fill="{colors[arm]}"/>')
        parts.append(f'<text x="{x + 10}" y="{top + 16}" class="label">{arm}</text>')
    parts.append(
        f'<text x="{x0 + panel_width / 2}" y="{bottom + 43}" '
        'text-anchor="middle" class="axis">family-bootstrap 95% CI</text>'
    )

    # Panel C: descriptive twelve-family scatter, deliberately no regression line.
    x0 = lefts[2]
    panel_c = stats["panel_c"]
    # Family values are copied from the Panel-A L53/L60 statistics only indirectly;
    # plot_points.csv supplies them to the caller, so this section receives a list.
    points = panel_c["points"]
    xs = [float(row["x_g_off"]) for row in points]
    ys = [float(row["y_delta_m_general"]) for row in points]
    x_pad = max(0.005, 0.1 * (max(xs) - min(xs) or 1.0))
    y_pad = max(0.05, 0.1 * (max(ys) - min(ys) or 1.0))
    x_low, x_high = min(xs) - x_pad, max(xs) + x_pad
    c_low, c_high = min(ys) - y_pad, max(ys) + y_pad
    parts.append(
        f'<text x="{x0}" y="28" class="title">C  Descriptive family-level association</text>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{bottom}" x2="{x0 + panel_width}" y2="{bottom}" stroke="#222"/>'
    )
    parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="#222"/>')
    for row in points:
        x = scale(float(row["x_g_off"]), x_low, x_high, x0, x0 + panel_width)
        y = scale(float(row["y_delta_m_general"]), c_low, c_high, bottom, top)
        label = html.escape(str(row["series"]).replace("CEDAR-DEV-", ""))
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#6f4e7c"/>')
        parts.append(f'<text x="{x + 6}" y="{y - 6}" class="label">{label}</text>')
    rho = float(panel_c["spearman_rho"])
    parts.append(
        f'<text x="{x0 + 10}" y="{top + 18}" class="label">'
        f"descriptive Spearman rho = {rho:.3f}</text>"
    )
    parts.append(
        f'<text x="{x0 + panel_width / 2}" y="{bottom + 43}" '
        'text-anchor="middle" class="axis">family mean G_off</text>'
    )
    parts.append(
        f'<text transform="translate({x0 - 38},{(top + bottom) / 2}) rotate(-90)" '
        'text-anchor="middle" class="axis">family mean delta m_general</text>'
    )

    parts.append(
        '<text x="55" y="497" class="caption">Downstream projections use the fixed layer-53 '
        "cap direction, not a layer-specific Assistant Axis; random references use their "
        "corresponding signed r53. Four references are a direction "
        "battery, not a random null distribution. Panel C is descriptive and has no fitted "
        "line.</text>"
    )

    # Panel D: full-depth native logit-lens profile. Curves are descriptive.
    panel_d = stats["panel_d"]["trace_sources"]
    d_left, d_right = 85.0, 1450.0
    d_width = d_right - d_left
    parts.append('<text x="55" y="550" class="title">D  Layer-resolved emergence</text>')
    cap_x0 = scale(46, 0, 63, d_left, d_right)
    cap_x1 = scale(53, 0, 63, d_left, d_right)
    series_style = {
        "U": {"color": "#1764ab", "dash": "none", "label": "U trace (primary)"},
        "P": {"color": "#c46c14", "dash": "7,4", "label": "P trace (sensitivity)"},
    }

    def depth_panel(metric: str, label: str, plot_top: float, plot_bottom: float) -> None:
        summaries = [
            panel_d[series][metric][str(layer)]
            for series in ("U", "P")
            for layer in range(64)
        ]
        low = min(float(item["family_cluster_bootstrap_95ci"][0]) for item in summaries)
        high = max(float(item["family_cluster_bootstrap_95ci"][1]) for item in summaries)
        padding = max(1e-6, 0.08 * (high - low or 1.0))
        if metric == "kl_opt":
            low = max(0.0, low - padding)
        else:
            low -= padding
        high += padding
        parts.append(
            f'<rect x="{cap_x0:.2f}" y="{plot_top}" width="{cap_x1 - cap_x0:.2f}" '
            f'height="{plot_bottom - plot_top}" fill="#c9d7e5" opacity="0.28"/>'
        )
        parts.append(
            f'<line x1="{d_left}" y1="{plot_bottom}" x2="{d_right}" y2="{plot_bottom}" '
            'stroke="#222"/>'
        )
        parts.append(
            f'<line x1="{d_left}" y1="{plot_top}" x2="{d_left}" y2="{plot_bottom}" '
            'stroke="#222"/>'
        )
        for fraction in (0.0, 0.5, 1.0):
            value = low + fraction * (high - low)
            y = scale(value, low, high, plot_bottom, plot_top)
            parts.append(
                f'<line x1="{d_left - 5}" y1="{y:.2f}" x2="{d_left}" y2="{y:.2f}" '
                'stroke="#222"/>'
            )
            parts.append(
                f'<text x="{d_left - 9}" y="{y + 4:.2f}" text-anchor="end" '
                f'class="axis">{value:.3g}</text>'
            )
        for layer in (0, 16, 32, 46, 53, 63):
            x = scale(layer, 0, 63, d_left, d_right)
            parts.append(
                f'<line x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" '
                f'y2="{plot_bottom + 5}" stroke="#222"/>'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{plot_bottom + 19}" text-anchor="middle" '
                f'class="axis">{layer}</text>'
            )
        for series in ("U", "P"):
            profile = panel_d[series][metric]
            style = series_style[series]
            upper = [
                (
                    scale(layer, 0, 63, d_left, d_right),
                    scale(
                        float(profile[str(layer)]["family_cluster_bootstrap_95ci"][1]),
                        low,
                        high,
                        plot_bottom,
                        plot_top,
                    ),
                )
                for layer in range(64)
            ]
            lower = [
                (
                    scale(layer, 0, 63, d_left, d_right),
                    scale(
                        float(profile[str(layer)]["family_cluster_bootstrap_95ci"][0]),
                        low,
                        high,
                        plot_bottom,
                        plot_top,
                    ),
                )
                for layer in range(64)
            ]
            band = upper + list(reversed(lower))
            band_body = " ".join(f"{x:.2f},{y:.2f}" for x, y in band)
            parts.append(
                f'<polygon points="{band_body}" fill="{style["color"]}" opacity="0.10"/>'
            )
            points = [
                (
                    scale(layer, 0, 63, d_left, d_right),
                    scale(
                        float(profile[str(layer)]["estimate"]),
                        low,
                        high,
                        plot_bottom,
                        plot_top,
                    ),
                )
                for layer in range(64)
            ]
            attrs: dict[str, Any] = {
                "fill": "none",
                "stroke": style["color"],
                "stroke_width": "2.8",
            }
            if style["dash"] != "none":
                attrs["stroke_dasharray"] = style["dash"]
            parts.append(polyline(points, **attrs))
            for layer in (*range(0, 64, 8), 63):
                x, y = points[layer]
                if series == "U":
                    parts.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" '
                        f'fill="{style["color"]}"/>'
                    )
                else:
                    parts.append(
                        f'<rect x="{x - 3:.2f}" y="{y - 3:.2f}" width="6" height="6" '
                        f'fill="{style["color"]}"/>'
                    )
        parts.append(
            f'<text transform="translate(28,{(plot_top + plot_bottom) / 2}) rotate(-90)" '
            f'text-anchor="middle" class="axis">{html.escape(label)}</text>'
        )

    depth_panel("delta_m_general_mean_noninitial", "delta m_general(l)", 585, 765)
    depth_panel("kl_opt", "KL_opt(l)", 835, 1015)
    parts.append(
        f'<text x="{d_left + d_width / 2}" y="1043" text-anchor="middle" '
        'class="axis">decoder layer (cap window 46-53 shaded)</text>'
    )
    for index, series in enumerate(("U", "P")):
        style = series_style[series]
        x = 1080 + index * 175
        dash = "" if style["dash"] == "none" else f' stroke-dasharray="{style["dash"]}"'
        parts.append(
            f'<line x1="{x}" y1="548" x2="{x + 35}" y2="548" '
            f'stroke="{style["color"]}" stroke-width="3"{dash}/>'
        )
        parts.append(
            f'<text x="{x + 42}" y="552" class="label">{style["label"]}</text>'
        )
    parts.append(
        '<text x="55" y="1090" class="caption">cache-faithful replay; divergence below '
        'layer 46 reflects cache-propagated effects, not direct rectification. Logit-lens '
        'readout; descriptive.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    analysis = read_json(output_root / "ANALYSIS.json")
    require(analysis.get("status") == "complete", "Plotting requires completed analysis")
    stats = read_json(output_root / "plot_stats.json")
    with (output_root / "plot_points.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stats["panel_c"]["points"] = [row for row in rows if row["panel"] == "C"]
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    svg_path = figures / "rg1p_stagea_main.svg"
    svg_path.write_text(generate_svg(stats), encoding="utf-8", newline="\n")

    soffice = shutil.which("soffice")
    pdftocairo = shutil.which("pdftocairo")
    require(soffice is not None, "LibreOffice is required for SVG-to-PDF conversion")
    require(pdftocairo is not None, "pdftocairo is required for PNG rendering")
    conversion = subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(figures), str(svg_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    pdf_path = figures / "rg1p_stagea_main.pdf"
    require(
        conversion.returncode == 0 and pdf_path.is_file(),
        f"PDF conversion failed: {conversion.stderr}",
    )
    png_prefix = figures / "rg1p_stagea_main"
    render = subprocess.run(
        [pdftocairo, "-png", "-singlefile", "-r", "180", str(pdf_path), str(png_prefix)],
        check=False,
        capture_output=True,
        text=True,
    )
    png_path = figures / "rg1p_stagea_main.png"
    require(render.returncode == 0 and png_path.is_file(), f"PNG render failed: {render.stderr}")
    artifact_paths = (
        output_root / "SMOKE_AUDIT.json",
        output_root / "FORMAL_AUDIT.json",
        output_root / "ANALYSIS.json",
        output_root / "ANALYSIS.md",
        output_root / "RESULT_SUMMARY_ZH.md",
        output_root / "records.jsonl",
        output_root / "hidden_states.index.json",
        *sorted((output_root / "hidden_states").glob("*.safetensors")),
        output_root / "layer_metrics.csv",
        output_root / "propagation_metrics.jsonl",
        output_root / "plot_points.csv",
        output_root / "plot_stats.json",
        svg_path,
        pdf_path,
        png_path,
    )
    require(all(path.is_file() for path in artifact_paths), "Delivery artifact missing")
    manifest = {
        "schema": "cedar.rg1p_stagea.delivery_manifest.v1",
        "status": "complete",
        "stage_b_started": False,
        "artifacts": {
            str(path.relative_to(output_root)): file_sha256(path) for path in artifact_paths
        },
        "figure_contract": {
            "downstream_p25_axis": "fixed_layer53_exit_direction",
            "random_exit_axis": "condition_specific_signed_layer53_direction",
            "random_reference_count": 4,
            "panel_c_descriptive_no_regression_line": True,
            "panel_d_full_depth_layers": list(range(64)),
            "panel_d_descriptive_logit_lens": True,
        },
    }
    write_json(output_root / "DELIVERY_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
