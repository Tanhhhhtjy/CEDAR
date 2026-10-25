"""Draw the CEDAR Figure 3 mechanism-chain draft.

This figure combines three audited results:
  A. E2B fixed-trace replay (commit 6a7e286).
  B. RG1-P downstream off-exit-axis propagation (commit bb0bd6b).
  C. Auxiliary layer-60 V/F decodability diagnostic from RG1-P.

The RG1 decomposition is deliberately described as an *axis line*:
span(a_Assistant) == span(v_cap), although a_Assistant ~= -v_cap.
Off-axis energy is therefore invariant to the sign convention.
"""

from __future__ import annotations

from pathlib import Path

from reportlab import rl_config
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Polygon, Rect, String
from reportlab.lib.colors import Color, HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


rl_config.invariant = 1

WIDTH_PT = 7.0 * 72
HEIGHT_PT = 2.78 * 72

INK = HexColor("#172231")
MUTED = HexColor("#617080")
GRID = HexColor("#D9E0E7")
PANEL_BG = HexColor("#F7F9FB")
CAP_SHADE = HexColor("#EDF2F7")
BLUE = HexColor("#2F6FB3")
BLUE_LIGHT = HexColor("#AFC8E5")
TEAL = HexColor("#0E8C7A")
ORANGE = HexColor("#DC542A")
GOLD = HexColor("#D98E04")
GRAY = HexColor("#929EAA")
LIGHT_GRAY = HexColor("#C8D0D8")
WHITE = HexColor("#FFFFFF")


E2B_ROWS = [
    # label, estimate, low, high, color, marker
    ("U trace", 2.078125, 1.7591145833, 2.4079861111, BLUE, "circle"),
    ("P trace", 0.6996527778, 0.4865451389, 0.9066840278, BLUE, "square"),
    ("U trace", -2.421875, -3.4114583333, -1.4479166667, ORANGE, "circle"),
    ("P trace", -0.796875, -1.21875, -0.390625, ORANGE, "square"),
]

LAYERS = list(range(53, 64))
OFFICIAL_REL = [
    0.0,
    0.035289053,
    0.048797490,
    0.072866617,
    0.093973584,
    0.124695764,
    0.131887747,
    0.160802530,
    0.169676085,
    0.198147204,
    0.231160678,
]
RANDOM_REFS_REL = [
    [
        0.0,
        0.000029677,
        0.000031005,
        0.000005522,
        -0.000067267,
        0.000010556,
        0.000039809,
        0.000041418,
        -0.000088813,
        -0.000033032,
        0.000004960,
    ],
    [
        0.0,
        0.000038279,
        0.000038709,
        0.000021098,
        -0.000032490,
        0.000010422,
        0.000021380,
        -0.000029016,
        -0.000197964,
        -0.000148058,
        0.000015064,
    ],
    [
        0.0,
        0.000103812,
        0.000052690,
        0.000139250,
        0.000114478,
        -0.000018902,
        -0.000179983,
        -0.000069518,
        -0.000102237,
        -0.000409904,
        -0.000161740,
    ],
    [
        0.0,
        0.000113664,
        0.000056093,
        0.000141735,
        0.000124095,
        -0.000030646,
        -0.000186041,
        -0.000074735,
        -0.000182550,
        -0.000368633,
        -0.000048814,
    ],
]

PROBE_ROWS = [
    ("uncapped state", 0.7690972222, TEAL, "circle"),
    ("p25 state", 0.7673611111, BLUE, "square"),
    ("text only", 0.5112847222, GRAY, "triangle"),
]


def _register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("CedarSans", str(regular)))
        pdfmetrics.registerFont(TTFont("CedarSans-Bold", str(bold)))
        return "CedarSans", "CedarSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7.2,
    color=INK,
    bold: bool = False,
    anchor: str = "start",
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName=FONT_BOLD if bold else FONT,
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def marker(drawing: Drawing, x: float, y: float, color, kind: str, radius: float = 3.2) -> None:
    if kind == "circle":
        drawing.add(Circle(x, y, radius, fillColor=color, strokeColor=WHITE, strokeWidth=0.8))
    elif kind == "square":
        drawing.add(
            Rect(
                x - radius,
                y - radius,
                radius * 2,
                radius * 2,
                fillColor=color,
                strokeColor=WHITE,
                strokeWidth=0.8,
            )
        )
    elif kind == "triangle":
        drawing.add(
            Polygon(
                [
                    x,
                    y + radius + 0.5,
                    x - radius - 0.5,
                    y - radius,
                    x + radius + 0.5,
                    y - radius,
                ],
                fillColor=color,
                strokeColor=WHITE,
                strokeWidth=0.8,
            )
        )
    else:
        raise ValueError(kind)


def panel_label(drawing: Drawing, x: float, y: float, label: str, title: str) -> None:
    add_text(drawing, x, y, label, size=10.0, bold=True)
    add_text(drawing, x + 16, y, title, size=9.0, bold=True)


def draw_panel_a(drawing: Drawing, x0: float, y0: float, width: float, height: float) -> None:
    panel_label(drawing, x0, y0 + height - 10, "A", "Frozen trace, shifted boundary")

    plot_left = x0 + 48
    plot_right = x0 + width - 6
    plot_top = y0 + height - 31
    plot_bottom = y0 + 25
    xmin, xmax = -3.8, 2.8

    def sx(value: float) -> float:
        return plot_left + (value - xmin) / (xmax - xmin) * (plot_right - plot_left)

    drawing.add(
        Rect(
            x0,
            y0 + 3,
            width,
            height - 23,
            fillColor=PANEL_BG,
            strokeColor=None,
        )
    )

    for tick in (-3, -2, -1, 0, 1, 2):
        x = sx(tick)
        drawing.add(
            Line(
                x,
                plot_bottom,
                x,
                plot_top,
                strokeColor=GRID if tick else MUTED,
                strokeWidth=0.55 if tick else 0.8,
                strokeDashArray=[2, 2] if tick == 0 else None,
            )
        )
        add_text(drawing, x, plot_bottom - 10, f"{tick:+d}" if tick else "0", size=6.5, color=MUTED, anchor="middle")

    ys = [plot_top - 17, plot_top - 39, plot_top - 75, plot_top - 97]
    add_text(drawing, x0 + 5, plot_top - 3, "broad displacement", size=6.9, color=BLUE, bold=True)
    add_text(drawing, x0 + 5, plot_top - 61, "V−F selectivity", size=6.9, color=ORANGE, bold=True)
    drawing.add(Line(x0 + 5, plot_top - 55, plot_right, plot_top - 55, strokeColor=GRID, strokeWidth=0.7))

    for (label, est, lo, hi, color, kind), y in zip(E2B_ROWS, ys):
        add_text(drawing, x0 + 7, y - 2.5, label, size=6.7, color=INK)
        drawing.add(Line(sx(lo), y, sx(hi), y, strokeColor=color, strokeWidth=1.5))
        drawing.add(Line(sx(lo), y - 3, sx(lo), y + 3, strokeColor=color, strokeWidth=0.9))
        drawing.add(Line(sx(hi), y - 3, sx(hi), y + 3, strokeColor=color, strokeWidth=0.9))
        marker(drawing, sx(est), y, color, kind)
        offset = 5 if est >= 0 else -5
        anchor = "start" if est >= 0 else "end"
        add_text(
            drawing,
            sx(est) + offset,
            y + 5,
            f"{est:+.2f}",
            size=6.8,
            color=color,
            bold=True,
            anchor=anchor,
        )

    add_text(
        drawing,
        (plot_left + plot_right) / 2,
        y0 + 7,
        "p25 − uncapped (logits)",
        size=6.7,
        color=MUTED,
        anchor="middle",
    )


def draw_panel_b(drawing: Drawing, x0: float, y0: float, width: float, height: float) -> None:
    panel_label(drawing, x0, y0 + height - 10, "B", "Correction leaves the cap axis line")

    drawing.add(
        Rect(
            x0,
            y0 + 3,
            width,
            height - 23,
            fillColor=PANEL_BG,
            strokeColor=None,
        )
    )

    plot_left = x0 + 35
    plot_right = x0 + width - 8
    plot_top = y0 + height - 43
    plot_bottom = y0 + 39
    ymin, ymax = -0.02, 0.25

    def sx(layer: float) -> float:
        return plot_left + (layer - 53) / 10 * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom + (value - ymin) / (ymax - ymin) * (plot_top - plot_bottom)

    for tick in (0.0, 0.1, 0.2):
        y = sy(tick)
        drawing.add(Line(plot_left, y, plot_right, y, strokeColor=GRID, strokeWidth=0.6))
        add_text(drawing, plot_left - 5, y - 2.5, f"{tick:.1f}", size=6.4, color=MUTED, anchor="end")
    for layer in (53, 55, 57, 60, 63):
        x = sx(layer)
        drawing.add(Line(x, plot_bottom, x, plot_bottom - 3.5, strokeColor=MUTED, strokeWidth=0.6))
        add_text(drawing, x, plot_bottom - 12, str(layer), size=6.4, color=MUTED, anchor="middle")

    drawing.add(
        Rect(
            sx(53) - 3,
            plot_bottom,
            6,
            plot_top - plot_bottom,
            fillColor=CAP_SHADE,
            strokeColor=None,
        )
    )
    drawing.add(Line(sx(53), plot_bottom, sx(53), plot_top, strokeColor=MUTED, strokeWidth=0.75, strokeDashArray=[2, 2]))
    add_text(drawing, sx(53) + 3, plot_top - 9, "cap exit", size=6.2, color=MUTED)

    for values in RANDOM_REFS_REL:
        points: list[float] = []
        for layer, value in zip(LAYERS, values):
            points.extend([sx(layer), sy(value)])
        drawing.add(
            PolyLine(
                points,
                strokeColor=LIGHT_GRAY,
                strokeWidth=0.75,
                strokeDashArray=[2, 2],
                fillColor=None,
            )
        )

    official_points: list[float] = []
    for layer, value in zip(LAYERS, OFFICIAL_REL):
        official_points.extend([sx(layer), sy(value)])
    drawing.add(PolyLine(official_points, strokeColor=BLUE, strokeWidth=2.0, fillColor=None))
    for layer, value in zip(LAYERS, OFFICIAL_REL):
        drawing.add(Circle(sx(layer), sy(value), 1.7, fillColor=BLUE, strokeColor=BLUE))

    x60, y60 = sx(60), sy(OFFICIAL_REL[7])
    drawing.add(Line(x60, sy(0), x60, y60, strokeColor=BLUE, strokeWidth=1.0))
    drawing.add(Line(x60 - 3, sy(0), x60 + 3, sy(0), strokeColor=BLUE, strokeWidth=1.0))
    drawing.add(Line(x60 - 3, y60, x60 + 3, y60, strokeColor=BLUE, strokeWidth=1.0))
    add_text(drawing, x60 + 5, y60 - 1, "Goff = +0.161", size=7.0, color=BLUE, bold=True)
    add_text(drawing, x60 + 5, y60 - 10, "[0.151, 0.170]", size=6.3, color=BLUE)

    add_text(drawing, plot_left + 4, sy(0) + 5, "matched random refs", size=6.1, color=GRAY)
    add_text(drawing, plot_right - 2, sy(OFFICIAL_REL[-1]) + 4, "released p25", size=6.3, color=BLUE, bold=True, anchor="end")

    add_text(
        drawing,
        x0 + 35,
        y0 + height - 29,
        "off-axis energy relative to the layer-53 exit",
        size=6.4,
        color=MUTED,
    )
    add_text(
        drawing,
        x0 + 5,
        y0 + 7,
        "sign-invariant axis line: span(aA) = span(vcap),  aA ≈ −vcap",
        size=5.9,
        color=MUTED,
    )


def draw_panel_c(drawing: Drawing, x0: float, y0: float, width: float, height: float) -> None:
    panel_label(drawing, x0, y0 + height - 10, "C", "Validity stays readable")

    drawing.add(
        Rect(
            x0,
            y0 + 3,
            width,
            height - 23,
            fillColor=PANEL_BG,
            strokeColor=None,
        )
    )

    plot_left = x0 + 42
    plot_right = x0 + width - 8
    plot_top = y0 + height - 44
    plot_bottom = y0 + 43
    xmin, xmax = 0.45, 0.82

    def sx(value: float) -> float:
        return plot_left + (value - xmin) / (xmax - xmin) * (plot_right - plot_left)

    for tick in (0.5, 0.6, 0.7, 0.8):
        x = sx(tick)
        drawing.add(Line(x, plot_bottom, x, plot_top, strokeColor=GRID, strokeWidth=0.6))
        add_text(drawing, x, plot_bottom - 11, f"{tick:.1f}", size=6.4, color=MUTED, anchor="middle")

    drawing.add(Line(sx(0.5), plot_bottom, sx(0.5), plot_top, strokeColor=MUTED, strokeWidth=0.8, strokeDashArray=[2, 2]))
    add_text(drawing, sx(0.5) + 3, plot_top - 6, "chance", size=5.9, color=MUTED)

    ys = [plot_top - 25, plot_top - 55, plot_top - 85]
    for (label, value, color, kind), y in zip(PROBE_ROWS, ys):
        add_text(drawing, x0 + 5, y - 2.5, label, size=6.5, color=INK)
        drawing.add(Line(sx(0.5), y, sx(value), y, strokeColor=color, strokeWidth=2.0))
        marker(drawing, sx(value), y, color, kind, radius=3.3)
        add_text(drawing, min(sx(value) + 5, plot_right - 1), y - 2.5, f"{value:.3f}", size=6.7, color=color, bold=True)

    add_text(drawing, x0 + 5, y0 + 28, "layer-60 family-grouped", size=6.3, color=MUTED)
    add_text(drawing, x0 + 5, y0 + 18, "linear probe · ROC–AUC", size=6.3, color=MUTED)
    add_text(drawing, x0 + 5, y0 + 7, "auxiliary / post-hoc diagnostic", size=5.9, color=MUTED)


def build_figure() -> Drawing:
    drawing = Drawing(WIDTH_PT, HEIGHT_PT)
    drawing.add(Rect(0, 0, WIDTH_PT, HEIGHT_PT, fillColor=WHITE, strokeColor=None))

    outer = 8
    gap = 10
    panel_a_w = 151
    panel_b_w = 194
    panel_c_w = WIDTH_PT - 2 * outer - 2 * gap - panel_a_w - panel_b_w
    y0 = 3
    height = HEIGHT_PT - 6

    draw_panel_a(drawing, outer, y0, panel_a_w, height)
    draw_panel_b(drawing, outer + panel_a_w + gap, y0, panel_b_w, height)
    draw_panel_c(drawing, outer + panel_a_w + gap + panel_b_w + gap, y0, panel_c_w, height)

    return drawing


def export() -> None:
    figures_dir = Path(__file__).resolve().parent.parent
    pdf_path = figures_dir / "cedar-figure3-mechanism-v2.pdf"
    svg_path = figures_dir / "cedar-figure3-mechanism-v2.svg"
    png_path = figures_dir / "cedar-figure3-mechanism-v2.png"

    drawing = build_figure()
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))

    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        image = document[0].render(scale=4.0).to_pil()
        image.save(png_path, dpi=(300, 300))
    except ImportError:
        print("PNG skipped: install pypdfium2 to rasterize the vector PDF.")

    assert abs(drawing.width / 72 - 7.0) < 1e-9
    assert abs(drawing.height / 72 - 2.78) < 1e-9
    print(f"PDF: {pdf_path}")
    print(f"SVG: {svg_path}")
    print(f"PNG: {png_path}")
    print("Canvas: 7.00 × 2.78 in; axis decomposition is invariant to vector sign.")


if __name__ == "__main__":
    export()
