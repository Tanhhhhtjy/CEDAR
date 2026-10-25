"""Replace Figure 2D's cross-model TBD panel with audited forest estimates."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ContentStream, TextStringObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "cedar-figure2-results-v2-paper.pdf"
BASE_FIGURE = ROOT / "cedar-figure2-results-v2-paper-base.pdf"

PAGE_W = 507.6
PAGE_H = 237.6

QWEN = "#0E8C7A"
LLAMA = "#DC542A"
TEXT = "#27313B"
MUTED = "#66727F"
GRID = "#CAD2D8"

ROWS = (
    ("Q broad", 2.0746527778, 1.3715277778, 2.7912543403, QWEN, "circle"),
    ("Q V-F", 0.0416666667, -0.4375, 0.5729166667, QWEN, "diamond"),
    ("L broad", 1.3910590278, 0.9075520833, 1.8650282118, LLAMA, "circle"),
    ("L V-F", 0.1145833333, -0.3645833333, 0.6145833333, LLAMA, "diamond"),
)


def register_fonts() -> None:
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    pdfmetrics.registerFont(TTFont("CedarSans", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("CedarSans-Bold", str(fonts / "arialbd.ttf")))


def x_map(value: float) -> float:
    x0, x1 = 349.0, 499.0
    lo, hi = -1.0, 3.25
    return x0 + (value - lo) / (hi - lo) * (x1 - x0)


def draw_marker(c: canvas.Canvas, x: float, y: float, color: str, kind: str) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color if kind == "circle" else "white")
    c.setLineWidth(1.0)
    if kind == "circle":
        c.circle(x, y, 2.5, stroke=1, fill=1)
        return
    path = c.beginPath()
    path.moveTo(x, y + 3.0)
    path.lineTo(x + 3.0, y)
    path.lineTo(x, y - 3.0)
    path.lineTo(x - 3.0, y)
    path.close()
    c.drawPath(path, stroke=1, fill=1)


def build_overlay() -> bytes:
    register_fonts()
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(PAGE_W, PAGE_H), pageCompression=1)

    # Cover only the former panel D; panels A--C remain untouched underneath.
    c.setFillColor("white")
    c.setStrokeColor("white")
    c.rect(272.0, 0.0, PAGE_W - 272.0, 116.0, stroke=0, fill=1)

    c.setFillColor(TEXT)
    c.setFont("CedarSans-Bold", 9.2)
    c.drawString(276.0, 105.0, "D")
    c.setFont("CedarSans", 9.0)
    c.drawString(301.0, 105.0, "Matched cross-model transport")

    axis_y = 20.5
    c.setStrokeColor(GRID)
    c.setLineWidth(0.55)
    c.line(x_map(-1.0), axis_y, x_map(3.25), axis_y)
    c.setDash(1.5, 1.5)
    c.line(x_map(0.0), axis_y, x_map(0.0), 91.0)
    c.setDash()

    c.setFillColor(MUTED)
    c.setFont("CedarSans", 6.3)
    for tick in (-1, 0, 1, 2, 3):
        x = x_map(float(tick))
        c.setStrokeColor(GRID)
        c.line(x, axis_y - 1.8, x, axis_y + 1.8)
        c.setFillColor(MUTED)
        c.drawCentredString(x, 12.8, str(tick))
    c.drawRightString(499.0, 5.8, "p25 - uncapped (logits)")

    row_ys = (82.0, 66.0, 49.0, 33.0)
    for (label, estimate, low, high, color, marker), y in zip(ROWS, row_ys):
        c.setFillColor(color)
        c.setFont("CedarSans", 6.8)
        c.drawRightString(344.0, y - 2.2, f"{label} {estimate:+.2f}")

        c.setStrokeColor(color)
        c.setLineWidth(1.15)
        c.line(x_map(low), y, x_map(high), y)
        c.line(x_map(low), y - 2.3, x_map(low), y + 2.3)
        c.line(x_map(high), y - 2.3, x_map(high), y + 2.3)
        draw_marker(c, x_map(estimate), y, color, marker)

    c.setFillColor(MUTED)
    c.setFont("CedarSans", 6.3)
    c.drawRightString(499.0, 93.0, "circle: broad   diamond: V-F")

    c.showPage()
    c.save()
    return buffer.getvalue()


def load_base_figure() -> bytes:
    if BASE_FIGURE.exists():
        return BASE_FIGURE.read_bytes()

    repo = ROOT.parent
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "show",
            "HEAD:figures/cedar-figure2-results-v2-paper.pdf",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    BASE_FIGURE.write_bytes(result.stdout)
    return result.stdout


def remove_hidden_placeholders(page, reader: PdfReader) -> None:
    content = ContentStream(page.get_contents(), reader)
    for operands, operator in content.operations:
        if operator == b"Tj" and operands and "TBD" in str(operands[0]):
            operands[0] = TextStringObject("")
        elif operator == b"TJ" and operands:
            operands[0] = ArrayObject(
                TextStringObject("") if "TBD" in str(item) else item
                for item in operands[0]
            )
    page.replace_contents(content)


def main() -> None:
    reader = PdfReader(io.BytesIO(load_base_figure()))
    if len(reader.pages) != 1:
        raise RuntimeError(f"Expected one-page figure, found {len(reader.pages)}")
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    if abs(width - PAGE_W) > 0.1 or abs(height - PAGE_H) > 0.1:
        raise RuntimeError(f"Unexpected canvas: {width:.2f} x {height:.2f} pt")

    remove_hidden_placeholders(page, reader)
    overlay = PdfReader(io.BytesIO(build_overlay())).pages[0]
    page.merge_page(overlay)

    output = FIGURE.with_suffix(".tmp.pdf")
    writer = PdfWriter()
    writer.add_page(page)
    with output.open("wb") as handle:
        writer.write(handle)
    os.replace(output, FIGURE)

    print(f"updated={FIGURE}")
    print("panel_d=qwen(+2.075,+0.042);llama(+1.391,+0.115)")


if __name__ == "__main__":
    main()
