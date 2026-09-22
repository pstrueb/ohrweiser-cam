#!/usr/bin/env python3
"""Generate a printable A4 PDF of line signs to hold in front of the camera.

    poetry run python make_signs.py            # default set
    poetry run python make_signs.py A B 1 2 71 # your own set

Four cards per page, dashed cut lines, big bold glyphs.
"""

import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DPI = 150
A4 = (int(8.27 * DPI), int(11.69 * DPI))
DEFAULT = ["A", "B", "D", "N", "O", "U", "1", "2", "3", "6", "43", "71"]


def kind(line: str) -> str:
    return "Tram" if line.isdigit() else "Bus"


def draw_card(page: ImageDraw.ImageDraw, box, text: str) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0

    for i in range(x0, x1, 20):                      # dashed cut border
        page.line([(i, y0), (min(i + 10, x1), y0)], fill="#bbbbbb", width=2)
        page.line([(i, y1), (min(i + 10, x1), y1)], fill="#bbbbbb", width=2)
    for i in range(y0, y1, 20):
        page.line([(x0, i), (x0, min(i + 10, y1))], fill="#bbbbbb", width=2)
        page.line([(x1, i), (x1, min(i + 10, y1))], fill="#bbbbbb", width=2)

    pt = int(h * 0.55)
    while pt > 10:
        font = ImageFont.truetype(FONT, pt)
        bb = page.textbbox((0, 0), text, font=font)
        if bb[2] - bb[0] <= w * 0.7 and bb[3] - bb[1] <= h * 0.55:
            break
        pt = int(pt * 0.92)

    cx, cy = x0 + w / 2, y0 + h / 2
    page.text((cx - (bb[2] - bb[0]) / 2 - bb[0], cy - (bb[3] - bb[1]) / 2 - bb[1] - h * 0.04),
              text, font=font, fill="black")
    page.text((cx, y1 - h * 0.11), f"{kind(text)} {text}",
              font=ImageFont.truetype(LABEL_FONT, 26), fill="#999999", anchor="mm")


def main() -> None:
    lines = sys.argv[1:] or DEFAULT
    margin, pages = 60, []

    for start in range(0, len(lines), 4):
        page = Image.new("RGB", A4, "white")
        draw = ImageDraw.Draw(page)
        cw = (A4[0] - 2 * margin) // 2
        ch = (A4[1] - 2 * margin) // 2
        for idx, text in enumerate(lines[start:start + 4]):
            col, row = idx % 2, idx // 2
            x0 = margin + col * cw
            y0 = margin + row * ch
            draw_card(draw, (x0, y0, x0 + cw - 20, y0 + ch - 20), text)
        pages.append(page)

    out = "line_signs.pdf"
    pages[0].save(out, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    print(f"Wrote {out} — {len(lines)} signs on {len(pages)} page(s): {' '.join(lines)}")


if __name__ == "__main__":
    main()
