"""Offline checks for the recognition pipeline -- no camera required.

Signs are rendered synthetically and pushed through the same code path the
live loop uses, so the OCR settings can be tuned without printing anything.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oeffi_cam import classify, mirror_box, read_line  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BUSES = ["A", "B", "D", "N", "O", "U"]
TRAMS = ["1", "2", "3", "6", "9", "43", "71"]


def render(text: str, angle: float = 0, blur: int = 0, size: int = 320,
           fill: float = 0.55) -> np.ndarray:
    """Draw `text` centered on a white square, optionally rotated and blurred."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    pt = int(size * fill)
    while pt > 8:                       # shrink until the string fits with a margin
        font = ImageFont.truetype(FONT, pt)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= size * 0.8 and box[3] - box[1] <= size * 0.8:
            break
        pt = int(pt * 0.9)

    draw.text(((size - (box[2] - box[0])) / 2 - box[0],
               (size - (box[3] - box[1])) / 2 - box[1]), text, font=font, fill="black")

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if angle:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        frame = cv2.warpAffine(frame, m, (size, size), borderValue=(255, 255, 255))
    if blur:
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    return frame


@pytest.mark.parametrize("text", BUSES)
def test_letters_are_buses(text):
    assert read_line(render(text))[:2] == (text, "bus")


@pytest.mark.parametrize("text", TRAMS)
def test_digits_are_trams(text):
    assert read_line(render(text))[:2] == (text, "tram")


@pytest.mark.parametrize("text", ["13A", "7B"])
def test_digit_letter_combo_is_a_bus(text):
    assert read_line(render(text))[:2] == (text, "bus")


@pytest.mark.parametrize("text", BUSES + TRAMS)
def test_survives_rotation_and_blur(text):
    assert read_line(render(text, angle=8, blur=5))[:2][0] == text


def test_blank_scene_detects_nothing():
    assert read_line(np.full((320, 320, 3), 240, np.uint8)) is None


def test_noise_detects_nothing():
    rng = np.random.default_rng(0)
    assert read_line((rng.random((320, 320, 3)) * 255).astype(np.uint8)) is None


@pytest.mark.parametrize("raw,expected", [
    ("A", ("A", "bus")),
    ("0", ("O", "bus")),        # a lone zero is really bus line O
    ("I", ("1", "tram")),       # ...and a lone I is really tram 1
    ("13A", ("13A", "bus")),
    ("71", ("71", "tram")),
    ("", None),
    ("ABCD", None),
    ("00", None),               # no line zero
])
def test_classify(raw, expected):
    assert classify(raw) == expected


def lit(text: str, strength: float = 0.75, contrast: float = 0.45,
        noise: float = 6, blur: int = 5, size: int = 320) -> np.ndarray:
    """A sheet of paper lit from one side in a dim room: grey ink, strong falloff.

    This is the situation that made a real K come back as an F -- a single global
    threshold lost the lower diagonal.
    """
    img = render(text, size=size).astype(np.float32) / 255.0
    img = 1.0 - (1.0 - img) * contrast
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img *= (1.0 - strength * (0.5 * xx + 0.5 * yy))[..., None]
    rng = np.random.default_rng(1)
    img = np.clip(img * 255 + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (blur, blur), 0) if blur else img


@pytest.mark.parametrize("text", ["K", "A", "E", "F", "M", "X", "2", "7", "43"])
def test_survives_uneven_lighting(text):
    assert read_line(lit(text))[:2][0] == text


@pytest.mark.parametrize("strength", [0.5, 0.75, 0.85])
def test_k_is_not_eaten_into_an_f(strength):
    """Regression: the lower diagonal of K must survive thresholding."""
    assert read_line(lit("K", strength=strength))[:2] == ("K", "bus")


def test_bad_light_fails_safe_rather_than_guessing():
    """A reading we cannot trust must be dropped, never announced as a wrong line."""
    result = read_line(lit("K", strength=0.95, contrast=0.12, noise=14))
    assert result is None or result[0] == "K"


@pytest.mark.parametrize("text", ["J", "K", "F", "L", "P", "R", "2", "7"])
def test_mirrored_frames_must_not_be_fed_to_ocr(text):
    """Regression: the preview is mirrored, OCR must NOT be.

    A mirrored asymmetric glyph is read backwards -- J came back as a confident
    U, K as an F -- while symmetric letters (U, W, X) still read correctly, which
    disguises a flipped image as flaky recognition.
    """
    upright = read_line(render(text))
    assert upright[:2][0] == text
    mirrored = read_line(cv2.flip(render(text), 1))
    assert mirrored is None or mirrored[0] != text


def test_mirror_box_maps_into_preview_coordinates():
    assert mirror_box((10, 20, 30, 40), 640) == (600, 20, 30, 40)
    x, y, w, h = mirror_box(mirror_box((10, 20, 30, 40), 640), 640)
    assert (x, y, w, h) == (10, 20, 30, 40)


def test_clutter_next_to_a_glyph_is_not_merged_into_it():
    """A dark strip at the frame edge must not join the letter (J + strip = U)."""
    frame = render("J")
    frame[:, 0:14] = 40                      # window frame / shadow at the edge
    result = read_line(frame)
    assert result is not None and result[0] == "J"


def test_foreground_mask_rejects_a_static_fixture():
    """A window mullion reads as a confident 4; only held-up signs should count.

    Without the mask the glyph is read; with an all-background mask it is not.
    """
    frame = render("4")
    assert read_line(frame)[:2] == ("4", "tram")
    background_only = np.zeros(frame.shape[:2], np.uint8)
    assert read_line(frame, foreground=background_only) is None


def test_foreground_mask_keeps_a_held_sign():
    frame = render("4")
    moving = np.full(frame.shape[:2], 255, np.uint8)
    assert read_line(frame, foreground=moving)[:2] == ("4", "tram")
