# Öffi-Cam

Prototype: point the webcam at a printed line sign, and it announces the line out loud.

**Letters are buses** (`A`, `B`, `O`, also `13A`), **digits are trams** (`1`, `2`, `43`, `71`).

## Why it is cheap to run

No model downloads, no GPU, no PyTorch:

| Step | Tool | Cost |
| --- | --- | --- |
| Capture | OpenCV | 640×480 @ 30 fps |
| Segment | Otsu threshold + connected components | sub-millisecond |
| Recognize | Tesseract LSTM in-process (tesserocr), 36-char whitelist | ~1 ms a call, CPU only |
| Speak | SVOX Pico (`pico2wave`), espeak fallback | ~2 MB, no model download |

A full 640x480 frame costs about **8 ms** end to end, so OCR runs on a worker
thread every frame and the preview stays smooth.

Two things dominated the cost before tuning, both worth knowing:

* `tesseract` as a subprocess costs ~130 ms per call in process startup alone,
  and a frame needs several calls. In-process via `tesserocr` it is ~1 ms.
* Estimating the illumination background with a large-sigma Gaussian at full
  resolution cost ~860 ms a frame. The background is low-frequency by
  definition, so it is estimated on a 1/8-scale copy instead.

## Setup

```bash
sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev libttspico-utils sox
poetry install
```

`libtesseract-dev` is needed to build `tesserocr`. Without it the code still
runs, falling back to the (much slower) `tesseract` subprocess.

## Run

```bash
poetry run python oeffi_cam.py                 # German announcements
poetry run python oeffi_cam.py --lang en       # English
poetry run python oeffi_cam.py --camera 2      # pick another /dev/video*
poetry run python oeffi_cam.py --rate 0.7      # slower speech
poetry run python oeffi_cam.py --mirror        # selfie-view preview
```

Hold a sign up anywhere in view; the box shows what it locked onto. Keys: `q`
quit · `m` mute · `space` repeat. The slider sets speech tempo live.

The preview is **not** mirrored by default, so printed letters read the right way
round. OCR never sees a mirrored frame either way — see below.

## Printable test signs

```bash
poetry run python make_signs.py                # writes line_signs.pdf (12 signs)
poetry run python make_signs.py A B 1 2 71     # or pick your own
```

Four cards per A4 page with cut lines. Print, cut, hold up.

## Tests

```bash
poetry run pytest
```

61 tests render signs synthetically and push them through the real pipeline — no
camera needed, so OCR settings can be tuned without printing anything.

## How a reading becomes an announcement

A single OCR result is never trusted. Each reading joins a 7-frame window, and a
line is confirmed only once **5 of the last 7** frames agree and tesseract's
confidence is ≥ 50%. Confirmed lines are re-announced at most every 6 seconds.

Two lookalikes are resolved by what actually exists in a transit network: a lone
`0` becomes bus **O**, a lone `I` becomes tram **1**.

A glyph must also sit on bright uniform paper *and* fall where the scene changed,
so static scenery is ignored. A sign held motionless for ~20 s is gradually
learned as background; move it slightly to re-trigger.

## Things that bit, and why they are subtle

* **The preview mirror reached the OCR.** `cv2.flip` was applied before
  recognition, so every glyph was read backwards: J came back as a confident U,
  K as an F. Mirror-symmetric letters (U, W, A, M, X) still read correctly, which
  disguised a flipped image as flaky recognition. OCR now always gets the frame
  as captured.
* **A rectangular crop re-admits rejected ink.** A dark strip at the frame edge
  fell inside the crop rectangle, and psm 10 fits one character to the whole
  image, so J + strip scored 93% as a U. Only accepted blobs are rendered now.
* **With no sign present it read the room.** Faces and window blinds produced
  confident letters. A glyph must now sit on a bright, *uniform* surround.
* **Confidence does not separate signal from clutter** -- a window frame reads as
  a 95% A. Agreement over time does, hence 5-of-7 voting.
* **Permanent fixtures read as confident line numbers.** A window mullion
  against bright sky scored a 96% `4`, over and over, because bright uniform sky
  passes the "sits on paper" test. Brightness and confidence cannot separate it;
  what does is that a sign is something you *hold up*. A glyph is now only
  accepted where the scene actually changed. Measured over 1600 frames of a real
  session this kept 100% of genuine readings and removed 87% of false ones.
* **No single OCR configuration reads every glyph.** A lone O or Q only comes
  back under psm 6; a lone P only at 64px tall; a lone B only at 80px or more.
  Hence the scale and psm fallbacks.

## Prototype limits

- Printed, high-contrast, roughly upright signs. Real vehicle displays (LED dot
  matrix, glare, motion) are not handled.
- Steep angles, heavy blur, or small glyphs degrade multi-character lines first —
  `71` may read as `7`. Fill the frame with the sign.
- A bare `I` is not read (it is indistinguishable from `1` and a thin bar).
- Across the alphabet plus digits it scores 38/39 clean and 38/39 in dim uneven
  light, with **zero wrong readings** — failures degrade to "no reading".
- The bus/tram rule is purely syntactic; it does not know real timetables.
