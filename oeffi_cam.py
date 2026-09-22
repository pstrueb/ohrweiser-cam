#!/usr/bin/env python3
"""Öffi-Cam: recognize bus/tram line signs from the webcam and announce them via TTS.

Prototype. Deliberately lightweight:
  * OpenCV for capture + preprocessing
  * Tesseract (CPU) for OCR -- no neural net downloads
  * SVOX Pico (or speech-dispatcher/espeak) for speech

Letters (A, B, 13A ...) -> bus, digits (1, 2, 71 ...) -> tram.
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from collections import Counter, deque

import cv2
import numpy as np
from PIL import Image, ImageTk

try:                       # in-process OCR: ~1 ms a call instead of ~130 ms
    from tesserocr import OEM, PSM, PyTessBaseAPI
    HAVE_TESSEROCR = True
except ImportError:        # falls back to spawning the tesseract binary
    HAVE_TESSEROCR = False

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# How many of the last N readings must agree before we trust (and speak) a line.
# Clutter can produce a 95%-confident letter for a frame or two -- a window frame
# reads as a confident A -- so agreement over time, not confidence, is the filter.
VOTE_WINDOW = 7
VOTE_MIN = 5
MIN_CONF = 50.0          # tesseract per-word confidence
SURROUND_MEAN = 150      # a glyph must sit on bright paper...
SURROUND_STD = 42        # ...that is uniform, not textured like a face or blinds
MAX_BLOBS = 24            # most promising blobs per frame; bounds worst-case cost
NOISE_COMPONENTS = 4000   # above this the frame is speckle, not a sign
MIN_FOREGROUND = 0.04     # A sign is something you HOLD UP: this fraction of the
                          # glyph's box must be moving foreground, not scenery.
                          # A window mullion otherwise reads as a confident 4.
                          # Measured over 1600 real frames: this keeps 100% of
                          # genuine readings and drops 87% of the false ones.
GLYPH_HEIGHTS = (64, 96)  # heights the cropped sign is normalized to before OCR.
                          # No single one works for every glyph: a lone P reads
                          # only at 64, a lone B only at 80+. Cheap to try both.
RESPEAK_AFTER = 6.0      # seconds before the same line is announced again
OCR_EVERY = 1            # frames handed to the OCR thread; it drops stale ones

DEBUG = False            # --debug: log every OCR attempt
DEBUG_DIR = ""           # --dump DIR: also write the ROI + binarized crops there

PHRASES = {
    "de": {"bus": "Buslinie {}", "tram": "Straßenbahn Linie {}"},
    "en": {"bus": "Bus line {}", "tram": "Tram line {}"},
}


# --------------------------------------------------------------------------- TTS
class Speaker:
    """Speaks short strings in a background thread, never blocking the UI.

    Backends, best first. Pico is a small formant-free unit-selection engine and
    sounds markedly less robotic than espeak, at ~2 MB and no model download.
    """

    PICO_LANG = {"de": "de-DE", "en": "en-GB"}

    def __init__(self, lang: str = "de", voice: str = "auto", rate: float = 0.85) -> None:
        self.lang = lang
        self.rate = rate               # 1.0 = engine default, lower = slower
        self.muted = False
        self._q: queue.Queue[str | None] = queue.Queue()
        self.backend = self._pick_backend(voice)
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _pick_backend(voice: str) -> str | None:
        have = shutil.which
        if voice in ("auto", "pico") and have("pico2wave") and (have("play") or have("aplay")):
            return "pico"
        if voice in ("auto", "espeak") and have("spd-say"):
            return "spd"
        if have("espeak-ng"):
            return "espeak"
        return None

    @property
    def available(self) -> bool:
        return self.backend is not None

    def say(self, text: str) -> None:
        if self.backend and not self.muted:
            self._q.put(text)

    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                self._speak(text)
            except Exception:
                pass          # a prototype must never die on a missing audio sink

    def _speak(self, text: str) -> None:
        if self.backend == "pico":
            self._speak_pico(text)
        elif self.backend == "spd":
            # spd-say rate runs -100..+100 around the engine default
            spd = max(-100, min(100, int(round((self.rate - 1.0) * 200))))
            subprocess.run(["spd-say", "-w", "-r", str(spd), "-l", self.lang, text],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            words = max(80, min(400, int(175 * self.rate)))
            subprocess.run(["espeak-ng", "-v", self.lang, "-s", str(words), text],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _speak_pico(self, text: str) -> None:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(["pico2wave", "-l", self.PICO_LANG.get(self.lang, "en-GB"),
                            "-w", wav, text], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.getsize(wav) == 0:
                return
            if shutil.which("play"):
                cmd = ["play", "-q", wav]
                if abs(self.rate - 1.0) > 0.01:
                    # sox 'tempo' stretches time without shifting pitch
                    cmd += ["tempo", "-s", f"{self.rate:.2f}"]
            else:
                cmd = ["aplay", "-q", wav]
            subprocess.run(cmd, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass


# --------------------------------------------------------------------------- OCR
def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Remove lighting gradients by dividing out a heavily blurred copy.

    Backlit desks put a slow brightness ramp across the paper. A single global
    Otsu threshold then eats part of a glyph -- which is exactly how a K loses
    its lower diagonal and comes back as an F.
    """
    # The background is by definition low-frequency, so estimate it on a small
    # copy: blurring at full resolution with this sigma cost ~860 ms a frame.
    h, w = gray.shape
    small = cv2.resize(gray, (max(w // 8, 16), max(h // 8, 16)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(small.shape) / 8.0)
    background = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, background, scale=255)


def _variants(bgr: np.ndarray):
    """Yield (name, binary, flattened gray). Under uneven light one threshold
    keeps a glyph whole where another breaks it, so we try several."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    flat = flatten_illumination(gray)
    for name, src in (("flat", flat), ("raw", gray)):
        _, b = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield name + "-otsu", b, flat
    yield "adaptive", cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12), flat


def _blobs(binary: np.ndarray, flat: np.ndarray, foreground: np.ndarray | None = None):
    """Dark blobs that look like ink sitting on paper.

    The decisive test is the *surround*: a printed glyph is ringed by bright,
    uniform paper. A face, a window blind or a plant is ringed by texture, so
    this is what stops the recognizer inventing letters out of the room.
    """
    h, w = binary.shape
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    if border.mean() < 127:
        binary = cv2.bitwise_not(binary)

    # heal strokes broken up by shadow or print banding before measuring blobs
    ink = cv2.morphologyEx(cv2.bitwise_not(binary), cv2.MORPH_CLOSE,
                           np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n > NOISE_COMPONENTS:      # a featureless frame speckles into ~100k blobs
        return [], binary, labels

    # Reject on size in one vectorized pass. Thresholding a featureless frame can
    # yield ~100k speckles, and looping over those in Python cost over a second.
    if n > 1:
        cws, chs, areas = stats[1:, 2], stats[1:, 3], stats[1:, 4]
        with np.errstate(divide="ignore", invalid="ignore"):
            density = areas / np.maximum(cws * chs, 1)
            aspect = cws / np.maximum(chs, 1)
        ok = ((chs >= 0.12 * h) & (chs <= 1.01 * h) &
              (cws >= 0.02 * w) & (cws <= 0.95 * w) &
              (density >= 0.10) & (density <= 0.92) &     # strokes, not blocks or dust
              (aspect >= 0.07) & (aspect <= 2.2))         # 0.07 keeps a bare I or 1
        order = np.argsort(-areas[ok])[:MAX_BLOBS]
        indices = (np.nonzero(ok)[0] + 1)[order]
    else:
        indices = ()

    found = []
    for i in indices:
        x, y, cw, ch, area = stats[i]

        mx, my = int(cw * 0.45) + 4, int(ch * 0.45) + 4
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + cw + mx), min(h, y + ch + my)
        # measure the paper only: every ink pixel is excluded, not just this
        # glyph's own box, so a neighbouring digit ("43") is not read as texture
        window = flat[y0:y1, x0:x1]
        paper = window[ink[y0:y1, x0:x1] == 0]
        if paper.size < 50:
            continue
        if paper.mean() < SURROUND_MEAN or paper.std() > SURROUND_STD:
            continue

        if foreground is not None:
            # A held sign is foreground; the window frame behind it never is.
            if (foreground[y:y + ch, x:x + cw] > 0).mean() < MIN_FOREGROUND:
                continue

        found.append({"box": (int(x), int(y), int(cw), int(ch)), "area": int(cw * ch),
                      "edge": x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1,
                      "label": i})
    return found, binary, labels


def _group(cands: list) -> list:
    """Anchor on the biggest interior blob, merge only genuinely aligned neighbours.

    Without this a letter sitting near the frame edge gets merged with whatever
    dark strip is behind it -- a J plus a window frame reads as a confident U.
    """
    interior = [c for c in cands if not c["edge"]]
    pool = interior or cands
    if not pool:
        return []
    anchor = max(pool, key=lambda c: c["area"])
    ax, ay, aw, ah = anchor["box"]

    # grow the line outwards: "13A" chains 1 -> 3 -> A, and measuring every gap
    # against the anchor alone would strand the far digit
    keep = [anchor]
    rest = [c for c in pool if c is not anchor]
    changed = True
    while changed:
        changed = False
        for c in list(rest):
            x, y, cw, ch = c["box"]
            overlap = max(0, min(ay + ah, y + ch) - max(ay, y)) / float(max(ah, ch))
            if overlap < 0.6 or not 0.6 <= ch / float(ah) <= 1.5:
                continue
            gap = min(max(kx - (x + cw), x - (kx + kw), 0)
                      for kx, _ky, kw, _kh in (k["box"] for k in keep))
            if gap <= 0.8 * ah:
                keep.append(c)
                rest.remove(c)
                changed = True
    return keep


def _tessdata_path() -> str:
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        return env
    for candidate in sorted(glob.glob("/usr/share/tesseract-ocr/*/tessdata")):
        return candidate + "/"
    return "/usr/share/tesseract-ocr/5/tessdata/"


_thread_local = threading.local()


def _api():
    """One tesseract API per thread. Re-using it is what makes this fast: each
    subprocess call pays ~130 ms of process startup, this pays ~1 ms."""
    api = getattr(_thread_local, "api", None)
    if api is None:
        api = PyTessBaseAPI(path=_tessdata_path(), psm=PSM.SINGLE_CHAR,
                            oem=OEM.LSTM_ONLY)
        api.SetVariable("tessedit_char_whitelist", WHITELIST)
        api.SetVariable("classify_bln_numeric_mode", "0")
        _thread_local.api = api
    return api


_PSM = {6: "SINGLE_BLOCK", 7: "SINGLE_LINE", 10: "SINGLE_CHAR"}


def ocr(image: np.ndarray, psm: int) -> tuple[str, float]:
    """Run tesseract on an image, returning (text, confidence)."""
    if HAVE_TESSEROCR:
        try:
            api = _api()
            api.SetPageSegMode(getattr(PSM, _PSM[psm]))
            api.SetImage(Image.fromarray(image))
            text = re.sub(r"[^A-Z0-9]", "", api.GetUTF8Text().upper())
            conf = api.MeanTextConf()
            return (text, float(conf)) if text and conf > 0 else ("", 0.0)
        except Exception:
            pass                       # fall through to the subprocess path
    return _ocr_subprocess(image, psm)


def _ocr_subprocess(image: np.ndarray, psm: int) -> tuple[str, float]:
    ok, png = cv2.imencode(".png", image)
    if not ok:
        return "", 0.0
    cmd = ["tesseract", "stdin", "stdout", "--psm", str(psm), "--oem", "1",
           "-c", f"tessedit_char_whitelist={WHITELIST}",
           "-c", "classify_bln_numeric_mode=0", "tsv"]
    try:
        out = subprocess.run(cmd, input=png.tobytes(), capture_output=True,
                             timeout=4).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired):
        return "", 0.0

    text, confs = [], []
    for row in out.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[11].strip() == "":
            continue
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        if conf > 0:
            text.append(cols[11].strip())
            confs.append(conf)
    if not text:
        return "", 0.0
    return "".join(text).upper(), float(np.mean(confs))


def classify(raw: str) -> tuple[str, str] | None:
    """Map raw OCR output onto (line, 'bus'|'tram'), or None if it makes no sense."""
    line = re.sub(r"[^A-Z0-9]", "", raw)
    if not (1 <= len(line) <= 3):
        return None
    # single-glyph lookalikes: there is no line "0" or "I", but bus O and tram 1 exist
    line = {"0": "O", "I": "1"}.get(line, line)
    if re.fullmatch(r"[A-Z]", line):                 # A, B, ...
        return line, "bus"
    if re.fullmatch(r"\d{1,2}[A-Z]", line):          # 13A, 7B -- Vienna-style bus
        return line, "bus"
    if re.fullmatch(r"\d{1,2}", line) and int(line) > 0:
        return line, "tram"
    return None


def read_line(frame: np.ndarray,
              foreground: np.ndarray | None = None) -> tuple[str, str, float, tuple] | None:
    """Find and read a line sign anywhere in the frame.

    IMPORTANT: pass the camera image as captured. A mirrored frame reads every
    asymmetric glyph backwards -- a J becomes a confident U, a K becomes an F --
    while symmetric letters like U, W and X still "work", which makes the fault
    look like flaky recognition rather than a flipped image.

    `foreground` is an optional motion mask; when given, a glyph is only accepted
    where the scene actually changed, which rejects permanent fixtures.

    Returns (line, kind, confidence, box) or None.
    """
    best = None
    for name, binary, flat in _variants(frame):
        cands, binary, labels = _blobs(binary, flat, foreground)
        keep = _group(cands)
        if not keep:
            if DEBUG:
                print(f"  [{name}] no glyph on paper", flush=True)
            continue

        # render ONLY the kept components: a plain rectangular crop re-admits ink
        # from blobs we just rejected, and psm 10 fits one glyph to the whole image
        clean = np.full_like(binary, 255)
        clean[np.isin(labels, [c["label"] for c in keep])] = 0

        boxes = [c["box"] for c in keep]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)

        pad = max(6, int(0.25 * (y1 - y0)))
        h, w = clean.shape
        crop = clean[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if crop.size == 0:
            continue

        # Segmentation does not depend on the OCR scale, so it stays out here;
        # only the normalized copy below is rebuilt per height.
        for glyph_height in GLYPH_HEIGHTS:
            scale = glyph_height / float(max(crop.shape[0], 1))
            if not 0.05 < scale < 20:
                continue
            # Both the size and the interpolation matter more than they should:
            # a lone P comes back empty at native size and under INTER_AREA, and
            # a lone B only reads at 80px or more.
            scaled = cv2.resize(crop, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_LINEAR)
            scaled = cv2.copyMakeBorder(scaled, 30, 30, 30, 30,
                                        cv2.BORDER_CONSTANT, value=255)

            # psm 10 = single character, psm 7 = single text line, psm 6 = block.
            # No single mode reads every glyph: a lone O or Q only comes back
            # under psm 6, so the fallbacks are load-bearing.
            for psm in ((10, 6, 7) if len(keep) == 1 else (7, 6)):
                raw, conf = ocr(scaled, psm)
                hit = classify(raw)
                if DEBUG:
                    print(f"  [{name}] h={glyph_height} n={len(keep)} psm={psm} "
                          f"raw={raw!r} conf={conf:5.1f} -> {hit}", flush=True)
                if hit and conf >= MIN_CONF:
                    # "43" at 96.8% must beat "3" at 96.9%: a partial read of a
                    # multi-glyph sign is wrong, not merely less confident
                    score = conf + 10.0 * (len(hit[0]) - 1)
                    if best is None or score > best[4]:
                        best = (hit[0], hit[1], conf, (x0, y0, x1 - x0, y1 - y0), score)
                if hit and conf >= 85:
                    break
            if best is not None and best[2] >= 85:
                break          # confident already; skip the second scale
        if best is not None and best[2] >= 85:
            break              # ...and the other binarizations

    if DEBUG:
        print(f"=> {best}", flush=True)
    return best[:4] if best else None


def mirror_box(box: tuple, width: int) -> tuple:
    """Move a box from captured coordinates into mirrored preview coordinates."""
    x, y, w, h = box
    return width - (x + w), y, w, h


# ---------------------------------------------------------------------------- UI
class OeffiCam:
    COLORS = {"bus": "#1d6fd6", "tram": "#d63b1d", "none": "#555555"}

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.speaker = Speaker(args.lang, args.voice, args.rate)
        self.votes: deque[str] = deque(maxlen=VOTE_WINDOW)
        self.confirmed: tuple[str, str] | None = None
        self.last_spoken = ("", 0.0)
        self.frame_no = 0
        self.fps_t, self.fps = time.time(), 0.0
        self.last_conf = 0.0
        self.last_box: tuple | None = None
        self.running = True

        # OCR runs off the UI thread: the pipeline costs ~100 ms and would
        # otherwise stutter the preview.
        # Learns the static scene so fixtures (a window frame, a poster) are
        # ignored. The rate is slow enough that a sign held still for ~20 s is
        # still treated as foreground.
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=32, detectShadows=False)
        self._pending: np.ndarray | None = None
        self._lock = threading.Lock()
        self._results: queue.Queue = queue.Queue()

        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Could not open camera {args.camera}. Another program may be holding "
                f"it -- close any other Öffi-Cam window, or try --camera 1.")

        threading.Thread(target=self._ocr_loop, daemon=True).start()
        self._build_ui()
        self.root.after(10, self.tick)

    # -- widgets
    def _build_ui(self) -> None:
        self.root.title("Öffi-Cam — Linienerkennung")
        self.root.configure(bg="#111111")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.video = tk.Label(self.root, bg="#111111")
        self.video.pack(padx=10, pady=(10, 6))

        self.result = tk.Label(self.root, text="—", font=("DejaVu Sans", 46, "bold"),
                               fg="#888888", bg="#111111")
        self.result.pack()

        self.detail = tk.Label(self.root, text="Halte ein Schild vor die Kamera",
                               font=("DejaVu Sans", 12), fg="#aaaaaa", bg="#111111")
        self.detail.pack(pady=(0, 6))

        bar = tk.Frame(self.root, bg="#111111")
        bar.pack(pady=(0, 4))
        self.mute_btn = tk.Button(bar, text="🔊 Ton an", width=11, command=self.toggle_mute)
        self.mute_btn.pack(side=tk.LEFT, padx=4)
        self.lang_btn = tk.Button(bar, text=f"Sprache: {self.args.lang.upper()}", width=13,
                                  command=self.toggle_lang)
        self.lang_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Nochmal sagen", width=13, command=self.repeat).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Beenden", width=9, command=self.quit).pack(side=tk.LEFT, padx=4)

        speed = tk.Frame(self.root, bg="#111111")
        speed.pack(pady=(0, 6))
        tk.Label(speed, text="Sprechtempo", font=("DejaVu Sans", 9),
                 fg="#aaaaaa", bg="#111111").pack(side=tk.LEFT, padx=(0, 6))
        self.rate_scale = tk.Scale(speed, from_=0.6, to=1.2, resolution=0.05,
                                   orient=tk.HORIZONTAL, length=220, showvalue=True,
                                   bg="#111111", fg="#aaaaaa", highlightthickness=0,
                                   troughcolor="#333333", command=self.set_rate)
        self.rate_scale.set(self.args.rate)
        self.rate_scale.pack(side=tk.LEFT)

        self.status = tk.Label(self.root, text="", font=("DejaVu Sans", 9),
                               fg="#666666", bg="#111111")
        self.status.pack(pady=(0, 6))

        self.root.bind("<q>", lambda _e: self.quit())
        self.root.bind("<Escape>", lambda _e: self.quit())
        self.root.bind("<m>", lambda _e: self.toggle_mute())
        self.root.bind("<space>", lambda _e: self.repeat())

        if not self.speaker.available:
            self.status.config(text="Kein TTS gefunden — nur Anzeige")

    # -- controls
    def set_rate(self, value) -> None:
        self.speaker.rate = float(value)

    def toggle_mute(self) -> None:
        self.speaker.muted = not self.speaker.muted
        self.mute_btn.config(text="🔇 Stumm" if self.speaker.muted else "🔊 Ton an")

    def toggle_lang(self) -> None:
        self.args.lang = "en" if self.args.lang == "de" else "de"
        self.speaker.lang = self.args.lang
        self.lang_btn.config(text=f"Sprache: {self.args.lang.upper()}")

    def repeat(self) -> None:
        if self.confirmed:
            self.announce(*self.confirmed, force=True)

    def announce(self, line: str, kind: str, force: bool = False) -> None:
        now = time.time()
        if not force and self.last_spoken[0] == line and now - self.last_spoken[1] < RESPEAK_AFTER:
            return
        self.last_spoken = (line, now)
        self.speaker.say(PHRASES[self.args.lang][kind].format(line))

    # -- OCR worker
    def _ocr_loop(self) -> None:
        while self.running:
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is None:
                time.sleep(0.01)
                continue
            roi, foreground = pending
            try:
                if DEBUG_DIR:
                    cv2.imwrite(os.path.join(
                        DEBUG_DIR, time.strftime("%H%M%S") + f"{time.time() % 1:.2f}"[1:]
                        + "_frame.png"), roi)
                self._results.put(read_line(roi, foreground))
            except Exception as exc:              # keep the camera alive regardless
                if DEBUG:
                    print("OCR error:", exc, flush=True)

    # -- main loop
    def tick(self) -> None:
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(50, self.tick)
            return

        self.frame_no += 1

        # OCR sees the frame as captured; only the preview is mirrored. Mirroring
        # before OCR reads every asymmetric glyph backwards (J -> U, K -> F).
        foreground = self.background.apply(frame, learningRate=0.002)
        if self.frame_no % OCR_EVERY == 0:
            with self._lock:
                self._pending = (frame.copy(), foreground.copy())

        # Off by default: a mirrored preview shows the printed letter backwards,
        # which is alarming when you are holding a sign up to check it. OCR was
        # never affected -- it always sees the frame as captured.
        display = cv2.flip(frame, 1) if self.args.mirror else frame

        while not self._results.empty():
            hit = self._results.get()
            self.votes.append(hit[0] if hit else "")
            self.last_conf = hit[2] if hit else 0.0
            self.last_box = hit[3] if hit else None
            self.update_decision()

        kind = self.confirmed[1] if self.confirmed else "none"
        color = {"bus": (214, 111, 29), "tram": (29, 59, 214), "none": (120, 120, 120)}[kind]
        if self.last_box is not None:
            x, y, w, h = (mirror_box(self.last_box, display.shape[1])
                          if self.args.mirror else self.last_box)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
            if self.confirmed:
                cv2.putText(display, self.confirmed[0], (x, max(28, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=img)
        self.video.image = img  # keep a reference alive

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.fps_t, 1e-3)
        self.fps_t = now
        if self.speaker.available:
            self.status.config(
                text=f"{self.fps:4.1f} fps · Konfidenz {self.last_conf:3.0f}% · "
                     f"Stimme: {self.speaker.backend} · q beenden · m stumm · Leertaste wiederholen")

        self.root.after(1, self.tick)

    def update_decision(self) -> None:
        line, count = Counter(self.votes).most_common(1)[0]
        if line and count >= VOTE_MIN:
            kind = classify(line)[1]
            self.confirmed = (line, kind)
            self.result.config(text=line, fg=self.COLORS[kind])
            self.detail.config(text="Bus" if kind == "bus" else "Straßenbahn",
                               fg=self.COLORS[kind])
            self.announce(line, kind)  # the re-speak cooldown keeps this from nagging
        elif not line and count >= VOTE_MIN:
            self.confirmed = None
            self.result.config(text="—", fg="#888888")
            self.detail.config(text="Halte ein Schild vor die Kamera", fg="#aaaaaa")

    def quit(self) -> None:
        self.running = False
        self.cap.release()
        self.root.destroy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--lang", choices=["de", "en"], default="de", help="speech language")
    p.add_argument("--voice", choices=["auto", "pico", "espeak"], default="auto",
                   help="TTS backend (default auto: pico if installed)")
    p.add_argument("--rate", type=float, default=0.85,
                   help="speech speed, 1.0 = engine default (default 0.85)")
    p.add_argument("--mirror", action="store_true",
                   help="mirror the preview (selfie view); OCR is never mirrored")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--debug", action="store_true", help="log every OCR attempt")
    p.add_argument("--dump", default="", metavar="DIR",
                   help="also write ROI + binarized crops there for inspection")
    args = p.parse_args()

    global DEBUG, DEBUG_DIR
    DEBUG = args.debug
    DEBUG_DIR = args.dump
    if DEBUG_DIR:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    if not shutil.which("tesseract"):
        raise SystemExit("tesseract not found — install with: sudo apt install tesseract-ocr")

    root = tk.Tk()
    OeffiCam(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
