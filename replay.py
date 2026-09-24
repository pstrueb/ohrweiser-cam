#!/usr/bin/env python3
"""Replay a recorded session through the recognizer and count what it would say.

    poetry run python ohrweiser.py --dump rec/          # record: frames + motion masks
    poetry run python replay.py rec/                    # every announcement + a contact sheet
    poetry run python replay.py rec/ --labels rec.txt   # score against what was really held up

A labels file has one line per sign you held up, as dump-file times:

    094536 094548 J        # held J from 09:45:36 to 09:45:48
    094610 094622 43

An announcement is correct when that line was being held at the time (2 s grace
after the end, for the vote to catch up). Everything else is a false
announcement; a label with no correct announcement is a miss.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from ohrweiser import Voter, read_line

GRACE = 2.0      # seconds a correct announcement may lag the end of its label


def clock(path: str) -> float:
    """Seconds of day from a dump name like 094536.11_frame.png."""
    stamp = os.path.basename(path).split("_")[0]
    return int(stamp[:2]) * 3600 + int(stamp[2:4]) * 60 + float(stamp[4:])


def load_labels(path: str) -> list[tuple[float, float, str]]:
    labels = []
    for row in open(path):
        row = row.split("#")[0].split()
        if row:
            start, end, line = row
            labels.append((clock(start), clock(end), line.upper()))
    return labels


def replay(folder: str) -> tuple[list, int, float]:
    """Return (announcements, frame count, minutes). Each announcement is
    (clock, line, kind, frame path, box)."""
    # mtime, not name: a session that crosses midnight would sort wrongly by name
    frames = sorted(glob.glob(os.path.join(folder, "*_frame.png")), key=os.path.getmtime)
    if not frames:
        raise SystemExit(f"no *_frame.png in {folder}")

    # Older dumps have no recorded mask; rebuild it as the app would have. This is
    # an approximation, since the app learns from every camera frame, not only
    # the ones the OCR thread took.
    background = None
    if not os.path.exists(frames[0].replace("_frame.png", "_fg.png")):
        print("no recorded motion masks: rebuilding them (approximate)", file=sys.stderr)
        background = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=32, detectShadows=False)

    voter, said = Voter(), []
    for n, path in enumerate(frames):
        frame = cv2.imread(path)
        if background is not None:
            foreground = background.apply(frame, learningRate=0.002)
        else:
            foreground = cv2.imread(path.replace("_frame.png", "_fg.png"), cv2.IMREAD_GRAYSCALE)
        hit = read_line(frame, foreground)
        due = voter.push(hit[0] if hit else "", os.path.getmtime(path))
        if due:
            said.append((clock(path), due[0], due[1], path, hit[3]))
        if n % 500 == 0:
            print(f"  {n}/{len(frames)}", end="\r", file=sys.stderr, flush=True)

    minutes = (os.path.getmtime(frames[-1]) - os.path.getmtime(frames[0])) / 60.0
    return said, len(frames), max(minutes, 1e-6)


def score(said: list, labels: list) -> tuple[list, list]:
    """Split announcements into (verdicts, missed labels)."""
    verdicts, heard = [], set()
    for t, line, *_ in said:
        match = next((i for i, (a, b, want) in enumerate(labels)
                      if a <= t <= b + GRACE and want == line), None)
        verdicts.append("ok" if match is not None else "FALSE")
        if match is not None:
            heard.add(match)
    return verdicts, [lab for i, lab in enumerate(labels) if i not in heard]


def contact_sheet(said: list, verdicts: list, out: str, per_row: int = 6) -> None:
    tiles = []
    for (t, line, _kind, path, box), verdict in zip(said, verdicts):
        img = cv2.imread(path)
        x, y, w, h = box
        color = (0, 160, 0) if verdict == "ok" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
        cv2.putText(img, f"{os.path.basename(path)[:9]} {line} {verdict}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (320, 240)))
    if not tiles:
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % per_row)
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    cv2.imwrite(out, np.vstack(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="a directory written by ohrweiser.py --dump")
    p.add_argument("--labels", help="what was really held up, and when")
    p.add_argument("--sheet", default="announcements.png",
                   help="contact sheet of every announcement (default announcements.png)")
    args = p.parse_args()

    said, frames, minutes = replay(args.folder)
    labels = load_labels(args.labels) if args.labels else []
    verdicts, missed = score(said, labels) if labels else (["?"] * len(said), [])

    for (t, line, kind, path, _box), verdict in zip(said, verdicts):
        print(f"{os.path.basename(path)[:9]}  {kind:4} {line:4} {verdict}")
    print(f"\n{frames} frames, {minutes:.1f} min, {len(said)} announcements")
    if labels:
        false = verdicts.count("FALSE")
        print(f"false announcements: {false} ({false / minutes:.2f} per minute)")
        print(f"signs missed: {len(missed)} of {len(labels)}"
              + "".join(f"\n  {line} at {int(a // 3600):02d}{int(a % 3600 // 60):02d}"
                        f"{int(a % 60):02d}" for a, _b, line in missed))
    contact_sheet(said, verdicts, args.sheet)
    if said:
        print(f"contact sheet: {args.sheet}")


if __name__ == "__main__":
    main()
