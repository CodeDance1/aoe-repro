#!/usr/bin/env python3
"""Generate a self-contained synthetic egocentric clip (data/sample_clip.mp4).

Pans/zooms a viewport across a large textured canvas to create camera motion with
plenty of trackable features, so the trajectory / depth / segmentation / QC stages
all exercise out of the box. (It contains no real hand, so the hand stage will
report zero detections; point the hand demo at a real egocentric clip — see
README / scripts/download_egodex.py.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_canvas(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = (rng.integers(0, 60, size=(h, w, 3))).astype(np.uint8)
    for _ in range(400):
        c = tuple(int(x) for x in rng.integers(40, 255, size=3))
        pt = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.circle(canvas, pt, int(rng.integers(6, 40)), c, -1)
        else:
            p2 = (pt[0] + int(rng.integers(10, 80)), pt[1] + int(rng.integers(10, 80)))
            cv2.rectangle(canvas, pt, p2, c, -1)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_clip.mp4")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    W, H = args.width, args.height
    canvas = make_canvas(H * 3, W * 3, seed=7)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    for t in range(args.frames):
        a = t / max(args.frames - 1, 1)
        x = int(a * (canvas.shape[1] - W * 1.3))
        y = int((0.3 + 0.4 * np.sin(a * np.pi)) * (canvas.shape[0] - H * 1.3))
        zoom = 1.0 + 0.15 * a
        cw, ch = int(W * zoom), int(H * zoom)
        view = canvas[y : y + ch, x : x + cw]
        writer.write(cv2.resize(view, (W, H)))
    writer.release()
    print(f"wrote {out} ({args.frames} frames @ {args.fps}fps, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
