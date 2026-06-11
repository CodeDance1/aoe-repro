#!/usr/bin/env python3
"""Build a Ken-Burns (pan + zoom) egocentric-style clip from a single image.

Useful for exercising the hand-reconstruction path without a dataset download:
point it at any photo containing a clearly visible hand and it synthesizes smooth
camera motion so the trajectory, hand, and QC stages all have something to chew on.

    python scripts/make_hand_clip.py --image hand.jpg --out datasets/hand_clip.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="datasets/hand_clip.mp4")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")
    H0, W0 = img.shape[:2]
    W, H = args.width, args.height
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    for t in range(args.frames):
        a = t / max(args.frames - 1, 1)
        zoom = 1.10 - 0.06 * a                       # gentle dolly-in
        cw, ch = int(W0 / zoom), int(H0 / zoom)
        # gentle diagonal arc at realistic egocentric pace, staying in bounds
        x = int((0.30 + 0.12 * a) * (W0 - cw))
        y = int((0.28 + 0.05 * np.sin(a * np.pi)) * (H0 - ch))
        x = max(0, min(x, W0 - cw))
        y = max(0, min(y, H0 - ch))
        crop = img[y : y + ch, x : x + cw]
        writer.write(cv2.resize(crop, (W, H)))
    writer.release()
    print(f"wrote {out} ({args.frames} frames @ {args.fps}fps from {args.image})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
