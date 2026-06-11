#!/usr/bin/env python3
"""Render HaWoR-style Front|Top|Side orthographic views of the reconstructed hands.

For a processed clip, animates the 21-joint hand skeleton over time:
  - camera frame: hand pose relative to the camera (clean, shows reconstruction
    quality from new angles),
  - world frame: hands placed in world coords + the camera-trajectory trail
    (HaWoR-style; qualitative because our monocular VO is up-to-scale).

Outputs <clip_dir>/viz/hand_views_{camera,world}.mp4 (+ small .gif for inline view).

    python scripts/render_hand_views.py --clip-dir output/my_clip --frame both
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from aoe_pipeline.viz import render_multiview, render_world_scene


def to_gif(mp4: Path, speed: float = 0.33, fps: int = 8, width: int = 600) -> Path | None:
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found; skipping GIF")
        return None
    gif = mp4.with_name(mp4.stem + ".gif")
    vf = (f"setpts={speed}*PTS,fps={fps},scale={width}:-1:flags=lanczos,"
          f"split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer")
    subprocess.run([ff, "-y", "-i", str(mp4), "-vf", vf, str(gif)], check=True, capture_output=True)
    return gif


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--frame", choices=["camera", "world", "both", "scene"], default="both")
    ap.add_argument("--camera-depth", choices=["saved", "per_joint"], default="per_joint",
                    help="'per_joint' gives finger thickness in top/side; 'saved' is the planar wrist-anchored reconstruction")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    from pathlib import Path

    if args.frame == "scene":
        mp4 = render_world_scene(args.clip_dir, fps=args.fps)
        print(f"scene  -> {mp4}")
        if not args.no_gif and (gif := to_gif(mp4, speed=1.0, fps=10, width=720)):
            print(f"       -> {gif}")
        return 0

    frames = ["camera", "world"] if args.frame == "both" else [args.frame]
    for fr in frames:
        out = None
        if fr == "camera" and args.camera_depth == "per_joint":
            out = Path(args.clip_dir) / "viz" / "hand_views_camera_perjoint.mp4"
        mp4 = render_multiview(args.clip_dir, frame=fr, fps=args.fps,
                               camera_depth=args.camera_depth, out_mp4=out)
        print(f"{fr:6} -> {mp4}")
        if not args.no_gif:
            gif = to_gif(mp4)
            if gif:
                print(f"       -> {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
