#!/usr/bin/env python3
"""Visualize atomic-action segments for a processed clip.

Produces, under ``<clip_dir>/viz/``:
  - segments_timeline.png      colored time strip (gray=idle, colored=interaction)
  - segments_contactsheet.png  one mid-frame per interaction segment, labeled
  - segments_annotated.mp4      hand overlays + a per-frame segment banner
and, under ``<clip_dir>/segments_clips/``:
  - interaction_k_<a>-<b>s.mp4  the source video cut to each interaction segment

Usage:
    python scripts/visualize_segments.py --clip-dir output/my_clip \
        --source data/my_clip.mp4
"""

from __future__ import annotations

import argparse
import colorsys
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

IDLE_COLOR = (150, 150, 150)  # BGR gray


def color_for(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.137) % 1.0, 0.65, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def seg_color_map(segs) -> dict:
    cmap, k = {}, 0
    for s in segs:
        if s["label"].startswith("interaction"):
            cmap[s["label"]] = color_for(k)
            k += 1
        else:
            cmap[s["label"]] = IDLE_COLOR
    return cmap


def label_for_frame(segs, t: int) -> str:
    for s in segs:
        if s["start_frame"] <= t < s["end_frame"]:
            return s["label"]
    return "idle"


def display_label(s: dict) -> str:
    """Banner text: semantic label when present (ASCII only — cv2 fonts), else raw label."""
    if s.get("label_en"):
        n = s["label"].split("_")[1] if s["label"].startswith("interaction") else s["label"]
        return f"{n}: {s['label_en']}"
    return s["label"]


def timeline(segs, man, out: Path, W: int = 1100, H: int = 130) -> Path:
    T, fps = man["num_frames"], man["fps"]
    img = np.full((H, W, 3), 255, np.uint8)
    cmap = seg_color_map(segs)
    y0, y1 = 34, 86
    for s in segs:
        x0, x1 = int(s["start_frame"] / T * W), int(s["end_frame"] / T * W)
        cv2.rectangle(img, (x0, y0), (x1, y1), cmap[s["label"]], -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), 1)
        if s["label"].startswith("interaction") and (x1 - x0) > 10:
            cv2.putText(img, s["label"].split("_")[1], (x0 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for sec in range(0, int(T / fps) + 1, 2):
        x = int(sec * fps / T * W)
        cv2.line(img, (x, y1), (x, y1 + 6), (0, 0, 0), 1)
        cv2.putText(img, f"{sec}s", (x - 7, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "action segments  (gray = idle, colored = interaction)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(out), img)
    return out


def contact_sheet(clip_dir: Path, segs, out: Path, cols: int = 5,
                  cell: tuple[int, int] = (320, 180)) -> tuple[Path, int]:
    """One mid-frame per interaction segment; if there are none, fall back to
    frames sampled uniformly across the clip (so the sheet is never blank).

    Returns (path, n_cells).
    """
    cmap = seg_color_map(segs)
    inter = [s for s in segs if s["label"].startswith("interaction")]
    if inter:
        cells = [((s["start_frame"] + s["end_frame"]) // 2,
                  f"{display_label(s)}  {s['start_time']:.1f}-{s['end_time']:.1f}s",
                  cmap[s["label"]]) for s in inter]
    else:
        total = max((s["end_frame"] for s in segs), default=1)
        n = min(10, max(1, total))
        idxs = np.linspace(0, max(total - 1, 0), n).round().astype(int)
        cells = [(int(fi), f"{label_for_frame(segs, int(fi))}  f{int(fi)}",
                  cmap.get(label_for_frame(segs, int(fi)), IDLE_COLOR)) for fi in idxs]

    cw, ch = cell
    pad, banner = 8, 24
    rows = max(1, (len(cells) + cols - 1) // cols)
    sheet = np.full((rows * (ch + banner + pad) + pad, cols * (cw + pad) + pad, 3), 255, np.uint8)
    hands_dir = clip_dir / "viz" / "hands"
    for idx, (frame_idx, text, color) in enumerate(cells):
        img = cv2.imread(str(hands_dir / f"hands_{frame_idx:06d}.png"))
        img = cv2.resize(img, (cw, ch)) if img is not None else np.zeros((ch, cw, 3), np.uint8)
        r, c = divmod(idx, cols)
        y, x = pad + r * (ch + banner + pad), pad + c * (cw + pad)
        cv2.rectangle(sheet, (x, y), (x + cw, y + banner), color, -1)
        cv2.putText(sheet, text, (x + 4, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        sheet[y + banner : y + banner + ch, x : x + cw] = img
    cv2.imwrite(str(out), sheet)
    return out, len(cells)


def annotated_video(clip_dir: Path, segs, man, out: Path) -> Path:
    T, fps = man["num_frames"], man["fps"]
    cmap = seg_color_map(segs)
    hands_dir = clip_dir / "viz" / "hands"
    first = cv2.imread(str(hands_dir / "hands_000000.png"))
    H, W = first.shape[:2]
    banner = 28
    by_label = {s["label"]: s for s in segs}
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + banner))
    for t in range(T):
        img = cv2.imread(str(hands_dir / f"hands_{t:06d}.png"))
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        lab = label_for_frame(segs, t)
        text = display_label(by_label[lab]) if lab in by_label else lab
        canvas = np.full((H + banner, W, 3), 255, np.uint8)
        canvas[banner:] = img
        cv2.rectangle(canvas, (0, 0), (W, banner), cmap.get(lab, IDLE_COLOR), -1)
        cv2.putText(canvas, f"{text}   t={t / fps:5.2f}s   frame {t}", (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.line(canvas, (int(t / T * W), 0), (int(t / T * W), banner), (0, 0, 0), 1)
        writer.write(canvas)
    writer.release()
    return out


def cut_clips(src: Path, segs, out_dir: Path) -> list[Path]:
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found; skipping clip cutting")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for s in segs:
        if not s["label"].startswith("interaction"):
            continue
        slug = "_" + s["label_en"].replace(" ", "-") if s.get("label_en") else ""
        o = out_dir / f"{s['label']}{slug}_{s['start_time']:.1f}-{s['end_time']:.1f}s.mp4"
        subprocess.run(
            [ff, "-y", "-i", str(src), "-ss", str(s["start_time"]), "-to", str(s["end_time"]),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(o)],
            check=True, capture_output=True,
        )
        made.append(o)
    return made


def to_gif(mp4: Path, speed: float = 0.5, fps: int = 10, width: int = 640) -> Path | None:
    """Convert an annotated mp4 to a GIF via the ffmpeg palette two-step."""
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found; skipping GIF")
        return None
    gif = mp4.with_suffix(".gif")
    vf = (f"setpts={speed}*PTS,fps={fps},scale={width}:-1:flags=lanczos,"
          f"split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer")
    subprocess.run([ff, "-y", "-i", str(mp4), "-vf", vf, str(gif)], check=True, capture_output=True)
    return gif


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--source", help="source video, to cut interaction sub-clips")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated mp4")
    ap.add_argument("--gif", action="store_true", help="also convert the annotated mp4 to a GIF")
    args = ap.parse_args()

    clip_dir = Path(args.clip_dir)
    segs = json.load(open(clip_dir / "segments.json"))
    man = json.load(open(clip_dir / "manifest.json"))
    viz = clip_dir / "viz"
    viz.mkdir(parents=True, exist_ok=True)

    t = timeline(segs, man, viz / "segments_timeline.png")
    cs, n = contact_sheet(clip_dir, segs, viz / "segments_contactsheet.png")
    print(f"timeline      -> {t}")
    print(f"contact sheet -> {cs}  ({n} cells)")
    if not args.no_video:
        av = annotated_video(clip_dir, segs, man, viz / "segments_annotated.mp4")
        print(f"annotated mp4 -> {av}")
        if args.gif and (g := to_gif(av)):
            print(f"annotated gif -> {g}")
    if args.source:
        clips = cut_clips(Path(args.source), segs, clip_dir / "segments_clips")
        print(f"cut {len(clips)} interaction clips -> {clip_dir / 'segments_clips'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
