#!/usr/bin/env python3
"""Label a processed clip's interaction segments with a VLM (+ verify voting).

Codifies the "label + adversarial verify" labeling into a portable, API-based
script (no Claude-Code session required): for each interaction segment it samples
N keyframes, asks a hosted VLM for the atomic action label, optionally runs K
adversarial verify votes (adopting a corrected label if the majority refute), and
writes ``<clip_dir>/semantic_labels.json`` — which the ``label`` pipeline stage
then merges into ``segments.json``.

Requires the ``vlm`` extra (``pip install -e '.[vlm]'``) and an API key in the env
(``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``). Use ``--dry-run`` to exercise the
plumbing without any API calls (writes placeholder labels — handy for CI smoke).

    python scripts/label_segments_vlm.py --clip-dir output/<clip> --provider openai --verify 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--keyframes", type=int, default=3)
    ap.add_argument("--verify", type=int, default=2, help="adversarial verify votes (0 = off)")
    ap.add_argument("--out", default=None, help="output path (default <clip-dir>/semantic_labels.json)")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N segments (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; write placeholder labels")
    args = ap.parse_args()

    clip = Path(args.clip_dir)
    segs = json.loads((clip / "segments.json").read_text())
    inter = [s for s in segs if s["label"].startswith("interaction")]
    if args.limit:
        inter = inter[: args.limit]
    if not inter:
        print("no interaction segments; nothing to label")
        return 0

    if not args.dry_run:
        key = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key):
            print(f"error: {key} not set — this needs a {args.provider} API key (or use --dry-run)")
            return 2

    from aoe_pipeline.vlm import label_one  # imported lazily so --help needs no extras

    frames_dir = clip / "frames"
    labels = []
    for s in inter:
        kf = _load_keyframes(frames_dir, s, args.keyframes)
        if not kf:
            print(f"  {s['label']}: no frames found, skipped")
            continue
        if args.dry_run:
            lab = {"label_zh": "占位", "label_en": "dry-run placeholder",
                   "hand": "none", "object": "none", "confidence": 0.0}
        else:
            try:
                lab = label_one(kf, provider=args.provider, model=args.model, verify=args.verify)
            except Exception as exc:  # noqa: BLE001 — clear abort on API/auth errors
                print(f"  {s['label']}: VLM call failed: {exc}")
                return 1
        rec = {"id": s["label"], **{k: lab[k] for k in
               ("label_zh", "label_en", "hand", "object", "confidence") if k in lab}}
        labels.append(rec)
        print(f"  {rec['id']}: {rec.get('label_zh')} / {rec.get('label_en')}")

    out = Path(args.out) if args.out else clip / "semantic_labels.json"
    out.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    print(f"wrote {len(labels)} labels -> {out}")
    return 0


def _load_keyframes(frames_dir: Path, seg: dict, n: int):
    a, b = int(seg["start_frame"]), int(seg["end_frame"])
    idxs = np.linspace(a, max(a, b - 1), min(n, max(1, b - a))).round().astype(int)
    out = []
    for i in idxs:
        img = cv2.imread(str(frames_dir / f"frame_{int(i):06d}.png"))
        if img is not None:
            out.append(img)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
