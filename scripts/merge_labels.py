#!/usr/bin/env python3
"""Merge semantic action labels into a clip's segments.json.

Input labels file: JSON array of {id, label_zh, label_en, hand, object,
confidence, ...} where ``id`` matches the segment's ``label`` field
(e.g. "interaction_3"). Matched segments gain label_zh/label_en/hand/object/
confidence fields; the original ``label`` id is preserved.

Usage:
    python scripts/merge_labels.py --clip-dir output/my_clip --labels labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    seg_path = Path(args.clip_dir) / "segments.json"
    segs = json.loads(seg_path.read_text())
    labels = {l["id"]: l for l in json.loads(Path(args.labels).read_text())}

    merged = 0
    for s in segs:
        lab = labels.get(s["label"])
        if not lab:
            continue
        for k in ("label_zh", "label_en", "hand", "object", "confidence"):
            if k in lab:
                s[k] = lab[k]
        merged += 1
    seg_path.write_text(json.dumps(segs, ensure_ascii=False, indent=2))
    print(f"merged {merged} labels into {seg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
