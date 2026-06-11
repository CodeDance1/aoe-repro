#!/usr/bin/env python3
"""Fetch an EgoDex sample for hand-pose benchmarking.

EgoDex (Apple) ships egocentric video with 3D hand-joint ground truth captured
on Apple Vision Pro (ARKit + on-device SLAM), making it the lowest-friction GT
target for reproducing the paper's hand-pose precision.

Source / docs: https://github.com/apple/ml-egodex

Usage:
    # show access instructions
    python scripts/download_egodex.py

    # if the release is mirrored on the Hugging Face Hub, fetch a subset:
    python scripts/download_egodex.py --hf-repo <org/dataset> --dest datasets/egodex \
        --allow "test/*"

This script intentionally does not hard-code a download URL: confirm the current
distribution channel from the GitHub repo above, then pass it explicitly.
"""

from __future__ import annotations

import argparse
import sys

INSTRUCTIONS = """\
EgoDex access
=============
1. Open https://github.com/apple/ml-egodex and follow the dataset download
   instructions there (it documents the current host + any usage terms).
2. The test split (~7 hours) is enough to reproduce hand-pose precision.
3. Each sample provides RGB video + 3D hand-joint GT. After downloading, point
   the eval at a clip:

     aoe-pipeline run --video <egodex_clip.mp4> --output-dir output
     aoe-pipeline eval-hands --pred output/<clip>/hands/joints_world.npy \\
         --gt <egodex_clip_joints.npy>

   Use src/aoe_pipeline/eval/joint_maps.py to align joint ordering to EgoDex's
   layout before trusting absolute MPJPE.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch an EgoDex sample.")
    ap.add_argument("--hf-repo", help="Hugging Face dataset repo id, if mirrored")
    ap.add_argument("--dest", default="datasets/egodex")
    ap.add_argument("--allow", nargs="*", default=None, help="allow_patterns for snapshot")
    args = ap.parse_args()

    if not args.hf_repo:
        print(INSTRUCTIONS)
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -e '.[download]'", file=sys.stderr)
        return 1

    path = snapshot_download(
        repo_id=args.hf_repo, repo_type="dataset",
        local_dir=args.dest, allow_patterns=args.allow,
    )
    print(f"Downloaded to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
