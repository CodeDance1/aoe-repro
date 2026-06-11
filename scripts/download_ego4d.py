#!/usr/bin/env python3
"""Guide for fetching an Ego4D clip (license-gated).

Ego4D is free but requires accepting a license; AWS credentials are emailed
after approval (~48h) and expire in ~14 days. The paper additionally uses VITRA
annotations on an Ego4D subset.

This script prints the steps and, once you have the `ego4d` CLI + credentials,
shows the exact command to pull a single clip.
"""

from __future__ import annotations

import argparse

STEPS = """\
Ego4D access
============
1. Register and accept the license at https://ego4d-data.org/ and wait for the
   approval email containing AWS credentials.
2. Install the CLI:   pip install ego4d
3. Configure AWS creds (aws configure) with the emailed keys.
4. Download a single clip (replace <UID>):

     ego4d --output_directory datasets/ego4d \\
            --datasets clips --video_uids <UID>

5. Run the pipeline:

     aoe-pipeline run --video datasets/ego4d/v2/clips/<UID>.mp4 --output-dir output

Note: VITRA (https://huggingface.co/datasets/VITRA-VLA/VITRA-1M) provides MANO
hand reconstructions + camera params for an Ego4D subset if you want bundled GT.
"""


def main() -> int:
    argparse.ArgumentParser(description="Ego4D download guide.").parse_args()
    print(STEPS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
