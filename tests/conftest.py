"""Shared test fixtures."""

from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.fixture
def synthetic_clip(tmp_path):
    """A 1-second 320x240 test clip generated with ffmpeg's testsrc."""
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg not available")
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [ff, "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out
