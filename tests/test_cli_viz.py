"""Tests for the `aoe-pipeline run --viz-segments` wiring in the CLI."""

from __future__ import annotations

import inspect

import aoe_pipeline.cli as climod


def test_run_has_viz_segments_option():
    assert "viz_segments" in inspect.signature(climod.run).parameters


def test_run_segment_viz_invokes_script_with_gif(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    climod._run_segment_viz(tmp_path, video)

    cmd = captured["cmd"]
    assert any("visualize_segments.py" in str(c) for c in cmd)
    assert "--clip-dir" in cmd and str(tmp_path) in cmd
    assert "--source" in cmd and str(video) in cmd
    assert "--gif" in cmd


def test_run_segment_viz_skips_when_script_missing(monkeypatch, tmp_path):
    called = {"ran": False}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: called.__setitem__("ran", True))
    # Make the resolved script path look absent -> graceful skip, no subprocess call.
    monkeypatch.setattr(climod.Path, "exists", lambda self: False)
    climod._run_segment_viz(tmp_path, tmp_path / "v.mp4")
    assert called["ran"] is False
