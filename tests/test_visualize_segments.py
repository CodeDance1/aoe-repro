"""Tests for scripts/visualize_segments.py (loaded by path — scripts/ isn't a package)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vs = _load("visualize_segments", "scripts/visualize_segments.py")


def _segments():
    return [
        {"label": "idle", "start_frame": 0, "end_frame": 10, "start_time": 0.0, "end_time": 1.0},
        {"label": "interaction_0", "start_frame": 10, "end_frame": 20, "start_time": 1.0,
         "end_time": 2.0, "label_en": "pick up cup"},
        {"label": "idle", "start_frame": 20, "end_frame": 30, "start_time": 2.0, "end_time": 3.0},
        {"label": "interaction_1", "start_frame": 30, "end_frame": 40, "start_time": 3.0,
         "end_time": 4.0},
    ]


def test_seg_color_map_idle_gray_interactions_distinct():
    cmap = vs.seg_color_map(_segments())
    assert cmap["idle"] == vs.IDLE_COLOR
    assert cmap["interaction_0"] != vs.IDLE_COLOR
    assert cmap["interaction_0"] != cmap["interaction_1"]


def test_label_for_frame():
    segs = _segments()
    assert vs.label_for_frame(segs, 5) == "idle"
    assert vs.label_for_frame(segs, 15) == "interaction_0"
    assert vs.label_for_frame(segs, 35) == "interaction_1"
    assert vs.label_for_frame(segs, 999) == "idle"  # out of range -> default


def test_display_label_semantic_and_plain():
    assert vs.display_label({"label": "interaction_0", "label_en": "pick up cup"}) == "0: pick up cup"
    assert vs.display_label({"label": "interaction_1"}) == "interaction_1"
    assert vs.display_label({"label": "idle"}) == "idle"


def test_to_gif_returns_none_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(vs.shutil, "which", lambda _: None)
    assert vs.to_gif(tmp_path / "x.mp4") is None


def test_timeline_and_contactsheet_write_png(tmp_path):
    segs = _segments()
    man = {"num_frames": 40, "fps": 10.0}
    (tmp_path / "viz").mkdir()
    out_t = vs.timeline(segs, man, tmp_path / "viz" / "timeline.png")
    out_c, n = vs.contact_sheet(tmp_path, segs, tmp_path / "viz" / "cs.png")
    assert out_t.exists()
    assert out_c.exists()
    assert n == 2  # two interaction segments


def test_contact_sheet_fallback_when_no_interaction(tmp_path):
    # all-idle clip: contact sheet must still be populated (sampled frames), not blank
    segs = [{"label": "idle", "start_frame": 0, "end_frame": 40, "start_time": 0.0, "end_time": 4.0}]
    (tmp_path / "viz").mkdir()
    out_c, n = vs.contact_sheet(tmp_path, segs, tmp_path / "viz" / "cs.png")
    assert out_c.exists()
    assert n == 10  # uniformly sampled cells across the clip
