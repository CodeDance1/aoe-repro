"""Tests for the semantic-labeling stage (`label`) and vlm.label_segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aoe_pipeline.config import PipelineConfig
from aoe_pipeline.schema import ClipManifest
from aoe_pipeline.stages.base import ClipContext
from aoe_pipeline.stages.registry import available
from aoe_pipeline.stages.s2b_label import LabelStage


def _ctx(tmp_path: Path, segments) -> ClipContext:
    return ClipContext(
        clip_id="c", clip_dir=tmp_path, video_path=Path("v.mp4"),
        config=PipelineConfig.default(), manifest=ClipManifest(clip_id="c", video_path="v.mp4"),
        blackboard={"segments": segments},
    )


def test_label_registered_and_in_default_pipeline():
    assert "label" in available()
    assert "label" in PipelineConfig.default().pipeline


def test_label_stage_merges_from_file(tmp_path):
    segs = [
        {"label": "idle", "start_frame": 0, "end_frame": 10},
        {"label": "interaction_0", "start_frame": 10, "end_frame": 20, "start_time": 1.0, "end_time": 2.0},
    ]
    (tmp_path / "semantic_labels.json").write_text(json.dumps(
        [{"id": "interaction_0", "label_zh": "拿起杯子", "label_en": "pick up cup", "hand": "left"}],
        ensure_ascii=False,
    ))
    ctx = _ctx(tmp_path, segs)
    LabelStage({}).run(ctx)

    out = json.loads((tmp_path / "segments.json").read_text())
    i0 = next(s for s in out if s["label"] == "interaction_0")
    assert i0["label_en"] == "pick up cup"
    assert i0["label_zh"] == "拿起杯子"
    # idle segment untouched
    assert "label_en" not in out[0]
    st = ctx.manifest.stages["label"]
    assert st.status == "ok"
    assert st.info["labeled"] == 1
    assert st.info["source"].startswith("file:")


def test_label_stage_skips_without_labels_or_provider(tmp_path, monkeypatch):
    import aoe_pipeline.stages.s2b_label as ls

    monkeypatch.setattr(ls, "_provider_available", lambda _p: False)
    ctx = _ctx(tmp_path, [{"label": "interaction_0", "start_frame": 0, "end_frame": 10}])
    ls.LabelStage({}).run(ctx)
    assert ctx.manifest.stages["label"].status == "skipped"


def test_label_stage_skips_when_no_interaction(tmp_path):
    ctx = _ctx(tmp_path, [{"label": "idle", "start_frame": 0, "end_frame": 10}])
    LabelStage({}).run(ctx)
    assert ctx.manifest.stages["label"].status == "skipped"


def test_pipeline_runs_label_stage_and_merges(tmp_path):
    """Integration: the label stage runs through the orchestrator and merges a
    semantic_labels.json (no models / no API)."""
    from aoe_pipeline.pipeline import Pipeline

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")  # only needs to exist as a path for the run
    out = tmp_path / "out"
    clip_dir = out / "clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "segments.json").write_text(json.dumps([
        {"label": "idle", "start_frame": 0, "end_frame": 10},
        {"label": "interaction_0", "start_frame": 10, "end_frame": 20},
    ]))
    (clip_dir / "semantic_labels.json").write_text(json.dumps(
        [{"id": "interaction_0", "label_en": "pick up cup", "label_zh": "拿起杯子"}], ensure_ascii=False))

    ctx = Pipeline(PipelineConfig.default()).run(video, out, only={"label"})

    merged = json.loads((clip_dir / "segments.json").read_text())
    i0 = next(s for s in merged if s["label"] == "interaction_0")
    assert i0["label_en"] == "pick up cup"
    assert ctx.manifest.stages["label"].status == "ok"


def test_label_segments_attaches_labels_with_mock_provider(monkeypatch):
    import aoe_pipeline.vlm as vlm

    monkeypatch.setattr(
        vlm, "_query_label",
        lambda kf, provider, model: '{"label_zh":"拿杯","label_en":"grab cup",'
        '"hand":"left","object":"cup","confidence":0.9}',
    )
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(20)]
    segs = [{"label": "idle", "start_frame": 0, "end_frame": 5},
            {"label": "interaction_0", "start_frame": 5, "end_frame": 15}]
    out = vlm.label_segments(frames, segs, fps=10.0, provider="openai", verify=0)
    i0 = next(s for s in out if s["label"] == "interaction_0")
    assert i0["label_en"] == "grab cup"
    assert i0["hand"] == "left"
    assert "label_en" not in out[0]  # idle untouched


def _kf():
    return [np.zeros((4, 4, 3), np.uint8)]


def test_label_one_keeps_initial_when_verifiers_agree(monkeypatch):
    import aoe_pipeline.vlm as vlm

    monkeypatch.setattr(vlm, "_query_label", lambda kf, p, m:
                        '{"label_zh":"拿杯","label_en":"grab cup","hand":"left","object":"cup","confidence":0.9}')
    monkeypatch.setattr(vlm, "_query_verify", lambda kf, prop, p, m:
                        '{"agree": true, "label_zh":"拿杯","label_en":"grab cup","hand":"left","reason":"ok"}')
    lab = vlm.label_one(_kf(), provider="openai", verify=2)
    assert lab["label_en"] == "grab cup"
    assert lab["confidence"] == 0.9


def test_label_one_adopts_correction_when_majority_disagree(monkeypatch):
    import aoe_pipeline.vlm as vlm

    monkeypatch.setattr(vlm, "_query_label", lambda kf, p, m:
                        '{"label_zh":"伸手","label_en":"reach for cup","hand":"left","object":"cup","confidence":0.8}')
    monkeypatch.setattr(vlm, "_query_verify", lambda kf, prop, p, m:
                        '{"agree": false, "label_zh":"收回手","label_en":"withdraw hand","hand":"left","reason":"retracts"}')
    lab = vlm.label_one(_kf(), provider="openai", verify=2)
    assert lab["label_en"] == "withdraw hand"
    assert lab["label_zh"] == "收回手"
    assert lab["object"] == "cup"        # object kept from the initial label
    assert lab["confidence"] <= 0.6      # confidence lowered after correction


def test_label_one_verify_zero_skips_verification(monkeypatch):
    import aoe_pipeline.vlm as vlm

    monkeypatch.setattr(vlm, "_query_label", lambda kf, p, m:
                        '{"label_zh":"拿杯","label_en":"grab cup","confidence":0.9}')

    def _boom(*a, **k):
        raise AssertionError("verify must not be called when verify=0")

    monkeypatch.setattr(vlm, "_query_verify", _boom)
    assert vlm.label_one(_kf(), verify=0)["label_en"] == "grab cup"


def test_label_script_dry_run(tmp_path):
    import subprocess
    import sys

    import cv2

    (tmp_path / "frames").mkdir()
    for i in (0, 5, 10, 15, 20):
        cv2.imwrite(str(tmp_path / "frames" / f"frame_{i:06d}.png"), np.zeros((8, 8, 3), np.uint8))
    segs = [{"label": "idle", "start_frame": 0, "end_frame": 5},
            {"label": "interaction_0", "start_frame": 5, "end_frame": 20}]
    (tmp_path / "segments.json").write_text(json.dumps(segs))

    script = Path(__file__).resolve().parents[1] / "scripts" / "label_segments_vlm.py"
    r = subprocess.run([sys.executable, str(script), "--clip-dir", str(tmp_path), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    labels = json.loads((tmp_path / "semantic_labels.json").read_text())
    assert len(labels) == 1 and labels[0]["id"] == "interaction_0"
    assert labels[0]["label_en"] == "dry-run placeholder"
