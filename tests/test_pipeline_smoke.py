from __future__ import annotations

import os

import pytest

from aoe_pipeline.config import PipelineConfig
from aoe_pipeline.pipeline import Pipeline
from aoe_pipeline.stages.registry import available


def test_all_stages_registered():
    for name in ["ingest", "trajectory", "hands", "segment", "qc", "augment"]:
        assert name in available()


def test_ingest_smoke(synthetic_clip, tmp_path):
    cfg = PipelineConfig.default()
    ctx = Pipeline(cfg).run(synthetic_clip, tmp_path / "out", only={"ingest"})

    frames = sorted((ctx.clip_dir / "frames").glob("frame_*.png"))
    assert len(frames) >= 5  # ~10 frames for a 1s @ 10fps clip
    assert (ctx.clip_dir / "manifest.json").exists()
    assert ctx.manifest.intrinsics is not None
    assert ctx.manifest.width == 320 and ctx.manifest.height == 240
    assert ctx.manifest.stages["ingest"].status == "ok"


@pytest.mark.skipif(
    os.environ.get("AOE_RUN_E2E") != "1",
    reason="end-to-end run needs model downloads; set AOE_RUN_E2E=1 to enable",
)
def test_full_pipeline_real_models(synthetic_clip, tmp_path):
    """Full ordered run completes; augment is disabled by default -> skipped."""
    cfg = PipelineConfig.default()
    ctx = Pipeline(cfg).run(synthetic_clip, tmp_path / "out")
    statuses = {n: s.status for n, s in ctx.manifest.stages.items()}
    assert statuses["ingest"] == "ok"
    assert statuses["augment"] == "skipped"
    for name in ["trajectory", "hands", "segment", "qc"]:
        assert statuses[name] == "ok"
