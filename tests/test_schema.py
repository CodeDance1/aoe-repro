from __future__ import annotations

import numpy as np

from aoe_pipeline.schema import CameraIntrinsics, ClipManifest


def test_intrinsics_from_fov_and_K():
    intr = CameraIntrinsics.from_fov(640, 480, hfov_deg=90.0)
    assert intr.cx == 320 and intr.cy == 240
    # 90deg hfov on width 640 -> fx = 320 / tan(45) = 320
    assert np.isclose(intr.fx, 320.0, atol=1e-6)
    K = intr.K
    assert K.shape == (3, 3)
    assert K[0, 0] == intr.fx and K[2, 2] == 1.0


def test_manifest_roundtrip(tmp_path):
    m = ClipManifest(clip_id="c1", video_path="/v/clip.mp4", fps=30.0, num_frames=10,
                     width=640, height=480, intrinsics=CameraIntrinsics.from_fov(640, 480))
    m.set_stage("ingest", "ok", num_frames=10)
    m.set_stage("hands", "skipped")
    path = tmp_path / "manifest.json"
    m.save(path)

    loaded = ClipManifest.load(path)
    assert loaded.clip_id == "c1"
    assert loaded.num_frames == 10
    assert loaded.intrinsics is not None
    assert np.isclose(loaded.intrinsics.fx, m.intrinsics.fx)
    assert loaded.stages["ingest"].status == "ok"
    assert loaded.stages["ingest"].info["num_frames"] == 10
    assert loaded.stages["hands"].status == "skipped"
