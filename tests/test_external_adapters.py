from __future__ import annotations

import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aoe_pipeline.external import ExternalPipelineError, render_command, run_external
from aoe_pipeline.config import PipelineConfig
from aoe_pipeline.schema import CameraIntrinsics, ClipManifest
from aoe_pipeline.stages.base import ClipContext
from aoe_pipeline.stages.s3_trajectory import _read_tum_poses
from aoe_pipeline.stages.s4_hands import HandStage


def test_render_command_rejects_unknown_placeholder():
    try:
        render_command("python {missing}", {"known": "x"})
    except ExternalPipelineError as exc:
        assert "unknown placeholder" in str(exc)
    else:
        raise AssertionError("expected ExternalPipelineError")


def test_run_external_formats_list_command(tmp_path):
    marker = tmp_path / "marker.txt"
    run_external(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path(r'{marker}').write_text('ok')",
        ],
        {"marker": marker},
    )

    assert marker.read_text() == "ok"


def test_read_tum_poses(tmp_path):
    path = tmp_path / "trajectory.tum"
    path.write_text(
        "0.000000 1 2 3 0 0 0 1\n"
        "0.033333 4 5 6 0 0 0 1\n"
    )

    poses = _read_tum_poses(path)

    assert len(poses) == 2
    assert np.allclose(poses[0][:3, 3], [1, 2, 3])
    assert np.allclose(poses[1][:3, :3], np.eye(3))


def test_hybrid_hands_imports_hawor_outputs_and_meshes(tmp_path, monkeypatch):
    ctx = _make_hand_ctx(tmp_path, frames=3)
    monkeypatch.setattr(HandStage, "_run_mediapipe_detection", lambda self, ctx: _fake_detection(3))
    stage = HandStage(
        {
            "backend": "hybrid",
            "viz": False,
            "hybrid": {
                "fallback_to_mediapipe": True,
                "original": {"command": _fake_hawor_command(include_mesh=True)},
            },
        }
    )

    stage.run(ctx)

    joints_world = np.load(ctx.hands_dir / "joints_world.npy")
    joints_2d = np.load(ctx.hands_dir / "joints_2d.npy")
    meta = json.loads((ctx.hands_dir / "meta.json").read_text())

    assert joints_world.shape == (3, 2, 21, 3)
    assert joints_2d.shape == (3, 2, 21, 2)
    assert np.isfinite(joints_world[0, 1]).all()
    assert (ctx.hands_dir / "verts_cam.npy").exists()
    assert (ctx.hands_dir / "verts_world.npy").exists()
    assert (ctx.hands_dir / "faces.npy").exists()
    assert (ctx.hands_dir / "mediapipe_hints.npz").exists()
    assert (ctx.hands_dir / "hawor_meta.json").exists()
    hints = np.load(ctx.hands_dir / "mediapipe_hints.npz")
    assert hints["slot_labels"].tolist() == ["Left", "Right"]
    assert hints["handedness"][1, 0] == "Left"
    assert hints["presence"].shape == (3, 2)
    assert meta["backend"] == "hybrid"
    assert meta["hawor_frames"] == 1
    assert meta["mediapipe_fallback_frames"] == 1
    assert meta["hawor_meta"] == "hawor_meta.json"
    assert ctx.manifest.stages["hands"].info["backend"] == "hybrid"


def test_hybrid_hands_fills_missing_hawor_slots_from_mediapipe(tmp_path, monkeypatch):
    ctx = _make_hand_ctx(tmp_path, frames=3)
    detection = _fake_detection(3)
    monkeypatch.setattr(HandStage, "_run_mediapipe_detection", lambda self, ctx: detection)
    stage = HandStage(
        {
            "backend": "hybrid",
            "viz": False,
            "hybrid": {
                "fallback_to_mediapipe": True,
                "original": {"command": _fake_hawor_command(include_mesh=False)},
            },
        }
    )

    stage.run(ctx)

    joints_world = np.load(ctx.hands_dir / "joints_world.npy")
    joints_cam = np.load(ctx.hands_dir / "joints_cam.npy")
    joints_2d = np.load(ctx.hands_dir / "joints_2d.npy")

    assert np.allclose(joints_world[1, 0], detection["joints_world"][1, 0])
    assert np.allclose(joints_cam[1, 0], detection["joints_cam"][1, 0])
    assert np.allclose(joints_2d[1, 0], detection["joints_2d"][1, 0])


def test_hybrid_hands_rejects_bad_hawor_shapes(tmp_path, monkeypatch):
    ctx = _make_hand_ctx(tmp_path, frames=3)
    monkeypatch.setattr(HandStage, "_run_mediapipe_detection", lambda self, ctx: _fake_detection(3))
    stage = HandStage(
        {
            "backend": "hybrid",
            "viz": False,
            "hybrid": {
                "original": {"command": _fake_hawor_command(bad_shape=True)},
            },
        }
    )

    with pytest.raises(ExternalPipelineError, match="joints_world"):
        stage.run(ctx)


def test_hybrid_hands_rejects_partial_mesh_outputs(tmp_path):
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    np.save(external_dir / "verts_cam.npy", np.zeros((3, 2, 4, 3), np.float32))
    stage = HandStage({"backend": "hybrid"})

    with pytest.raises(ExternalPipelineError, match="partial MANO mesh outputs"):
        stage._import_optional_meshes(external_dir, {}, expected_t=3)


def test_hybrid_hands_rejects_faces_outside_mesh_vertices(tmp_path):
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    np.save(external_dir / "verts_cam.npy", np.zeros((3, 2, 4, 3), np.float32))
    np.save(external_dir / "verts_world.npy", np.zeros((3, 2, 4, 3), np.float32))
    np.save(external_dir / "faces.npy", np.array([[0, 1, 99]], np.int32))
    stage = HandStage({"backend": "hybrid"})

    with pytest.raises(ExternalPipelineError, match="outside verts"):
        stage._import_optional_meshes(external_dir, {}, expected_t=3)


def test_hawor_wrapper_pad_time_keeps_full_sequence_and_validates_tail_dim():
    mod = _load_hawor_wrapper_module()
    arr = np.ones((2, 2, 21, 3), np.float32)

    padded = mod._pad_time(arr, 3, dims=3)

    assert padded.shape == (3, 2, 21, 3)
    assert np.allclose(padded[:2], arr)
    assert np.isnan(padded[2]).all()
    with pytest.raises(ValueError, match="expected last dimension"):
        mod._pad_time(np.ones((2, 2, 21, 2), np.float32), 3, dims=3)


def test_hawor_wrapper_converts_valid_mask_to_time_slot_order():
    mod = _load_hawor_wrapper_module()
    pred_valid = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=np.float32,
    )

    valid = mod._valid_time_slots(pred_valid, 1, 4)

    assert valid.shape == (3, 2)
    assert valid.tolist() == [[False, True], [True, True], [False, False]]


def test_hawor_wrapper_stages_aoe_frames_and_clears_hawor_cache(tmp_path):
    cv2 = pytest.importorskip("cv2")
    mod = _load_hawor_wrapper_module()
    frames_dir = tmp_path / "clip" / "frames"
    frames_dir.mkdir(parents=True)
    for idx in range(2):
        frame = np.full((12, 16, 3), idx * 80, np.uint8)
        assert cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.png"), frame)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    seq = tmp_path / "video"
    stale_track = seq / "tracks_0_99"
    stale_track.mkdir(parents=True)
    (stale_track / "model_boxes.npy").write_bytes(b"stale")
    (seq / "world_space_res.pth").write_bytes(b"stale")

    staged = mod._stage_aoe_frames_for_hawor(video, frames_dir, expected_t=2)

    assert [p.name for p in staged] == ["0000.jpg", "0001.jpg"]
    assert all(p.exists() for p in staged)
    assert not stale_track.exists()
    assert not (seq / "world_space_res.pth").exists()
    assert len(list((seq / "extracted_images").glob("*.jpg"))) == 2


def test_hawor_wrapper_uses_output_local_work_video_path(tmp_path):
    mod = _load_hawor_wrapper_module()
    src_video = tmp_path / "readonly_data" / "clip.mp4"
    out_dir = tmp_path / "external" / "hands_hybrid"

    work_video = mod._hawor_work_video_path(out_dir, src_video)

    assert work_video == out_dir / "_hawor_work" / "clip.mp4"
    assert work_video.exists()
    assert work_video.parent.exists()
    assert not src_video.parent.exists()


def test_hawor_wrapper_links_existing_source_video_into_work_dir(tmp_path):
    mod = _load_hawor_wrapper_module()
    src_video = tmp_path / "data" / "clip.mp4"
    out_dir = tmp_path / "external" / "hands_hybrid"
    src_video.parent.mkdir()
    src_video.write_bytes(b"video")

    work_video = mod._hawor_work_video_path(out_dir, src_video)

    assert work_video == out_dir / "_hawor_work" / "clip.mp4"
    assert work_video.exists()
    assert work_video.read_bytes() == b"video"
    assert src_video.exists()


def test_hawor_wrapper_resolves_relative_paths_under_hawor_root(tmp_path):
    mod = _load_hawor_wrapper_module()
    hawor_root = tmp_path / "HaWoR"
    args = type(
        "Args",
        (),
        {
            "video": tmp_path / "data" / "clip.mp4",
            "frames": tmp_path / "clip" / "frames",
            "trajectory": tmp_path / "clip" / "trajectory.tum",
            "intrinsics": tmp_path / "clip" / "intrinsics.json",
            "mediapipe_hints": tmp_path / "clip" / "mediapipe_hints.npz",
            "out": tmp_path / "out",
            "hawor_root": None,
            "checkpoint": Path("./weights/hawor/checkpoints/hawor.ckpt"),
            "infiller_weight": Path("./weights/hawor/checkpoints/infiller.pt"),
        },
    )()

    resolved = mod._resolve_args(args, hawor_root)

    assert resolved.video.is_absolute()
    assert resolved.out.is_absolute()
    assert resolved.hawor_root == hawor_root
    assert resolved.checkpoint == hawor_root / "weights/hawor/checkpoints/hawor.ckpt"
    assert resolved.infiller_weight == hawor_root / "weights/hawor/checkpoints/infiller.pt"


def test_hawor_wrapper_loads_tum_and_maps_camera_points_to_aoe_world(tmp_path):
    mod = _load_hawor_wrapper_module()
    tum = tmp_path / "trajectory.tum"
    tum.write_text(
        "0.000000 1 2 3 0 0 0 1\n"
        "0.033333 4 5 6 0 0 0 1\n"
    )
    cam = np.zeros((2, 2, 1, 3), np.float32)
    cam[0, :, 0] = [0.5, 0.0, 1.0]
    cam[1, :, 0] = [1.0, 0.5, 2.0]

    poses = mod._load_tum_camera_to_world(tum, 0, 2)
    world = mod._camera_to_world(cam, poses)

    assert poses.shape == (2, 4, 4)
    assert np.allclose(world[0, 0, 0], [1.5, 2.0, 4.0])
    assert np.allclose(world[1, 1, 0], [5.0, 5.5, 8.0])


def test_hawor_wrapper_reports_time_length_mismatch():
    mod = _load_hawor_wrapper_module()
    points = np.zeros((2, 2, 1, 3), np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 1, axis=0)

    with pytest.raises(ValueError, match="camera_to_world expected time length 2"):
        mod._camera_to_world(points, poses)


def _make_hand_ctx(tmp_path, frames: int) -> ClipContext:
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    manifest = ClipManifest(
        clip_id="clip",
        video_path=str(tmp_path / "clip.mp4"),
        fps=30.0,
        num_frames=frames,
        width=64,
        height=48,
        intrinsics=CameraIntrinsics.from_fov(64, 48),
    )
    trajectory = "\n".join(f"{i / 30:.6f} 0 0 {i * 0.01:.3f} 0 0 0 1" for i in range(frames))
    (clip_dir / "trajectory.tum").write_text(trajectory + "\n")
    return ClipContext(
        clip_id="clip",
        clip_dir=clip_dir,
        video_path=tmp_path / "clip.mp4",
        config=PipelineConfig.default(),
        manifest=manifest,
        blackboard={"frames": [np.zeros((48, 64, 3), np.uint8) for _ in range(frames)]},
    )


def _load_hawor_wrapper_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_hawor_with_mediapipe_hints.py"
    spec = importlib.util.spec_from_file_location("run_hawor_with_mediapipe_hints", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_detection(frames: int) -> dict:
    joints_2d = np.full((frames, 2, 21, 2), np.nan, np.float32)
    joints_cam = np.full((frames, 2, 21, 3), np.nan, np.float32)
    joints_world = np.full((frames, 2, 21, 3), np.nan, np.float32)
    boxes = np.full((frames, 2, 4), np.nan, np.float32)
    presence = np.zeros((frames, 2), dtype=bool)

    pts2 = np.stack([np.linspace(10, 30, 21), np.linspace(12, 28, 21)], axis=1).astype(np.float32)
    pts3 = np.column_stack([pts2 / 100.0, np.ones(21, dtype=np.float32)])
    joints_2d[1, 0] = pts2
    joints_cam[1, 0] = pts3
    joints_world[1, 0] = pts3 + np.array([0.1, 0.2, 0.3], np.float32)
    boxes[1, 0] = [10, 12, 30, 28]
    presence[1, 0] = True
    return {
        "joints_2d": joints_2d,
        "joints_cam": joints_cam,
        "joints_world": joints_world,
        "handedness_per_frame": [[None, None], ["Left", None], [None, None]],
        "boxes": boxes,
        "presence": presence,
    }


def _fake_hawor_command(include_mesh: bool = False, bad_shape: bool = False) -> list[str]:
    code = """
import pathlib
import sys
import numpy as np
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
T = 3
jw = np.full((T, 2, 21, 3), np.nan, np.float32)
j2 = np.full((T, 2, 21, 2), np.nan, np.float32)
jc = np.full((T, 2, 21, 3), np.nan, np.float32)
if sys.argv[3] == "bad":
    jw = np.zeros((T - 1, 2, 21, 3), np.float32)
else:
    jw[0, 1] = 2.0
    j2[0, 1] = 20.0
    jc[0, 1] = 1.0
np.save(out / "joints_world.npy", jw)
np.save(out / "joints_2d.npy", j2)
np.save(out / "joints_cam.npy", jc)
if sys.argv[2] == "mesh":
    verts = np.full((T, 2, 4, 3), np.nan, np.float32)
    verts[0, 1] = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    np.save(out / "verts_cam.npy", verts)
    np.save(out / "verts_world.npy", verts)
    np.save(out / "faces.npy", np.array([[0, 1, 2], [0, 2, 3]], np.int32))
(out / "meta.json").write_text('{{"source": "fake_hawor"}}')
"""
    return [
        sys.executable,
        "-c",
        code,
        "{external_dir}",
        "mesh" if include_mesh else "nomesh",
        "bad" if bad_shape else "ok",
    ]
