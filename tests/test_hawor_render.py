from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from aoe_pipeline.hawor_check import check_hawor_outputs
from aoe_pipeline.hawor_render import render_camera_panel, render_hawor_demo
from aoe_pipeline.schema import CameraIntrinsics, ClipManifest


def test_render_hawor_demo_uses_mesh_and_writes_square_mp4(tmp_path):
    clip_dir = _make_render_clip(tmp_path, include_mesh=True)
    out = clip_dir / "viz" / "hawor_demo.mp4"

    render_hawor_demo(clip_dir, out, fps=10.0, size=240, prefer_mesh=True)

    assert out.exists()
    width, height, frames = _video_info(out)
    assert (width, height) == (240, 240)
    assert frames >= 1

    report = check_hawor_outputs(clip_dir, out, require_mesh=True, expected_size=240)
    assert report["ok"] is True
    assert report["frames_with_mesh"] >= 1
    assert report["median_joint_reprojection_px"] <= 1.0
    assert report["visible_projected_mesh_vertices"] >= 1


def test_render_hawor_demo_falls_back_to_joint_hull_without_mesh(tmp_path):
    clip_dir = _make_render_clip(tmp_path, include_mesh=False)
    out = clip_dir / "viz" / "hawor_demo_joint_hull.mp4"

    render_hawor_demo(clip_dir, out, fps=10.0, size=240, prefer_mesh=True)

    assert out.exists()
    width, height, frames = _video_info(out)
    assert (width, height) == (240, 240)
    assert frames >= 1


def test_hawor_demo_cli_accepts_documented_bool_values(tmp_path):
    typer_testing = pytest.importorskip("typer.testing")
    from aoe_pipeline.cli import app

    runner = typer_testing.CliRunner()
    clip_dir = _make_render_clip(tmp_path, include_mesh=True)
    out = clip_dir / "viz" / "hawor_demo.mp4"

    render_result = runner.invoke(
        app,
        [
            "render-hawor-demo",
            "--clip-dir",
            str(clip_dir),
            "--out",
            str(out),
            "--fps",
            "10",
            "--size",
            "240",
            "--prefer-mesh",
            "true",
        ],
    )

    assert render_result.exit_code == 0, render_result.output
    assert out.exists()

    check_result = runner.invoke(
        app,
        [
            "check-hawor-demo",
            "--clip-dir",
            str(clip_dir),
            "--demo-mp4",
            str(out),
            "--require-mesh",
            "true",
            "--expected-size",
            "240",
        ],
    )

    assert check_result.exit_code == 0, check_result.output
    assert "HaWoR outputs validated" in check_result.output


def test_hawor_demo_cli_accepts_false_bool_values(tmp_path):
    typer_testing = pytest.importorskip("typer.testing")
    from aoe_pipeline.cli import app

    runner = typer_testing.CliRunner()
    clip_dir = _make_render_clip(tmp_path, include_mesh=False)
    out = clip_dir / "viz" / "hawor_demo_joint_hull.mp4"

    render_result = runner.invoke(
        app,
        [
            "render-hawor-demo",
            "--clip-dir",
            str(clip_dir),
            "--out",
            str(out),
            "--fps",
            "10",
            "--size",
            "240",
            "--prefer-mesh",
            "false",
        ],
    )

    assert render_result.exit_code == 0, render_result.output

    check_result = runner.invoke(
        app,
        [
            "check-hawor-demo",
            "--clip-dir",
            str(clip_dir),
            "--demo-mp4",
            str(out),
            "--require-mesh",
            "false",
            "--expected-size",
            "240",
        ],
    )

    assert check_result.exit_code == 0, check_result.output


def test_check_hawor_outputs_requires_aoe_world_metadata_for_mesh(tmp_path):
    clip_dir = _make_render_clip(tmp_path, include_mesh=True)
    out = clip_dir / "viz" / "hawor_demo.mp4"
    render_hawor_demo(clip_dir, out, fps=10.0, size=240, prefer_mesh=True)
    (clip_dir / "hands" / "hawor_meta.json").write_text(
        json.dumps({"world_coordinates": "HaWoR SLAM world"}, indent=2)
    )

    with pytest.raises(ValueError, match="AoE trajectory.tum world coordinates"):
        check_hawor_outputs(clip_dir, out, require_mesh=True, expected_size=240)


def test_check_hawor_outputs_requires_demo_mp4(tmp_path):
    clip_dir = _make_render_clip(tmp_path, include_mesh=True)

    with pytest.raises(ValueError, match="missing demo MP4"):
        check_hawor_outputs(clip_dir, require_mesh=True, expected_size=240)


def test_check_hawor_outputs_rejects_inconsistent_camera_projection(tmp_path):
    clip_dir = _make_render_clip(tmp_path, include_mesh=True)
    out = clip_dir / "viz" / "hawor_demo.mp4"
    render_hawor_demo(clip_dir, out, fps=10.0, size=240, prefer_mesh=True)
    joints_2d = np.load(clip_dir / "hands" / "joints_2d.npy")
    joints_2d[np.isfinite(joints_2d)] += 20.0
    np.save(clip_dir / "hands" / "joints_2d.npy", joints_2d)

    with pytest.raises(ValueError, match="reprojection error"):
        check_hawor_outputs(clip_dir, out, require_mesh=True, expected_size=240)


def test_camera_panel_renders_mesh_with_shaded_surface():
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:] = (30, 40, 50)
    verts = np.full((2, 5, 3), np.nan, np.float32)
    verts[1] = np.array(
        [
            [-0.20, -0.20, 1.0],
            [0.22, -0.18, 0.9],
            [0.25, 0.22, 1.1],
            [-0.18, 0.24, 1.2],
            [0.02, 0.00, 0.75],
        ],
        np.float32,
    )
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], np.int32)
    K = CameraIntrinsics.from_fov(64, 48).K

    panel = render_camera_panel(frame, verts, faces, K, size=120)

    base = np.zeros((120, 120, 3), np.uint8)
    base[15:105] = cv2.resize(frame, (120, 90))
    diff = np.abs(panel.astype(int) - base.astype(int)).sum(axis=2)
    assert np.count_nonzero(diff > 50) > 500
    assert len(np.unique(panel.reshape(-1, 3), axis=0)) > 10


def _make_render_clip(tmp_path, include_mesh: bool):
    clip_dir = tmp_path / "clip"
    frames_dir = clip_dir / "frames"
    hands_dir = clip_dir / "hands"
    frames_dir.mkdir(parents=True)
    hands_dir.mkdir()

    T = 3
    manifest = ClipManifest(
        clip_id="clip",
        video_path=str(tmp_path / "clip.mp4"),
        fps=10.0,
        num_frames=T,
        width=64,
        height=48,
        intrinsics=CameraIntrinsics.from_fov(64, 48),
    )
    (clip_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
    (clip_dir / "trajectory.tum").write_text(
        "\n".join(f"{i / 10:.6f} {i * 0.02:.4f} 0 {i * 0.03:.4f} 0 0 0 1" for i in range(T))
        + "\n"
    )

    for t in range(T):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, 0] = 40 + t * 25
        frame[:, :, 1] = np.linspace(0, 180, 64, dtype=np.uint8)
        cv2.rectangle(frame, (16 + t, 12), (45 + t, 38), (220, 220, 220), -1)
        cv2.imwrite(str(frames_dir / f"frame_{t:06d}.png"), frame)

    joints_2d = np.full((T, 2, 21, 2), np.nan, np.float32)
    joints_cam = np.full((T, 2, 21, 3), np.nan, np.float32)
    joints_world = np.full((T, 2, 21, 3), np.nan, np.float32)
    intr = manifest.intrinsics
    for t in range(T):
        x = np.linspace(20 + t, 42 + t, 21)
        y = 24 + 8 * np.sin(np.linspace(0, np.pi, 21))
        joints_2d[t, 1] = np.column_stack([x, y])
        joints_cam[t, 1] = np.column_stack(
            [
                (x - intr.cx) / intr.fx,
                (y - intr.cy) / intr.fy,
                np.ones(21),
            ]
        )
        joints_world[t, 1] = joints_cam[t, 1] + np.array([t * 0.02, 0.0, t * 0.03])
    np.save(hands_dir / "joints_2d.npy", joints_2d)
    np.save(hands_dir / "joints_cam.npy", joints_cam)
    np.save(hands_dir / "joints_world.npy", joints_world)
    (hands_dir / "meta.json").write_text(
        json.dumps(
            {
                "backend": "hybrid",
                "method": "MediaPipe+HaWoR+MANO",
                "frames_total": T,
                "frames_with_hand": T,
            },
            indent=2,
        )
    )

    if include_mesh:
        (hands_dir / "hawor_meta.json").write_text(
            json.dumps(
                {
                    "backend": "hawor",
                    "world_coordinates": "AoE trajectory.tum camera-to-world",
                },
                indent=2,
            )
        )
        verts_cam = np.full((T, 2, 4, 3), np.nan, np.float32)
        verts_world = np.full((T, 2, 4, 3), np.nan, np.float32)
        for t in range(T):
            verts_cam[t, 1] = np.array(
                [[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.2, 0.2, 1.0], [-0.2, 0.2, 1.0]],
                np.float32,
            )
            verts_world[t, 1] = verts_cam[t, 1] + np.array([t * 0.02, 0.0, t * 0.03])
        np.save(hands_dir / "verts_cam.npy", verts_cam)
        np.save(hands_dir / "verts_world.npy", verts_world)
        np.save(hands_dir / "faces.npy", np.array([[0, 1, 2], [0, 2, 3]], np.int32))
    return clip_dir


def _video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()
