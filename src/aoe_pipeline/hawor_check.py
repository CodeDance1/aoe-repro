"""Validation helpers for HaWoR hybrid outputs and demo videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def check_hawor_outputs(
    clip_dir: Path,
    demo_mp4: Path | None = None,
    require_mesh: bool = True,
    expected_size: int = 720,
) -> dict[str, Any]:
    """Validate the artifacts needed for the HaWoR-style MANO mesh demo."""
    clip_dir = Path(clip_dir)
    hands_dir = clip_dir / "hands"
    report: dict[str, Any] = {"clip_dir": str(clip_dir), "checks": []}

    manifest = _load_json(clip_dir / "manifest.json")
    meta = _load_json(hands_dir / "meta.json")
    _require(meta.get("backend") == "hybrid", "hands/meta.json must report backend=hybrid")
    _require(
        meta.get("method") == "MediaPipe+HaWoR+MANO",
        "hands/meta.json must report method=MediaPipe+HaWoR+MANO",
    )
    report["backend"] = meta.get("backend")
    report["method"] = meta.get("method")

    mp4_path = demo_mp4 or clip_dir / "viz" / "hawor_demo.mp4"
    _require(mp4_path.exists(), f"missing demo MP4 at {mp4_path}")

    if require_mesh:
        hawor_meta = _load_json(hands_dir / "hawor_meta.json")
        _require(
            hawor_meta.get("world_coordinates") == "AoE trajectory.tum camera-to-world",
            "hands/hawor_meta.json must report AoE trajectory.tum world coordinates",
        )
        report["world_coordinates"] = hawor_meta.get("world_coordinates")

    joints_world = _load_array(hands_dir / "joints_world.npy", "joints_world", dims=3)
    joints_cam = _load_array(hands_dir / "joints_cam.npy", "joints_cam", dims=3)
    joints_2d = _load_array(hands_dir / "joints_2d.npy", "joints_2d", dims=2)
    t = joints_world.shape[0]
    _require(joints_cam.shape[0] == t and joints_2d.shape[0] == t, "joint arrays must share T")
    _require(joints_world.shape[1:3] == (2, 21), "joints_world must have shape (T,2,21,3)")
    _require(joints_cam.shape[1:3] == (2, 21), "joints_cam must have shape (T,2,21,3)")
    _require(joints_2d.shape[1:3] == (2, 21), "joints_2d must have shape (T,2,21,2)")
    finite_frames = int(np.isfinite(joints_world[..., 0]).any(axis=(1, 2)).sum())
    _require(finite_frames > 0, "joints_world must contain at least one finite hand frame")
    report["frames_total"] = int(t)
    report["frames_with_world_hand"] = finite_frames
    reproj_error = _median_joint_reprojection_error(joints_cam, joints_2d, manifest)
    _require(reproj_error <= 1.0, f"joints_cam reprojection error too high: {reproj_error:.3f}px")
    report["median_joint_reprojection_px"] = float(reproj_error)

    if require_mesh:
        verts_cam = _load_mesh_array(hands_dir / "verts_cam.npy", "verts_cam", t)
        verts_world = _load_mesh_array(hands_dir / "verts_world.npy", "verts_world", t)
        faces = np.load(hands_dir / "faces.npy")
        _require(faces.ndim == 2 and faces.shape[1] == 3, "faces.npy must have shape (F,3)")
        _require(
            len(faces) == 0 or int(faces.max()) < verts_cam.shape[2],
            "faces.npy references vertices outside verts_cam.npy",
        )
        _require(
            verts_cam.shape[:3] == verts_world.shape[:3],
            "verts_cam.npy and verts_world.npy must share (T,2,V)",
        )
        mesh_frames = int(np.isfinite(verts_world[..., 0]).any(axis=(1, 2)).sum())
        _require(mesh_frames > 0, "verts_world must contain at least one finite MANO mesh frame")
        visible_vertices = _count_visible_projected_vertices(verts_cam, manifest)
        _require(visible_vertices > 0, "verts_cam must project at least one MANO vertex into frame")
        report["mesh_vertices"] = int(verts_world.shape[2])
        report["mesh_faces"] = int(faces.shape[0])
        report["frames_with_mesh"] = mesh_frames
        report["visible_projected_mesh_vertices"] = visible_vertices

    video_info = _video_info(mp4_path)
    _require(
        video_info["width"] == expected_size and video_info["height"] == expected_size,
        f"demo MP4 must be {expected_size}x{expected_size}",
    )
    _require(video_info["frames"] > 0, "demo MP4 must contain frames")
    report["demo_mp4"] = str(mp4_path)
    report["demo_video"] = video_info

    report["manifest_fps"] = manifest.get("fps")
    report["ok"] = True
    return report


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.exists(), f"missing {path}")
    return json.loads(path.read_text())


def _load_array(path: Path, name: str, dims: int) -> np.ndarray:
    _require(path.exists(), f"missing {path}")
    arr = np.load(path)
    _require(arr.ndim == 4 and arr.shape[-1] == dims, f"{name} must have shape (T,2,21,{dims})")
    return arr


def _load_mesh_array(path: Path, name: str, expected_t: int) -> np.ndarray:
    _require(path.exists(), f"missing {path}")
    arr = np.load(path)
    _require(
        arr.ndim == 4 and arr.shape[0] == expected_t and arr.shape[1] == 2 and arr.shape[-1] == 3,
        f"{name} must have shape (T,2,V,3)",
    )
    return arr


def _median_joint_reprojection_error(
    joints_cam: np.ndarray,
    joints_2d: np.ndarray,
    manifest: dict[str, Any],
) -> float:
    proj = _project_camera_points(joints_cam, manifest)
    valid = np.isfinite(proj).all(axis=-1) & np.isfinite(joints_2d).all(axis=-1)
    if not valid.any():
        return float("inf")
    return float(np.median(np.linalg.norm(proj[valid] - joints_2d[valid], axis=-1)))


def _count_visible_projected_vertices(verts_cam: np.ndarray, manifest: dict[str, Any]) -> int:
    proj = _project_camera_points(verts_cam, manifest)
    width = float(manifest["width"])
    height = float(manifest["height"])
    valid = (
        np.isfinite(proj).all(axis=-1)
        & (verts_cam[..., 2] > 1e-6)
        & (proj[..., 0] >= 0)
        & (proj[..., 0] < width)
        & (proj[..., 1] >= 0)
        & (proj[..., 1] < height)
    )
    return int(valid.sum())


def _project_camera_points(points: np.ndarray, manifest: dict[str, Any]) -> np.ndarray:
    intr = manifest["intrinsics"]
    z = points[..., 2]
    x = intr["fx"] * points[..., 0] / z + intr["cx"]
    y = intr["fy"] * points[..., 1] / z + intr["cy"]
    out = np.stack([x, y], axis=-1).astype(float)
    out[~np.isfinite(out).all(axis=-1) | (z <= 1e-6)] = np.nan
    return out


def _video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        _require(cap.isOpened(), f"failed to open {path}")
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
