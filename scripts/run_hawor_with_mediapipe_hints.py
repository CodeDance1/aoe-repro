#!/usr/bin/env python3
"""AoE wrapper for ThunderVVV/HaWoR.

Run this script inside the HaWoR Python environment. It follows HaWoR's public
``demo.py`` pipeline, then exports AoE-standard arrays:

    joints_world.npy  (T,2,21,3)
    joints_cam.npy    (T,2,21,3)
    joints_2d.npy     (T,2,21,2)
    verts_world.npy   (T,2,V,3)
    verts_cam.npy     (T,2,V,3)
    faces.npy         (F,3)

The MediaPipe hints file is used as the AoE frame-count and fallback reference.
HaWoR still performs its own detection/tracking internally because the upstream
demo pipeline does not currently expose a stable hints injection API.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def main() -> int:
    args = parse_args()
    hawor_root = (args.hawor_root or Path(os.environ.get("HAWOR_ROOT", "."))).resolve()
    args = _resolve_args(args, hawor_root)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    hints = np.load(args.mediapipe_hints)
    expected_t = int(hints["presence"].shape[0])
    intrinsics = json.loads(args.intrinsics.read_text())

    sys.path.insert(0, str(hawor_root))
    cwd = Path.cwd()
    os.chdir(hawor_root)
    try:
        exported = run_hawor(args, intrinsics, expected_t)
    finally:
        os.chdir(cwd)

    for name, arr in exported.items():
        np.save(out_dir / f"{name}.npy", arr)

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "backend": "hawor",
                "hawor_root": str(hawor_root),
                "hawor_work_dir": str(args.out / "_hawor_work"),
                "frames_source": str(args.frames),
                "checkpoint": str(args.checkpoint),
                "infiller_weight": str(args.infiller_weight),
                "frames_total": expected_t,
                "frames_with_hawor": int(
                    np.isfinite(exported["joints_world"][..., 0]).any(axis=(1, 2)).sum()
                ),
                "world_coordinates": "AoE trajectory.tum camera-to-world",
                "note": (
                    "MediaPipe hints were used for AoE alignment; "
                    "HaWoR detection ran internally."
                ),
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--mediapipe-hints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./weights/hawor/checkpoints/hawor.ckpt"),
    )
    parser.add_argument(
        "--infiller-weight",
        type=Path,
        default=Path("./weights/hawor/checkpoints/infiller.pt"),
    )
    return parser.parse_args()


def _resolve_args(args: argparse.Namespace, hawor_root: Path) -> argparse.Namespace:
    args.video = args.video.resolve()
    args.frames = args.frames.resolve()
    args.trajectory = args.trajectory.resolve()
    args.intrinsics = args.intrinsics.resolve()
    args.mediapipe_hints = args.mediapipe_hints.resolve()
    args.out = args.out.resolve()
    args.hawor_root = hawor_root
    args.checkpoint = _resolve_under(args.checkpoint, hawor_root)
    args.infiller_weight = _resolve_under(args.infiller_weight, hawor_root)
    return args


def _resolve_under(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_hawor(args: argparse.Namespace, intrinsics: dict, expected_t: int) -> dict[str, np.ndarray]:
    import torch

    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation

    hawor_video_path = _hawor_work_video_path(args.out, args.video)
    hawor_args = SimpleNamespace(
        img_focal=float(intrinsics["fx"]),
        video_path=str(hawor_video_path),
        input_type="file",
        checkpoint=str(args.checkpoint),
        infiller_weight=str(args.infiller_weight),
        vis_mode="world",
    )

    staged_frames = _stage_aoe_frames_for_hawor(hawor_video_path, args.frames, expected_t)
    start_idx, end_idx, seq_folder, _imgfiles = detect_track_video(hawor_args)
    if len(staged_frames) != end_idx - start_idx:
        raise RuntimeError(
            "HaWoR frame staging mismatch: "
            f"staged {len(staged_frames)} frames but detect_track_video saw {end_idx - start_idx}"
        )
    frame_chunks_all, _img_focal = hawor_motion_estimation(
        hawor_args,
        start_idx,
        end_idx,
        seq_folder,
    )
    slam_path = Path(seq_folder) / f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if not slam_path.exists():
        hawor_slam(hawor_args, start_idx, end_idx)
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(str(slam_path))

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        hawor_args,
        start_idx,
        end_idx,
        frame_chunks_all,
    )

    faces = _hawor_faces(get_mano_faces())
    vis_start = 0
    vis_end = pred_trans.shape[1]
    world, joints_world = _mano_world_outputs(
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        run_mano,
        run_mano_left,
        vis_start,
        vis_end,
    )

    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    R_c2w = torch.einsum("ij,njk->nik", R_x, R_c2w).cpu().numpy()
    t_c2w = torch.einsum("ij,nj->ni", R_x, t_c2w).cpu().numpy()
    R_w2c = np.swapaxes(R_c2w, -1, -2)
    t_w2c = -np.einsum("bij,bj->bi", R_w2c, t_c2w)

    world = np.einsum("ij,tsvj->tsvi", R_x.numpy(), world)
    joints_world = np.einsum("ij,tskj->tski", R_x.numpy(), joints_world)
    cam = _world_to_camera(world, R_w2c[vis_start:vis_end], t_w2c[vis_start:vis_end])
    joints_cam = _world_to_camera(joints_world, R_w2c[vis_start:vis_end], t_w2c[vis_start:vis_end])
    aoe_c2w = _load_tum_camera_to_world(args.trajectory, vis_start, vis_end)
    world = _camera_to_world(cam, aoe_c2w)
    joints_world = _camera_to_world(joints_cam, aoe_c2w)
    joints_2d = _project_joints(joints_cam, intrinsics)

    valid = _valid_time_slots(pred_valid, vis_start, vis_end)
    world[~valid] = np.nan
    cam[~valid] = np.nan
    joints_world[~valid] = np.nan
    joints_cam[~valid] = np.nan
    joints_2d[~valid] = np.nan

    return {
        "joints_world": _pad_time(joints_world, expected_t, dims=3),
        "joints_cam": _pad_time(joints_cam, expected_t, dims=3),
        "joints_2d": _pad_time(joints_2d, expected_t, dims=2),
        "verts_world": _pad_time(world, expected_t, dims=3),
        "verts_cam": _pad_time(cam, expected_t, dims=3),
        "faces": faces.astype(np.int32),
}


def _hawor_faces(base_faces) -> np.ndarray:
    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ],
        dtype=np.int32,
    )
    return np.concatenate([_to_numpy(base_faces).astype(np.int32), faces_new], axis=0)


def _mano_world_outputs(
    pred_trans,
    pred_rot,
    pred_hand_pose,
    pred_betas,
    run_mano,
    run_mano_left,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    # AoE convention: slot 0 = left, slot 1 = right.
    left = run_mano_left(
        pred_trans[0:1, start:end],
        pred_rot[0:1, start:end],
        pred_hand_pose[0:1, start:end],
        betas=pred_betas[0:1, start:end],
    )
    right = run_mano(
        pred_trans[1:2, start:end],
        pred_rot[1:2, start:end],
        pred_hand_pose[1:2, start:end],
        betas=pred_betas[1:2, start:end],
    )
    left_v = _to_numpy(left["vertices"])[0]
    right_v = _to_numpy(right["vertices"])[0]
    left_j = _mano_joints_or_vertices(left, left_v)
    right_j = _mano_joints_or_vertices(right, right_v)
    verts = np.stack([left_v, right_v], axis=1).astype(np.float32)
    joints = np.stack([left_j, right_j], axis=1).astype(np.float32)
    return verts, joints


def _mano_joints_or_vertices(mano_out: dict, verts: np.ndarray) -> np.ndarray:
    for key in ("joints", "joints3d", "joints_3d"):
        if key in mano_out:
            joints = _to_numpy(mano_out[key])[0]
            if joints.shape[1] >= 21:
                return joints[:, :21].astype(np.float32)
    # Last-resort fallback: keep the contract usable if the upstream helper only
    # exposes vertices. Prefer wrapper-side MANO joints whenever available.
    return verts[:, :21].astype(np.float32)


def _world_to_camera(points: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    _require_time_match("world_to_camera", points, R_w2c, t_w2c)
    return np.einsum("tij,tsnj->tsni", R_w2c, points) + t_w2c[:, None, None, :]


def _camera_to_world(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    _require_time_match("camera_to_world", points, c2w)
    return np.einsum("tij,tsnj->tsni", c2w[:, :3, :3], points) + c2w[:, None, None, :3, 3]


def _require_time_match(name: str, points: np.ndarray, *time_arrays: np.ndarray) -> None:
    expected = int(points.shape[0])
    for arr in time_arrays:
        if int(arr.shape[0]) != expected:
            raise ValueError(
                f"{name} expected time length {expected}, got {arr.shape[0]} for {arr.shape}"
            )


def _load_tum_camera_to_world(path: Path, start: int, end: int) -> np.ndarray:
    poses = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [float(v) for v in line.split()]
        if len(parts) != 8:
            raise ValueError(f"expected TUM line with 8 fields in {path}: {line}")
        _, tx, ty, tz, qx, qy, qz, qw = parts
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = _quat_xyzw_to_matrix(qx, qy, qz, qw)
        pose[:3, 3] = [tx, ty, tz]
        poses.append(pose)
    if len(poses) < end:
        raise ValueError(f"expected at least {end} poses in {path}, found {len(poses)}")
    return np.stack(poses[start:end], axis=0)


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.dot(q, q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    q *= np.sqrt(2.0 / n)
    x, y, z, w = q
    xx, xy, xz, xw = x * x, x * y, x * z, x * w
    yy, yz, yw = y * y, y * z, y * w
    zz, zw = z * z, z * w
    return np.array(
        [
            [1.0 - yy - zz, xy - zw, xz + yw],
            [xy + zw, 1.0 - xx - zz, yz - xw],
            [xz - yw, yz + xw, 1.0 - xx - yy],
        ],
        dtype=np.float32,
    )


def _project_joints(joints_cam: np.ndarray, intrinsics: dict) -> np.ndarray:
    z = joints_cam[..., 2]
    x = intrinsics["fx"] * joints_cam[..., 0] / z + intrinsics["cx"]
    y = intrinsics["fy"] * joints_cam[..., 1] / z + intrinsics["cy"]
    out = np.stack([x, y], axis=-1).astype(np.float32)
    out[~np.isfinite(out).all(axis=-1) | (z <= 1e-6)] = np.nan
    return out


def _stage_aoe_frames_for_hawor(video_path: Path, frames_dir: Path, expected_t: int) -> list[Path]:
    """Populate HaWoR's extracted_images folder from AoE frames.

    Upstream HaWoR re-extracts frames from ``video_path`` at 30 FPS. AoE may have
    applied stride/max_frames, so the processed ``frames_dir`` must be the source
    of truth for time alignment.
    """
    src_frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if len(src_frames) != expected_t:
        raise RuntimeError(
            f"expected {expected_t} AoE frames in {frames_dir}, found {len(src_frames)}"
        )

    video_path = Path(video_path)
    seq_folder = video_path.parent / video_path.stem
    img_folder = seq_folder / "extracted_images"
    seq_folder.mkdir(parents=True, exist_ok=True)
    _clear_hawor_frame_cache(seq_folder)
    img_folder.mkdir(parents=True, exist_ok=True)

    staged = []
    for idx, src in enumerate(src_frames):
        frame = cv2.imread(str(src))
        if frame is None:
            raise RuntimeError(f"failed to read AoE frame {src}")
        dst = img_folder / f"{idx:04d}.jpg"
        if not cv2.imwrite(str(dst), frame):
            raise RuntimeError(f"failed to write staged HaWoR frame {dst}")
        staged.append(dst)
    return staged


def _hawor_work_video_path(out_dir: Path, source_video: Path) -> Path:
    """Return a HaWoR-local video path so upstream writes stay inside --out."""
    path = Path(out_dir) / "_hawor_work" / Path(source_video).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    source_video = Path(source_video)
    if not source_video.exists():
        path.touch()
        return path
    try:
        path.symlink_to(source_video.resolve())
    except OSError:
        try:
            os.link(source_video, path)
        except OSError:
            shutil.copy2(source_video, path)
    return path


def _clear_hawor_frame_cache(seq_folder: Path) -> None:
    for path in seq_folder.glob("tracks_*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for name in ("extracted_images", "cam_space", "SLAM"):
        path = seq_folder / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("world_space_res.pth", "est_focal.txt"):
        path = seq_folder / name
        if path.exists():
            path.unlink()


def _pad_time(arr: np.ndarray, expected_t: int, dims: int) -> np.ndarray:
    if arr.shape[-1] != dims:
        raise ValueError(f"expected last dimension {dims}, got {arr.shape}")
    if arr.shape[0] == expected_t:
        return arr.astype(np.float32)
    out_shape = (expected_t, *arr.shape[1:])
    out = np.full(out_shape, np.nan, np.float32)
    n = min(expected_t, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _valid_time_slots(pred_valid, start: int, end: int) -> np.ndarray:
    """Convert HaWoR validity from (2,T) hand-major to AoE (T,2)."""
    valid = _to_numpy(pred_valid)[:, start:end].astype(bool)
    if valid.ndim != 2 or valid.shape[0] != 2:
        raise ValueError(f"expected pred_valid with shape (2,T); got {valid.shape}")
    return valid.T


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
