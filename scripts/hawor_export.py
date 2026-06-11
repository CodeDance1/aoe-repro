#!/usr/bin/env python3
"""Run HaWoR on a video and export to the aoe-pipeline schema.

RUNS INSIDE HaWoR's OWN ENV (py3.10 / torch1.13 / cu117) on a CUDA box — invoked
by the `hands_hawor` stage via `conda run -n hawor python scripts/hawor_export.py …`.

HaWoR's demo.py only renders visualizations; this wrapper calls the same pipeline
(detect_track_video -> hawor_motion_estimation -> hawor_slam/load_slam_cam ->
hawor_infiller -> run_mano[_left]) and dumps tensors to:

    <out>/hands/hawor_raw.npz   trans (2,n,3), global_orient (2,n,3,3),
                                hand_pose (2,n,45), betas (2,n,10), valid (2,n),
                                joints_world (n,2,21,3), joints_2d (n,2,21,2)
    <out>/trajectory.tum        metric camera-to-world trajectory (n frames)

Hand slot 0=Left (run_mano_left), 1=Right (run_mano) — matches HaWoR hand2idx and
the rest of the repo.

JOINT ORDER: HaWoR's MANO wrapper (lib/models/mano_wrapper.py) applies
``joint_map = [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]`` and returns
**MediaPipe-21 / OpenPose hand order** (wrist; thumb+tip; index+tip; …) — which is
ALSO the repo's storage convention. So joints are stored verbatim and the loader
does NOT remap them. (VERIFY on the GPU box: ``run_mano(...).joints.shape[1] == 21``
and that ``joint_map`` matches the pinned HaWoR commit; an smplx-default MANO would
return 16 — then assemble joints to MediaPipe order before storing.)

The pure assembly (frame reconciliation, validity masking, projection, TUM) is
``assemble_export()`` — unit-tested off-GPU; only the HaWoR forward calls in
``main()`` need the GPU box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

NUM_JOINTS = 21


def assemble_export(joints_per_hand, pred_trans, pred_rot, pred_hand_pose, pred_betas,
                    pred_valid, R_w2c, t_w2c, R_c2w, t_c2w, img_focal, width, height,
                    fps) -> dict:
    """Pure glue: HaWoR raw tensors -> aoe-pipeline export arrays. GPU-free/testable.

    ``joints_per_hand``: (2, T, 21, 3) world joints, MediaPipe-21 order.
    ``pred_*``: (2, T, …); ``R_*``/``t_*``: (T, 3, 3) / (T, 3). Everything is
    reconciled to a single frame count ``n`` (Bug-2 fix) and invalid frames are
    NaN-masked (Bug-3 fix).
    """
    j = np.asarray(joints_per_hand, np.float32)
    valid = np.asarray(pred_valid).astype(bool)          # (2, T)
    n = min(j.shape[1], len(R_w2c), len(R_c2w), pred_trans.shape[1], valid.shape[1])
    if not (j.shape[1] == len(R_w2c) == len(R_c2w) == pred_trans.shape[1] == valid.shape[1]):
        print(f"hawor_export: WARNING unequal frame counts "
              f"(joints={j.shape[1]} R_w2c={len(R_w2c)} R_c2w={len(R_c2w)} "
              f"trans={pred_trans.shape[1]} valid={valid.shape[1]}); truncating to {n}",
              file=sys.stderr)

    joints_world = np.transpose(j[:, :n], (1, 0, 2, 3)).copy()  # (n, 2, 21, 3)
    valid = valid[:, :n]
    for slot in (0, 1):                                  # NaN-mask invalid frames
        joints_world[~valid[slot], slot] = np.nan

    K = np.array([[img_focal, 0, width / 2], [0, img_focal, height / 2], [0, 0, 1.0]])
    joints_2d = _project(joints_world, _arr(R_w2c)[:n], _arr(t_w2c)[:n], K)

    return {
        "trans": _arr(pred_trans)[:, :n], "global_orient": _arr(pred_rot)[:, :n],
        "hand_pose": _arr(pred_hand_pose)[:, :n], "betas": _arr(pred_betas)[:, :n],
        "valid": valid, "joints_world": joints_world, "joints_2d": joints_2d,
        "tum_lines": _tum_lines(_arr(R_c2w)[:n], _arr(t_c2w)[:n], fps),
    }


def _project(joints_world, R_w2c, t_w2c, K) -> np.ndarray:
    """World joints -> pixels; NaN for missing OR behind-camera (z<=0) joints (Bug-5)."""
    n = joints_world.shape[0]
    out = np.full((n, 2, NUM_JOINTS, 2), np.nan, np.float32)
    for t in range(n):
        for s in range(joints_world.shape[1]):
            X = joints_world[t, s]
            if not np.isfinite(X).all():
                continue
            Xc = (R_w2c[t] @ X.T).T + t_w2c[t]           # (21, 3) camera frame
            z = Xc[:, 2]
            front = z > 1e-6
            uv = (K @ Xc.T).T
            proj = np.full((NUM_JOINTS, 2), np.nan, np.float32)
            proj[front] = uv[front, :2] / z[front, None]
            out[t, s] = proj
    return out


def _tum_lines(R_c2w, t_c2w, fps: float) -> list[str]:
    from scipy.spatial.transform import Rotation

    lines = []
    for t in range(len(R_c2w)):
        qx, qy, qz, qw = Rotation.from_matrix(R_c2w[t]).as_quat()
        tx, ty, tz = t_c2w[t]
        lines.append(f"{t / fps:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                     f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="aoe-pipeline clip dir to write into")
    ap.add_argument("--hawor-dir", required=True, help="path to the HaWoR repo checkout")
    ap.add_argument("--img-focal", type=float, default=None)
    args = ap.parse_args()

    import os

    hawor_dir = Path(args.hawor_dir).resolve()
    sys.path.insert(0, str(hawor_dir))
    os.chdir(hawor_dir)  # HaWoR resolves weights/ relative to its repo root

    import cv2
    import torch  # noqa: F401  (HaWoR env)
    from lib.eval_utils.custom_utils import run_mano, run_mano_left  # VERIFY import path
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_slam import load_slam_cam  # VERIFY import path
    from scripts.scripts_test_video.hawor_video import (
        hawor_infiller,
        hawor_motion_estimation,
        hawor_slam,
    )

    class A:  # the demo's args namespace
        video_path = str(Path(args.video).resolve())
        img_focal = args.img_focal
        checkpoint = str(hawor_dir / "weights/hawor/checkpoints/hawor.ckpt")
        infiller_weight = str(hawor_dir / "weights/hawor/checkpoints/infiller.pt")
        vis_mode = "world"

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(A)
    frame_chunks_all, img_focal = hawor_motion_estimation(A, start_idx, end_idx, seq_folder)
    slam_path = hawor_slam(A, start_idx, end_idx)
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        A, start_idx, end_idx, frame_chunks_all)

    T = pred_trans.shape[1]
    joints = np.full((2, T, NUM_JOINTS, 3), np.nan, np.float32)
    for slot, runner in ((0, run_mano_left), (1, run_mano)):
        out = runner(pred_trans[slot:slot + 1], pred_rot[slot:slot + 1],
                     pred_hand_pose[slot:slot + 1], betas=pred_betas[slot:slot + 1])
        joints[slot] = _extract_joints21(out)

    cap = cv2.VideoCapture(A.video_path)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    focal = img_focal if args.img_focal is None else args.img_focal

    ex = assemble_export(joints, pred_trans, pred_rot, pred_hand_pose, pred_betas,
                         pred_valid, R_w2c, t_w2c, R_c2w, t_c2w, focal, W, H, fps)

    out_dir = Path(args.out)
    (out_dir / "hands").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "hands" / "hawor_raw.npz",
                        **{k: ex[k] for k in ("trans", "global_orient", "hand_pose",
                                              "betas", "valid", "joints_world", "joints_2d")})
    (out_dir / "trajectory.tum").write_text("\n".join(ex["tum_lines"]) + "\n")
    print(f"hawor_export: {len(ex['tum_lines'])} frames -> {out_dir}/hands/hawor_raw.npz + trajectory.tum")
    return 0


def _extract_joints21(mano_out) -> np.ndarray:
    """(T,21,3) world joints in MediaPipe-21 order from a run_mano output.

    HaWoR's MANO subclass already returns 21 joints in MediaPipe order (see module
    docstring). If a 16-joint (smplx-default) output is encountered, fail loudly so
    the joint source is pinned on the GPU box rather than shipping a wrong remap.
    """
    if "joints" not in mano_out:
        raise RuntimeError("run_mano output lacks 'joints'; pin the joint source on the GPU box")
    j = _arr(mano_out["joints"])
    j = j.reshape(-1, j.shape[-2], 3)
    if j.shape[1] != NUM_JOINTS:
        raise RuntimeError(
            f"run_mano returned {j.shape[1]} joints, expected {NUM_JOINTS} (MediaPipe order). "
            "Confirm HaWoR's MANO joint_map on the GPU box.")
    return j


def _arr(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


if __name__ == "__main__":
    raise SystemExit(main())
