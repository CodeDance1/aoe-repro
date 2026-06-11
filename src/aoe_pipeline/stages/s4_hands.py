"""Stage 4 — hand reconstruction + world lift.

Substitute for HaWoR + MANO. MediaPipe Hands gives 21 2D landmarks (image px)
and 21 metric "world" landmarks per hand. We lift to 3D as follows:

  - per joint, back-project its 2D pixel through K^-1 at the monocular depth
    sampled at that pixel  ->  camera-frame 3D position (reprojection-consistent),
  - transform by the camera->world pose from Stage 3  ->  world coordinates,
  - Savitzky-Golay temporal smoothing for kinematic consistency.

Hands are placed in two fixed slots by handedness (0=Left, 1=Right) so the output
arrays are time-aligned for velocity-based QC. Missing detections are NaN.
"""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np

from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")

NUM_JOINTS = 21
SLOTS = ("Left", "Right")


@register("hands")
class HandStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from ..mp_assets import hand_landmarker_model

        frames = ctx.get_frames()
        T = len(frames)
        W, H = ctx.manifest.width, ctx.manifest.height
        K = ctx.manifest.intrinsics.K
        Kinv = np.linalg.inv(K)
        depths = ctx.blackboard.get("depth")  # list[(H,W) float32] or None
        poses = ctx.blackboard.get("poses")   # (T,4,4) cam->world or None

        joints_2d = np.full((T, 2, NUM_JOINTS, 2), np.nan, np.float32)
        joints_cam = np.full((T, 2, NUM_JOINTS, 3), np.nan, np.float32)
        joints_world = np.full((T, 2, NUM_JOINTS, 3), np.nan, np.float32)
        handed_per_frame: list[list] = [[None, None] for _ in range(T)]

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_landmarker_model())),
            num_hands=int(self.params.get("max_hands", 2)),
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        try:
            for t, frame in enumerate(frames):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_image)
                if not res.hand_landmarks:
                    continue
                for i, lms in enumerate(res.hand_landmarks):
                    label = "Right"
                    if res.handedness and i < len(res.handedness):
                        label = res.handedness[i][0].category_name
                    slot = 0 if label == "Left" else 1
                    pts = np.array([[lm.x * W, lm.y * H] for lm in lms], np.float32)
                    joints_2d[t, slot] = pts
                    anchor = self.params.get("depth_anchor", "wrist")
                    joints_cam[t, slot] = self._backproject(pts, Kinv, depths, t, W, H, anchor)
                    handed_per_frame[t][slot] = label
        finally:
            detector.close()

        # Temporal smoothing in the CAMERA frame. The hand's motion relative to
        # the camera is what should be kinematically smooth; smoothing in world
        # coordinates would couple it to the (scale-ambiguous) camera trajectory
        # and break reprojection consistency when the camera moves.
        window = int(self.params.get("smooth_window", 11))
        poly = int(self.params.get("smooth_poly", 3))
        for s in range(2):
            flat = joints_cam[:, s].reshape(T, NUM_JOINTS * 3)
            joints_cam[:, s] = _smooth_series(flat, window, poly).reshape(T, NUM_JOINTS, 3)

        # camera -> world (no smoothing here; trajectory handled by Stage 3)
        for t in range(T):
            P = poses[t] if (poses is not None and t < len(poses)) else np.eye(4)
            R, tvec = P[:3, :3], P[:3, 3]
            for s in range(2):
                if np.isnan(joints_cam[t, s]).any():
                    continue
                joints_world[t, s] = (R @ joints_cam[t, s].T).T + tvec

        # persist
        ctx.hands_dir.mkdir(parents=True, exist_ok=True)
        np.save(ctx.hands_dir / "joints_world.npy", joints_world)
        np.save(ctx.hands_dir / "joints_2d.npy", joints_2d)
        np.save(ctx.hands_dir / "joints_cam.npy", joints_cam)
        n_det = int(np.isfinite(joints_2d[..., 0]).any(axis=(1, 2)).sum())
        (ctx.hands_dir / "meta.json").write_text(
            json.dumps(
                {
                    "slots": list(SLOTS),
                    "num_joints": NUM_JOINTS,
                    "frames_total": T,
                    "frames_with_hand": n_det,
                    "handedness_per_frame": handed_per_frame,
                },
                indent=2,
            )
        )

        ctx.blackboard["joints_world"] = joints_world
        ctx.blackboard["joints_2d"] = joints_2d
        ctx.manifest.set_stage(
            self.name, "ok",
            frames_with_hand=n_det,
            depth_used=depths is not None,
            poses_used=poses is not None,
        )
        log.info("hands: detections in %d/%d frames (depth=%s, poses=%s)",
                 n_det, T, depths is not None, poses is not None)

        if self.params.get("viz", True):
            from ..viz import draw_hand_overlay

            draw_hand_overlay(ctx, joints_2d)

    @staticmethod
    def _backproject(pts2d, Kinv, depths, t, W, H, anchor="wrist") -> np.ndarray:
        """Back-project joints: X_cam = depth * K^-1 [u, v, 1].

        ``anchor='wrist'`` (default) places all 21 joints at the wrist's depth
        (a near-planar-hand approximation), which avoids per-joint monocular-depth
        jitter; ``anchor='per_joint'`` samples depth at every joint.
        """
        cam = np.empty((NUM_JOINTS, 3), np.float32)
        dmap = None
        if depths is not None and t < len(depths) and depths[t] is not None:
            dmap = depths[t]
            if dmap.shape[:2] != (H, W):
                dmap = cv2.resize(dmap, (W, H), interpolation=cv2.INTER_LINEAR)

        def sample(j: int) -> float:
            if dmap is None:
                return 1.0
            u = min(max(int(round(pts2d[j, 0])), 0), W - 1)
            v = min(max(int(round(pts2d[j, 1])), 0), H - 1)
            dv = float(dmap[v, u])
            return dv if (np.isfinite(dv) and dv > 0) else 1.0

        wrist_d = sample(0)
        for j in range(NUM_JOINTS):
            d = wrist_d if anchor == "wrist" else sample(j)
            ray = Kinv @ np.array([pts2d[j, 0], pts2d[j, 1], 1.0])
            cam[j] = ray * d
        return cam


def _smooth_series(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay smoothing over time for each column, NaN-aware.

    Interpolates across short gaps within the valid span; leaves frames outside
    the first/last valid detection as NaN.
    """
    from scipy.signal import savgol_filter

    T, C = arr.shape
    out = arr.copy()
    valid = ~np.isnan(arr).any(axis=1)
    idx = np.where(valid)[0]
    if len(idx) < 3:
        return out
    lo, hi = idx[0], idx[-1]
    span = np.arange(lo, hi + 1)
    w = min(window, len(span))
    if w % 2 == 0:
        w -= 1
    for c in range(C):
        col = arr[span, c]
        m = ~np.isnan(col)
        if m.sum() < 2:
            continue
        col_i = np.interp(span, span[m], col[m])
        if w >= 3 and w > poly:
            col_i = savgol_filter(col_i, w, poly)
        out[span, c] = col_i
    return out
