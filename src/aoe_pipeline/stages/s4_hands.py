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
from pathlib import Path

import cv2
import numpy as np

from ..external import ExternalPipelineError, run_external
from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")

NUM_JOINTS = 21
SLOTS = ("Left", "Right")


@register("hands")
class HandStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        backend = self.params.get("backend", "substitute")
        if backend == "original":
            self._run_original(ctx)
            return
        if backend == "hybrid":
            self._run_hybrid(ctx)
            return
        self._run_substitute(ctx)

    def _run_substitute(self, ctx: ClipContext) -> None:
        detection = self._run_mediapipe_detection(ctx)
        joints_2d = detection["joints_2d"]
        joints_cam = detection["joints_cam"]
        joints_world = detection["joints_world"]
        handed_per_frame = detection["handedness_per_frame"]
        T = int(joints_2d.shape[0])
        depths = ctx.blackboard.get("depth")
        poses = ctx.blackboard.get("poses")

        ctx.hands_dir.mkdir(parents=True, exist_ok=True)
        np.save(ctx.hands_dir / "joints_world.npy", joints_world)
        np.save(ctx.hands_dir / "joints_2d.npy", joints_2d)
        np.save(ctx.hands_dir / "joints_cam.npy", joints_cam)
        self._save_mediapipe_hints(ctx.hands_dir / "mediapipe_hints.npz", detection)
        n_det = int(detection["presence"].any(axis=1).sum())
        (ctx.hands_dir / "meta.json").write_text(
            json.dumps(
                {
                    "backend": "substitute",
                    "method": "MediaPipe",
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
        ctx.blackboard["joints_cam"] = joints_cam
        ctx.manifest.set_stage(
            self.name, "ok",
            backend="substitute",
            method="MediaPipe",
            frames_with_hand=n_det,
            depth_used=depths is not None,
            poses_used=poses is not None,
        )
        log.info("hands: detections in %d/%d frames (depth=%s, poses=%s)",
                 n_det, T, depths is not None, poses is not None)

        if self.params.get("viz", True):
            from ..viz import draw_hand_overlay

            draw_hand_overlay(ctx, joints_2d)

    def _run_mediapipe_detection(self, ctx: ClipContext) -> dict:
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
        boxes = np.full((T, 2, 4), np.nan, np.float32)
        presence = np.zeros((T, 2), dtype=bool)
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
                    x1, y1 = np.nanmin(pts, axis=0)
                    x2, y2 = np.nanmax(pts, axis=0)
                    boxes[t, slot] = [
                        np.clip(x1, 0, W - 1),
                        np.clip(y1, 0, H - 1),
                        np.clip(x2, 0, W - 1),
                        np.clip(y2, 0, H - 1),
                    ]
                    presence[t, slot] = True
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

        return {
            "joints_2d": joints_2d,
            "joints_cam": joints_cam,
            "joints_world": joints_world,
            "handedness_per_frame": handed_per_frame,
            "boxes": boxes,
            "presence": presence,
        }

    def _run_hybrid(self, ctx: ClipContext) -> None:
        """Run MediaPipe hints, invoke HaWoR, and import HaWoR/MANO outputs."""
        cfg = self.params.get("hybrid", {}) or {}
        hawor_cfg = cfg.get("original", {}) or self.params.get("original", {}) or {}
        detection = self._run_mediapipe_detection(ctx)
        T = int(detection["joints_2d"].shape[0])

        ctx.hands_dir.mkdir(parents=True, exist_ok=True)
        hints_path = ctx.hands_dir / "mediapipe_hints.npz"
        self._save_mediapipe_hints(hints_path, detection)

        external_dir = ctx.clip_dir / "paper_original" / "hands_hybrid"
        external_dir.mkdir(parents=True, exist_ok=True)
        intrinsics_json = external_dir / "intrinsics.json"
        intrinsics_json.write_text(json.dumps(ctx.manifest.intrinsics.to_dict(), indent=2))

        values = {
            "video_path": ctx.video_path,
            "frames_dir": ctx.frames_dir,
            "clip_dir": ctx.clip_dir,
            "external_dir": external_dir,
            "intrinsics_json": intrinsics_json,
            "trajectory_tum": ctx.clip_dir / "trajectory.tum",
            "mediapipe_hints": hints_path,
            "fps": ctx.fps,
        }
        run_external(
            hawor_cfg.get("command"),
            values,
            cwd=hawor_cfg.get("cwd"),
            shell=bool(hawor_cfg.get("shell", False)),
        )

        joints_world = _load_hawor_joints(
            external_dir / hawor_cfg.get("joints_world", "joints_world.npy"),
            "joints_world",
            T,
            3,
        )
        joints_2d = _load_hawor_joints(
            external_dir / hawor_cfg.get("joints_2d", "joints_2d.npy"),
            "joints_2d",
            T,
            2,
        )
        cam_path = external_dir / hawor_cfg.get("joints_cam", "joints_cam.npy")
        if cam_path.exists():
            joints_cam = _load_hawor_joints(cam_path, "joints_cam", T, 3)
        else:
            joints_cam = np.full((T, 2, NUM_JOINTS, 3), np.nan, np.float32)

        hawor_present = _slot_presence(joints_2d) | _slot_presence(joints_world)
        fallback_slots = np.zeros((T, 2), dtype=bool)
        if bool(cfg.get("fallback_to_mediapipe", True)):
            mp_present = detection["presence"]
            fallback_slots = (~hawor_present) & mp_present
            for t in range(T):
                for s in range(2):
                    if not fallback_slots[t, s]:
                        continue
                    joints_2d[t, s] = detection["joints_2d"][t, s]
                    joints_cam[t, s] = detection["joints_cam"][t, s]
                    joints_world[t, s] = detection["joints_world"][t, s]

        optional_meshes = self._import_optional_meshes(external_dir, hawor_cfg, T)
        hawor_meta_imported = self._import_optional_hawor_meta(external_dir, ctx.hands_dir)

        np.save(ctx.hands_dir / "joints_world.npy", joints_world.astype(np.float32))
        np.save(ctx.hands_dir / "joints_2d.npy", joints_2d.astype(np.float32))
        np.save(ctx.hands_dir / "joints_cam.npy", joints_cam.astype(np.float32))
        for name, arr in optional_meshes.items():
            np.save(ctx.hands_dir / name, arr)

        final_present = _slot_presence(joints_2d) | _slot_presence(joints_world)
        n_det = int(final_present.any(axis=1).sum())
        hawor_frames = int(hawor_present.any(axis=1).sum())
        fallback_frames = int(fallback_slots.any(axis=1).sum())
        (ctx.hands_dir / "meta.json").write_text(
            json.dumps(
                {
                    "backend": "hybrid",
                    "method": "MediaPipe+HaWoR+MANO",
                    "slots": list(SLOTS),
                    "num_joints": NUM_JOINTS,
                    "frames_total": T,
                    "frames_with_hand": n_det,
                    "hawor_frames": hawor_frames,
                    "mediapipe_fallback_frames": fallback_frames,
                    "mesh_files": sorted(optional_meshes),
                    "hawor_meta": "hawor_meta.json" if hawor_meta_imported else None,
                    "handedness_per_frame": detection["handedness_per_frame"],
                },
                indent=2,
            )
        )
        ctx.blackboard["joints_world"] = joints_world
        ctx.blackboard["joints_2d"] = joints_2d
        ctx.blackboard["joints_cam"] = joints_cam
        ctx.manifest.set_stage(
            self.name,
            "ok",
            backend="hybrid",
            method="MediaPipe+HaWoR+MANO",
            frames_with_hand=n_det,
            hawor_frames=hawor_frames,
            mediapipe_fallback_frames=fallback_frames,
            meshes_imported=sorted(optional_meshes),
            hawor_meta_imported=hawor_meta_imported,
        )
        log.info(
            "hands(hybrid): imported HaWoR/MANO in %d frames; MediaPipe fallback in %d frames",
            hawor_frames,
            fallback_frames,
        )

        if self.params.get("viz", True):
            from ..viz import draw_hand_overlay

            draw_hand_overlay(ctx, joints_2d)

    def _run_original(self, ctx: ClipContext) -> None:
        """Run/import the paper stack: HaWoR + MANO world-space hands."""
        cfg = self.params.get("original", {}) or {}
        external_dir = ctx.clip_dir / "paper_original" / "hands"
        external_dir.mkdir(parents=True, exist_ok=True)
        intrinsics_json = external_dir / "intrinsics.json"
        intrinsics_json.write_text(json.dumps(ctx.manifest.intrinsics.to_dict(), indent=2))

        values = {
            "video_path": ctx.video_path,
            "frames_dir": ctx.frames_dir,
            "clip_dir": ctx.clip_dir,
            "external_dir": external_dir,
            "intrinsics_json": intrinsics_json,
            "trajectory_tum": ctx.clip_dir / "trajectory.tum",
            "fps": ctx.fps,
        }
        run_external(
            cfg.get("command"),
            values,
            cwd=cfg.get("cwd"),
            shell=bool(cfg.get("shell", False)),
        )

        joints_world = _load_required_array(
            external_dir / cfg.get("joints_world", "joints_world.npy")
        )
        joints_2d = _load_required_array(external_dir / cfg.get("joints_2d", "joints_2d.npy"))
        cam_path = external_dir / cfg.get("joints_cam", "joints_cam.npy")
        joints_cam = np.load(cam_path) if cam_path.exists() else np.full_like(joints_world, np.nan)

        if joints_world.ndim != 4 or joints_world.shape[2:] != (NUM_JOINTS, 3):
            raise ExternalPipelineError(
                "HaWoR adapter expects joints_world with shape (T,2,21,3)"
            )
        if joints_2d.ndim != 4 or joints_2d.shape[2:] != (NUM_JOINTS, 2):
            raise ExternalPipelineError("HaWoR adapter expects joints_2d with shape (T,2,21,2)")

        ctx.hands_dir.mkdir(parents=True, exist_ok=True)
        np.save(ctx.hands_dir / "joints_world.npy", joints_world)
        np.save(ctx.hands_dir / "joints_2d.npy", joints_2d)
        np.save(ctx.hands_dir / "joints_cam.npy", joints_cam)
        n_det = int(np.isfinite(joints_2d[..., 0]).any(axis=(1, 2)).sum())
        (ctx.hands_dir / "meta.json").write_text(
            json.dumps(
                {
                    "backend": "original",
                    "method": "HaWoR+MANO",
                    "slots": list(SLOTS),
                    "num_joints": NUM_JOINTS,
                    "frames_total": int(joints_2d.shape[0]),
                    "frames_with_hand": n_det,
                },
                indent=2,
            )
        )
        ctx.blackboard["joints_world"] = joints_world
        ctx.blackboard["joints_2d"] = joints_2d
        ctx.manifest.set_stage(
            self.name,
            "ok",
            backend="original",
            method="HaWoR+MANO",
            frames_with_hand=n_det,
        )
        log.info("hands(original): imported HaWoR/MANO detections in %d frames", n_det)

    def _import_optional_meshes(
        self,
        external_dir: Path,
        cfg: dict,
        expected_t: int,
    ) -> dict[str, np.ndarray]:
        meshes = {}
        mesh_paths = {
            "verts_cam": external_dir / cfg.get("verts_cam", "verts_cam.npy"),
            "verts_world": external_dir / cfg.get("verts_world", "verts_world.npy"),
            "faces": external_dir / cfg.get("faces", "faces.npy"),
        }
        existing = {name for name, path in mesh_paths.items() if path.exists()}
        if existing and existing != set(mesh_paths):
            missing = sorted(set(mesh_paths) - existing)
            raise ExternalPipelineError(
                "HaWoR adapter got partial MANO mesh outputs; missing "
                + ", ".join(f"{name}.npy" for name in missing)
            )
        for name in ("verts_cam", "verts_world"):
            path = mesh_paths[name]
            if not path.exists():
                continue
            arr = np.load(path)
            if (
                arr.ndim != 4
                or arr.shape[0] != expected_t
                or arr.shape[1] != 2
                or arr.shape[-1] != 3
            ):
                raise ExternalPipelineError(
                    f"HaWoR adapter expects {name} with shape (T,2,V,3); got {arr.shape}"
                )
            meshes[f"{name}.npy"] = arr.astype(np.float32)
        faces_path = mesh_paths["faces"]
        if faces_path.exists():
            faces = np.load(faces_path)
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ExternalPipelineError(
                    f"HaWoR adapter expects faces with shape (F,3); got {faces.shape}"
                )
            if meshes and len(faces) and int(faces.max()) >= meshes["verts_cam.npy"].shape[2]:
                raise ExternalPipelineError(
                    "HaWoR adapter faces reference vertices outside verts_*.npy"
                )
            meshes["faces.npy"] = faces.astype(np.int32)
        return meshes

    @staticmethod
    def _import_optional_hawor_meta(external_dir: Path, hands_dir: Path) -> bool:
        src = external_dir / "meta.json"
        if not src.exists():
            return False
        dst = hands_dir / "hawor_meta.json"
        dst.write_text(src.read_text())
        return True

    @staticmethod
    def _save_mediapipe_hints(path: Path, detection: dict) -> None:
        np.savez_compressed(
            path,
            joints_2d=detection["joints_2d"].astype(np.float32),
            joints_cam=detection["joints_cam"].astype(np.float32),
            joints_world=detection["joints_world"].astype(np.float32),
            boxes=detection["boxes"].astype(np.float32),
            presence=detection["presence"].astype(bool),
            handedness=_handedness_array(detection["handedness_per_frame"]),
            slot_labels=np.asarray(SLOTS),
        )

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


def _load_required_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise ExternalPipelineError(f"expected HaWoR output array at {path}")
    return np.load(path)


def _load_hawor_joints(path: Path, name: str, expected_t: int, dims: int) -> np.ndarray:
    arr = _load_required_array(path)
    expected_tail = (NUM_JOINTS, dims)
    if (
        arr.ndim != 4
        or arr.shape[0] != expected_t
        or arr.shape[1] != 2
        or arr.shape[2:] != expected_tail
    ):
        raise ExternalPipelineError(
            f"HaWoR adapter expects {name} with shape (T,2,21,{dims}) "
            f"and T={expected_t}; got {arr.shape}"
        )
    return arr.astype(np.float32)


def _slot_presence(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 3:
        raise ExternalPipelineError(
            f"expected hand array with at least 3 dimensions; got {arr.shape}"
        )
    return np.isfinite(arr).any(axis=tuple(range(2, arr.ndim)))


def _handedness_array(handedness_per_frame: list[list]) -> np.ndarray:
    out = np.full((len(handedness_per_frame), 2), "", dtype="<U8")
    for t, labels in enumerate(handedness_per_frame):
        for s, label in enumerate(labels[:2]):
            if label is not None:
                out[t, s] = str(label)
    return out
