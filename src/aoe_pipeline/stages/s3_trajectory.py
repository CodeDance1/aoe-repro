"""Stage 3 — camera trajectory + monocular depth.

Substitute for MegaSAM + Lingbot-Depth:
  - monocular depth via Depth-Anything-V2-Small (HF transformers, MPS),
  - camera poses via OpenCV ORB visual odometry (essential matrix + recoverPose),
    accumulated into a camera->world SE(3) chain and exported as TUM.

Monocular VO is scale-ambiguous; translation is scaled by a fraction of the
median scene depth (or a fixed step). Evaluate trajectory with Sim(3)/scale-free
alignment (see eval/metrics.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..external import ExternalPipelineError, run_external
from ..utils import get_device
from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")


@register("trajectory")
class TrajectoryStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        frames = ctx.get_frames()
        K = ctx.manifest.intrinsics.K

        backend = self.params.get("backend", "substitute")
        if backend == "original":
            depths, poses = self._estimate_original(ctx, frames)
        else:
            depths = self._estimate_depth(ctx, frames)
            poses = self._estimate_poses(frames, K, depths)
        ctx.blackboard["depth"] = depths
        ctx.blackboard["poses"] = poses

        self._write_tum(ctx, poses)
        try:
            from ..viz import plot_trajectory

            ctx.viz_dir.mkdir(parents=True, exist_ok=True)
            plot_trajectory(poses, ctx.viz_dir / "trajectory.png")
        except Exception as exc:  # noqa: BLE001 — viz is non-critical
            log.warning("trajectory plot failed: %s", exc)

        ctx.manifest.set_stage(
            self.name, "ok",
            backend=backend,
            num_poses=len(poses),
            depth_model=self.params.get("depth_model"),
            device=get_device(),
        )

    # --- paper-faithful external adapter ---------------------------------------
    def _estimate_original(self, ctx: ClipContext, frames) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run/import the paper stack: LingBot-Depth + MegaSAM.

        AoE cites LingBot-Depth for depth refinement and MegaSAM for trajectory
        estimation. Their official implementations require separate CUDA/conda
        environments, so this adapter runs a user-configured command and imports
        standard artifacts back into this repo.
        """
        cfg = self.params.get("original", {}) or {}
        external_dir = ctx.clip_dir / "paper_original" / "trajectory"
        external_dir.mkdir(parents=True, exist_ok=True)
        intrinsics_json = external_dir / "intrinsics.json"
        intrinsics_json.write_text(
            __import__("json").dumps(ctx.manifest.intrinsics.to_dict(), indent=2)
        )

        values = {
            "video_path": ctx.video_path,
            "frames_dir": ctx.frames_dir,
            "clip_dir": ctx.clip_dir,
            "external_dir": external_dir,
            "intrinsics_json": intrinsics_json,
            "fps": ctx.fps,
        }
        run_external(
            cfg.get("command"),
            values,
            cwd=cfg.get("cwd"),
            shell=bool(cfg.get("shell", False)),
        )

        depths = self._load_external_depths(ctx, cfg, external_dir, len(frames))
        poses = self._load_external_poses(cfg, external_dir, len(frames))
        return depths, poses

    def _load_external_depths(
        self, ctx: ClipContext, cfg: dict, external_dir: Path, expected: int
    ) -> list[np.ndarray]:
        glob_pat = cfg.get("depth_glob", "depth/depth_*.npy")
        files = sorted(external_dir.glob(glob_pat))
        if not files:
            raise ExternalPipelineError(
                f"original trajectory backend expected LingBot/MegaSAM depth maps matching "
                f"{external_dir / glob_pat}"
            )
        if len(files) != expected:
            raise ExternalPipelineError(
                f"expected {expected} external depth maps, found {len(files)} from {glob_pat}"
            )
        ctx.depth_dir.mkdir(parents=True, exist_ok=True)
        depths = []
        for t, path in enumerate(files):
            arr = np.load(path).astype(np.float32)
            np.save(ctx.depth_dir / f"depth_{t:06d}.npy", arr)
            depths.append(arr)
        return depths

    def _load_external_poses(self, cfg: dict, external_dir: Path, expected: int) -> list[np.ndarray]:
        pose_path = external_dir / cfg.get("pose_path", "trajectory.tum")
        if pose_path.suffix == ".npy":
            poses = [p.astype(float) for p in np.load(pose_path)]
        else:
            poses = _read_tum_poses(pose_path)
        if len(poses) != expected:
            raise ExternalPipelineError(
                f"expected {expected} external poses, found {len(poses)} in {pose_path}"
            )
        return poses

    # --- depth -----------------------------------------------------------------
    def _estimate_depth(self, ctx: ClipContext, frames) -> list[np.ndarray]:
        import torch
        from PIL import Image
        from transformers import pipeline

        model_id = self.params.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf")
        device = get_device()
        near = float(self.params.get("depth_near", 0.3))
        far = float(self.params.get("depth_far", 3.0))
        log.info("loading depth model %s on %s", model_id, device)
        depther = pipeline("depth-estimation", model=model_id, device=torch.device(device))

        W, H = ctx.manifest.width, ctx.manifest.height
        ctx.depth_dir.mkdir(parents=True, exist_ok=True)
        depths = []
        for t, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred = depther(Image.fromarray(rgb))["predicted_depth"]
            arr = pred.squeeze().detach().float().cpu().numpy()
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
            # Depth-Anything outputs relative inverse-depth (higher = closer);
            # map to a pseudo-metric distance in [near, far].
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-6:
                dist = np.full((H, W), (near + far) / 2, np.float32)
            else:
                closeness = (arr - lo) / (hi - lo)
                dist = (near + (1.0 - closeness) * (far - near)).astype(np.float32)
            np.save(ctx.depth_dir / f"depth_{t:06d}.npy", dist)
            depths.append(dist)
        log.info("depth: estimated %d maps", len(depths))
        return depths

    # --- poses -----------------------------------------------------------------
    def _estimate_poses(self, frames, K, depths) -> list[np.ndarray]:
        nfeat = int(self.params.get("orb_features", 1500))
        scale_from_depth = bool(self.params.get("scale_from_depth", True))
        depth_frac = float(self.params.get("translation_depth_frac", 0.1))
        step_scale = float(self.params.get("step_scale", 0.05))

        orb = cv2.ORB_create(nfeatures=nfeat)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        feats = [orb.detectAndCompute(g, None) for g in grays]

        poses = [np.eye(4)]
        degenerate = 0
        for i in range(1, len(frames)):
            (kp1, d1), (kp2, d2) = feats[i - 1], feats[i]
            rel = np.eye(4)
            ok = False
            if d1 is not None and d2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
                knn = bf.knnMatch(d1, d2, k=2)
                good = [mn[0] for mn in knn if len(mn) == 2 and mn[0].distance < 0.75 * mn[1].distance]
                if len(good) >= 8:
                    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
                    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
                    E, mask = cv2.findEssentialMat(
                        p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    if E is not None and E.shape[0] >= 3:
                        E = E[:3, :3]
                        _, R, tvec, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
                        scale = float(np.median(depths[i - 1])) * depth_frac if (
                            scale_from_depth and depths
                        ) else step_scale
                        t_rel = tvec.reshape(3) * scale
                        # recoverPose gives X_i = R X_{i-1} + t  (prev-cam -> cur-cam).
                        # cam->world chain needs the inverse motion.
                        T_cur_prev = np.eye(4)
                        T_cur_prev[:3, :3] = R
                        T_cur_prev[:3, 3] = t_rel
                        rel = np.linalg.inv(T_cur_prev)
                        ok = True
            if not ok:
                degenerate += 1
            poses.append(poses[-1] @ rel)
        if degenerate:
            log.info("trajectory: %d/%d frame pairs degenerate (held constant)",
                     degenerate, len(frames) - 1)
        return poses

    # --- io --------------------------------------------------------------------
    def _write_tum(self, ctx: ClipContext, poses) -> None:
        from scipy.spatial.transform import Rotation

        fps = ctx.fps
        lines = []
        for t, P in enumerate(poses):
            ts = t / fps
            tx, ty, tz = P[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(P[:3, :3]).as_quat()
            lines.append(
                f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
            )
        (ctx.clip_dir / "trajectory.tum").write_text("\n".join(lines) + "\n")


def _read_tum_poses(path: Path) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation

    if not path.exists():
        raise ExternalPipelineError(f"expected external trajectory file at {path}")
    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [float(x) for x in line.split()]
        if len(parts) != 8:
            raise ExternalPipelineError(f"invalid TUM pose line in {path}: {line}")
        _, tx, ty, tz, qx, qy, qz, qw = parts
        P = np.eye(4)
        P[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        P[:3, 3] = [tx, ty, tz]
        poses.append(P)
    return poses
