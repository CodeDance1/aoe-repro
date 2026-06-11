"""Stage 1 — ingest & calibration.

Decode the video to frames and resolve camera intrinsics. Paper uses Android
Camera2 factory intrinsics; here we resolve, in order:
  1. a metadata sidecar ``<video>.intrinsics.json`` (fx, fy, cx, cy[, distortion]),
  2. a checkerboard calibration set (if ``checkerboard_dir`` param is given),
  3. an FOV-based pinhole default.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

from ..schema import CameraIntrinsics
from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")


@register("ingest")
class IngestStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        stride = max(1, int(self.params.get("stride", 1)))
        max_frames = int(self.params.get("max_frames", 0))
        hfov = float(self.params.get("hfov_deg", 70.0))

        ctx.frames_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(ctx.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {ctx.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        frames = []
        idx = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                cv2.imwrite(str(ctx.frames_dir / f"frame_{saved:06d}.png"), frame)
                frames.append(frame)
                saved += 1
                if max_frames and saved >= max_frames:
                    break
            idx += 1
        cap.release()

        if saved == 0:
            raise RuntimeError("decoded 0 frames")

        h, w = frames[0].shape[:2]
        eff_fps = (fps / stride) if fps > 0 else 30.0
        intr = self._resolve_intrinsics(ctx, w, h, hfov)

        ctx.blackboard["frames"] = frames
        ctx.blackboard["fps"] = eff_fps
        ctx.manifest.fps = eff_fps
        ctx.manifest.num_frames = saved
        ctx.manifest.width, ctx.manifest.height = w, h
        ctx.manifest.intrinsics = intr
        ctx.manifest.set_stage(
            self.name, "ok", num_frames=saved, fps=eff_fps, intrinsics_source=intr.source
        )
        log.info("ingest: %d frames @ %.2f fps (%dx%d), intrinsics=%s",
                 saved, eff_fps, w, h, intr.source)

    def _resolve_intrinsics(self, ctx: ClipContext, w: int, h: int, hfov: float) -> CameraIntrinsics:
        sidecar = Path(str(ctx.video_path) + ".intrinsics.json")
        if sidecar.exists():
            d = json.loads(sidecar.read_text())
            return CameraIntrinsics(
                fx=d["fx"], fy=d["fy"], cx=d.get("cx", w / 2), cy=d.get("cy", h / 2),
                width=w, height=h, distortion=d.get("distortion", [0.0] * 5), source="metadata",
            )

        cb_dir = self.params.get("checkerboard_dir")
        if cb_dir:
            intr = _calibrate_checkerboard(Path(cb_dir), self.params.get("checkerboard", [9, 6]))
            if intr is not None:
                intr.width, intr.height = w, h
                return intr

        return CameraIntrinsics.from_fov(w, h, hfov_deg=hfov, source="fov_default")


def _calibrate_checkerboard(cb_dir: Path, pattern) -> CameraIntrinsics | None:
    """OpenCV checkerboard calibration over images in ``cb_dir``."""
    import numpy as np

    cols, rows = int(pattern[0]), int(pattern[1])
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_pts, img_pts = [], []
    gray = None
    for img_path in sorted(cb_dir.glob("*")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        obj_pts.append(objp)
        img_pts.append(corners)
    if not obj_pts or gray is None:
        return None
    _, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)
    return CameraIntrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=0, height=0, distortion=[float(x) for x in dist.ravel()[:5]], source="checkerboard",
    )
