"""Stage 5 — data augmentation (optional, off by default).

Lightweight substitute for the paper's GAN/diffusion augmentation:
  - background replacement via MediaPipe Selfie Segmentation + compositing,
  - hand removal via OpenCV inpainting over a dilated hand mask.

Writes augmented frames to ``augment/`` and leaves the labels untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")


@register("augment")
class AugmentStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        frames = ctx.get_frames()
        bg_path = self.params.get("background")
        inpaint = bool(self.params.get("inpaint_hands", False))
        if not bg_path and not inpaint:
            ctx.manifest.set_stage(self.name, "skipped", reason="no augmentation configured")
            return

        out_dir = ctx.clip_dir / "augment"
        out_dir.mkdir(parents=True, exist_ok=True)
        bg = self._load_bg(bg_path, frames[0].shape) if bg_path else None
        masks = self._person_masks(frames) if bg is not None else None
        j2d = ctx.blackboard.get("joints_2d")
        if j2d is None and (ctx.hands_dir / "joints_2d.npy").exists():
            j2d = np.load(ctx.hands_dir / "joints_2d.npy")

        for t, frame in enumerate(frames):
            img = frame
            if bg is not None and masks is not None:
                m = masks[t][..., None]
                img = (img * m + bg * (1 - m)).astype(np.uint8)
            if inpaint and j2d is not None:
                img = _inpaint_hands(img, j2d[t])
            cv2.imwrite(str(out_dir / f"aug_{t:06d}.png"), img)

        ctx.manifest.set_stage(self.name, "ok", background=bool(bg_path),
                               inpaint_hands=inpaint, num_frames=len(frames))
        log.info("augment: wrote %d frames (bg=%s, inpaint=%s)",
                 len(frames), bool(bg_path), inpaint)

    def _load_bg(self, path, shape) -> np.ndarray:
        bg = cv2.imread(str(path))
        if bg is None:
            raise RuntimeError(f"cannot read background image: {path}")
        return cv2.resize(bg, (shape[1], shape[0]))

    def _person_masks(self, frames) -> list[np.ndarray]:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from ..mp_assets import selfie_segmenter_model

        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(selfie_segmenter_model())),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        masks = []
        seg = vision.ImageSegmenter.create_from_options(options)
        try:
            for f in frames:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                )
                res = seg.segment(mp_image)
                conf = np.squeeze(res.confidence_masks[0].numpy_view())  # -> (H, W)
                masks.append((conf > 0.5).astype(np.float32))
        finally:
            seg.close()
        return masks


def _inpaint_hands(img: np.ndarray, j2d_frame: np.ndarray) -> np.ndarray:
    mask = np.zeros(img.shape[:2], np.uint8)
    drew = False
    for s in range(j2d_frame.shape[0]):
        pts = j2d_frame[s]
        if np.isnan(pts).any():
            continue
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        drew = True
    if not drew:
        return img
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
