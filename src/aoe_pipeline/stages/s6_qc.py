"""Stage 6 — quality control / filtering.

Flags kinematic outliers (per-joint velocity z-score > sigma) and high
reprojection error (> px), records per-clip pass/fail, and samples 5% of flagged
frames for manual inspection (the paper's human-in-the-loop check).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from .. import qc
from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")


@register("qc")
class QCStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        joints_world = _load(ctx, "joints_world")
        joints_2d = _load(ctx, "joints_2d")
        if joints_world is None or joints_2d is None:
            ctx.manifest.set_stage(self.name, "skipped", reason="no hand joints")
            log.info("qc: skipped (no hand joints)")
            return

        sigma = float(self.params.get("velocity_sigma", 3.0))
        px = float(self.params.get("reproj_px", 5.0))
        max_bad_ratio = float(self.params.get("max_bad_ratio", 0.1))
        dt = 1.0 / ctx.fps
        K = ctx.manifest.intrinsics.K
        poses = ctx.blackboard.get("poses")

        kin_flags, vel = qc.kinematic_outliers(joints_world, dt, sigma)
        err = qc.reprojection_error(joints_world, joints_2d, poses, K)
        reproj_flags = qc.reprojection_outliers(err, px)

        any_flags = kin_flags | reproj_flags
        bad_frames = np.where(qc.frames_flagged(any_flags))[0]
        n_with_hand = int(np.isfinite(joints_2d[..., 0]).any(axis=(1, 2)).sum())
        bad_ratio = (len(bad_frames) / n_with_hand) if n_with_hand else 0.0
        passed = bad_ratio <= max_bad_ratio

        # deterministic 5% manual-inspection sample of flagged frames
        rng = np.random.default_rng(0)
        n_inspect = int(np.ceil(0.05 * len(bad_frames))) if len(bad_frames) else 0
        inspect = sorted(rng.choice(bad_frames, size=min(n_inspect, len(bad_frames)),
                                    replace=False).tolist()) if n_inspect else []

        finite_err = err[np.isfinite(err)]
        report = {
            "thresholds": {"velocity_sigma": sigma, "reproj_px": px,
                           "max_bad_ratio": max_bad_ratio},
            "frames_total": int(joints_world.shape[0]),
            "frames_with_hand": n_with_hand,
            "kinematic_outlier_count": int(kin_flags.sum()),
            "reprojection_outlier_count": int(reproj_flags.sum()),
            "flagged_frames": [int(x) for x in bad_frames],
            "flagged_frame_ratio": round(bad_ratio, 4),
            "reprojection_px": {
                "mean": _f(finite_err.mean() if finite_err.size else np.nan),
                "max": _f(finite_err.max() if finite_err.size else np.nan),
            },
            "pass": bool(passed),
            "manual_inspect_frames": [int(x) for x in inspect],
        }
        (ctx.clip_dir / "qc_report.json").write_text(json.dumps(report, indent=2))
        ctx.blackboard["qc_report"] = report
        ctx.manifest.set_stage(
            self.name, "ok",
            passed=bool(passed),
            flagged_frames=len(bad_frames),
            kinematic_outliers=int(kin_flags.sum()),
            reprojection_outliers=int(reproj_flags.sum()),
        )
        log.info("qc: pass=%s, %d flagged frames (%.1f%%), mean reproj=%.2fpx",
                 passed, len(bad_frames), 100 * bad_ratio,
                 finite_err.mean() if finite_err.size else float("nan"))


def _load(ctx: ClipContext, name: str):
    arr = ctx.blackboard.get(name)
    if arr is not None:
        return arr
    path = ctx.hands_dir / f"{name}.npy"
    return np.load(path) if path.exists() else None


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)
