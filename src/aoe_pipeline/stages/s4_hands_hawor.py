"""Faithful Stage 4 — HaWoR + MANO hands (subprocess adapter).

Runs HaWoR (CVPR'25, the hand-reconstruction model the AoE paper names) in its own
conda env via ``scripts/hawor_export.py`` and loads its export. HaWoR is
end-to-end — video in, world-space MANO hands + a *metric* camera out (masked
DROID-SLAM + Metric3D + infiller, bundled hand detector) — so this stage
**supersedes both** the lite ``trajectory`` and ``hands`` stages: it populates
``poses``/``joints_world``/``joints_2d``/``mano`` and writes ``trajectory.tum``.

Gated: skips (with a reason) unless the HaWoR env + repo are configured in params,
so the lite profile and CI stay green on machines without CUDA/HaWoR.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from ..schema import MANOSequence
from .adapter import SubprocessStage
from .base import ClipContext
from .registry import register

log = logging.getLogger("aoe")


def load_hawor_export(clip_dir: Path) -> dict:
    """Parse a HaWoR export (written by scripts/hawor_export.py) into repo
    conventions. Pure + GPU-free, so it is unit-testable anywhere.

    Reads ``hands/hawor_raw.npz`` + ``trajectory.tum``; writes
    ``hands/joints_world.npy``/``joints_2d.npy`` and ``hands/mano.npz``; returns
    the blackboard entries.

    HaWoR's MANO wrapper already returns joints in MediaPipe-21 order (its
    ``joint_map``) — the repo's storage convention — so joints are copied verbatim,
    NOT remapped (a remap here would scramble them).
    """
    from ..viz import load_tum_poses

    hands_dir = clip_dir / "hands"
    raw = np.load(hands_dir / "hawor_raw.npz")

    joints_world = raw["joints_world"].astype(np.float32)  # (T,2,21,3) MediaPipe-21
    np.save(hands_dir / "joints_world.npy", joints_world)

    joints_2d = None
    if "joints_2d" in raw.files:
        joints_2d = raw["joints_2d"].astype(np.float32)
        np.save(hands_dir / "joints_2d.npy", joints_2d)

    mano = MANOSequence(
        trans=raw["trans"], global_orient=raw["global_orient"],
        hand_pose=raw["hand_pose"], betas=raw["betas"],
        valid=raw["valid"].astype(bool),
    )
    mano.save(hands_dir / "mano.npz")

    poses = load_tum_poses(clip_dir / "trajectory.tum")
    out = {"joints_world": joints_world, "poses": list(poses), "mano": mano}
    if joints_2d is not None:
        out["joints_2d"] = joints_2d
    return out


@register("hands_hawor")
class HandStageHaWoR(SubprocessStage):
    timeout_s = 7200  # SLAM + infiller on long clips

    def available(self, ctx: ClipContext) -> str | None:
        hawor_dir = self.params.get("hawor_dir")
        if not hawor_dir:
            return "hands_hawor.params.hawor_dir not configured"
        hawor_dir = Path(hawor_dir)
        if not hawor_dir.exists():
            return f"HaWoR repo not found at {hawor_dir}"
        # checkpoints must be downloaded, else the subprocess hard-errors mid-run
        for w in ("weights/hawor/checkpoints/hawor.ckpt", "weights/hawor/checkpoints/infiller.pt"):
            if not (hawor_dir / w).exists():
                return f"HaWoR weight missing: {hawor_dir / w} (run envs/hawor.sh)"
        if not (shutil.which("conda") or shutil.which("mamba")):
            return "conda/mamba not on PATH"
        mano_dir = self.params.get("mano_dir")
        if mano_dir and not Path(mano_dir).exists():
            return f"MANO models not found at {mano_dir} (license-gated download)"
        return None

    def command(self, ctx: ClipContext) -> list[str]:
        env = self.params.get("conda_env", "hawor")
        script = self.params.get(
            "export_script",
            str(Path(__file__).resolve().parents[3] / "scripts" / "hawor_export.py"),
        )
        argv = ["python", script,
                "--video", str(ctx.video_path.resolve()),
                "--out", str(ctx.clip_dir.resolve()),
                "--hawor-dir", str(Path(self.params["hawor_dir"]).resolve())]
        if self.params.get("img_focal"):
            argv += ["--img-focal", str(self.params["img_focal"])]
        return self.conda_run(env, *argv)

    def load_outputs(self, ctx: ClipContext) -> dict:
        entries = load_hawor_export(ctx.clip_dir)
        ctx.blackboard.update(entries)
        mano: MANOSequence = entries["mano"]
        n_valid = int(mano.valid.any(axis=0).sum())
        log.info("hands_hawor: %d frames, %d with a valid hand (metric world)",
                 mano.num_frames, n_valid)
        return {"frames": mano.num_frames, "frames_with_hand": n_valid,
                "metric": True, "model": "HaWoR+MANO"}
