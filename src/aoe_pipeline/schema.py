"""Per-clip output schema.

A processed clip lives in ``output/<clip_id>/`` with:

    manifest.json          # this structure (intrinsics, fps, per-stage status)
    frames/frame_%06d.png  # decoded RGB frames
    depth/depth_%06d.npy   # monocular depth (float32, HxW)
    trajectory.tum         # camera poses, TUM format (consumable by `evo`)
    hands/joints_world.npy # (T, H, 21, 3) world-coordinate 3D joints
    hands/joints_2d.npy    # (T, H, 21, 2) image-pixel 2D joints
    hands/mano.npz         # faithful profile: MANO params (see MANOSequence)
    segments.json          # atomic-action segments
    qc_report.json         # quality-control flags
    viz/                   # optional overlays / plots

where T = #frames and H = #hands tracked (padded with NaN when absent).
Joint storage convention is MediaPipe-21 order (see eval/joint_maps.py).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    source: str = "fov_default"  # metadata | checkerboard | fov_default

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_fov(
        cls, width: int, height: int, hfov_deg: float = 70.0, source: str = "fov_default"
    ) -> "CameraIntrinsics":
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height,
                   distortion=[0.0] * 5, source=source)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        return cls(**d)


@dataclass
class MANOSequence:
    """Per-clip MANO parameters from a faithful hand stage (e.g. HaWoR).

    Hand-slot axis matches the rest of the repo: 0 = Left, 1 = Right.
    Saved as ``hands/mano.npz``. Frames where a hand is absent have
    ``valid=False`` (param rows are zeros there).
    """

    trans: np.ndarray         # (2, T, 3)   world translation
    global_orient: np.ndarray  # (2, T, 3, 3) world rotation matrices
    hand_pose: np.ndarray     # (2, T, 45)  articulation (15 joints x 3, axis-angle)
    betas: np.ndarray         # (2, T, 10)  shape
    valid: np.ndarray         # (2, T)      bool

    @property
    def num_frames(self) -> int:
        return int(self.trans.shape[1])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, trans=self.trans, global_orient=self.global_orient,
            hand_pose=self.hand_pose, betas=self.betas, valid=self.valid,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MANOSequence":
        d = np.load(path)
        return cls(trans=d["trans"], global_orient=d["global_orient"],
                   hand_pose=d["hand_pose"], betas=d["betas"], valid=d["valid"])


@dataclass
class StageStatus:
    name: str
    status: str = "pending"  # pending | ok | skipped | error
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StageStatus":
        return cls(name=d["name"], status=d.get("status", "pending"), info=d.get("info", {}))


@dataclass
class ClipManifest:
    clip_id: str
    video_path: str
    fps: float = 0.0
    num_frames: int = 0
    width: int = 0
    height: int = 0
    intrinsics: CameraIntrinsics | None = None
    stages: dict[str, StageStatus] = field(default_factory=dict)

    def set_stage(self, name: str, status: str, **info) -> None:
        self.stages[name] = StageStatus(name=name, status=status, info=info)

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClipManifest":
        intr = d.get("intrinsics")
        return cls(
            clip_id=d["clip_id"],
            video_path=d["video_path"],
            fps=d.get("fps", 0.0),
            num_frames=d.get("num_frames", 0),
            width=d.get("width", 0),
            height=d.get("height", 0),
            intrinsics=CameraIntrinsics.from_dict(intr) if intr else None,
            stages={k: StageStatus.from_dict(v) for k, v in d.get("stages", {}).items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ClipManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))
