"""Stage base class and the per-clip execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import PipelineConfig
from ..schema import ClipManifest


@dataclass
class ClipContext:
    """Carries everything a stage needs for one clip.

    ``blackboard`` is in-memory shared state for a single pipeline run (decoded
    frames, poses, hand joints, ...). Stages should also persist their outputs
    to ``clip_dir`` so later runs / the eval & viz commands can read them.
    """

    clip_id: str
    clip_dir: Path
    video_path: Path
    config: PipelineConfig
    manifest: ClipManifest
    blackboard: dict = field(default_factory=dict)

    @property
    def frames_dir(self) -> Path:
        return self.clip_dir / "frames"

    @property
    def depth_dir(self) -> Path:
        return self.clip_dir / "depth"

    @property
    def hands_dir(self) -> Path:
        return self.clip_dir / "hands"

    @property
    def viz_dir(self) -> Path:
        return self.clip_dir / "viz"

    def get_frames(self) -> list[np.ndarray]:
        """Frames as BGR uint8 arrays, from the blackboard or decoded PNGs."""
        if "frames" in self.blackboard:
            return self.blackboard["frames"]
        files = sorted(self.frames_dir.glob("frame_*.png"))
        frames = [cv2.imread(str(f)) for f in files]
        self.blackboard["frames"] = frames
        return frames

    @property
    def fps(self) -> float:
        return self.blackboard.get("fps") or self.manifest.fps or 30.0


class Stage(ABC):
    """A pipeline stage. Subclasses register via ``@register("name")``."""

    name: str = "stage"

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def run(self, ctx: ClipContext) -> None:
        """Read inputs from ``ctx`` (blackboard / disk), write outputs to both."""
        raise NotImplementedError
