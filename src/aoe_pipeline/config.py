"""Pipeline configuration (YAML <-> pydantic)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class StageCfg(BaseModel):
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Ordered list of stages to run plus per-stage params.

    A stage named in ``pipeline`` but with ``enabled: false`` is recorded as
    skipped rather than executed.
    """

    pipeline: list[str] = Field(default_factory=list)
    stages: dict[str, StageCfg] = Field(default_factory=dict)

    def stage_cfg(self, name: str) -> StageCfg:
        return self.stages.get(name, StageCfg())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Built-in default identical to configs/default.yaml."""
        return cls(
            pipeline=["ingest", "trajectory", "hands", "segment", "qc", "augment"],
            stages={
                "ingest": StageCfg(params={"stride": 1, "max_frames": 0, "hfov_deg": 70.0}),
                "trajectory": StageCfg(
                    params={
                        "depth_model": "depth-anything/Depth-Anything-V2-Small-hf",
                        "orb_features": 1500,
                        "scale_from_depth": True,
                    }
                ),
                "hands": StageCfg(
                    params={"max_hands": 2, "smooth_window": 5, "smooth_poly": 3,
                            "depth_anchor": "wrist"}
                ),
                "segment": StageCfg(
                    params={
                        "backend": "heuristic",
                        "motion_threshold": "auto",
                        "motion_percentile": 65,
                        "min_segment_frames": 8,
                    }
                ),
                "qc": StageCfg(params={"velocity_sigma": 3.0, "reproj_px": 5.0}),
                "augment": StageCfg(enabled=False, params={"background": None, "inpaint_hands": False}),
            },
        )
