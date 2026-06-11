"""Pipeline orchestrator: run ordered stages over one clip, persist a manifest."""

from __future__ import annotations

import logging
from pathlib import Path

from . import stages as _stages  # noqa: F401  (ensures stages are registered)
from .config import PipelineConfig
from .schema import ClipManifest
from .stages.base import ClipContext
from .stages.registry import get_stage

log = logging.getLogger("aoe")


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        clip_id: str | None = None,
        only: set[str] | None = None,
    ) -> ClipContext:
        video_path = Path(video_path)
        clip_id = clip_id or video_path.stem
        clip_dir = Path(output_dir) / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = clip_dir / "manifest.json"

        if manifest_path.exists():
            manifest = ClipManifest.load(manifest_path)
        else:
            manifest = ClipManifest(clip_id=clip_id, video_path=str(video_path))

        ctx = ClipContext(
            clip_id=clip_id,
            clip_dir=clip_dir,
            video_path=video_path,
            config=self.config,
            manifest=manifest,
        )

        for name in self.config.pipeline:
            if only and name not in only:
                continue
            cfg = self.config.stage_cfg(name)
            if not cfg.enabled:
                manifest.set_stage(name, "skipped")
                manifest.save(manifest_path)
                log.info("stage %s: skipped (disabled)", name)
                continue

            stage = get_stage(name)(cfg.params)
            log.info("stage %s: running", name)
            try:
                stage.run(ctx)
            except Exception as exc:  # noqa: BLE001 — record and re-raise
                manifest.set_stage(name, "error", error=str(exc))
                manifest.save(manifest_path)
                log.exception("stage %s: error", name)
                raise
            # Stages normally set their own 'ok' status; backfill if they didn't.
            st = manifest.stages.get(name)
            if st is None or st.status == "pending":
                manifest.set_stage(name, "ok")
            manifest.save(manifest_path)

        return ctx
