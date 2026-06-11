"""Stage 2b — semantic action labeling of interaction segments.

Adds ``label_zh`` / ``label_en`` / ``hand`` / ``object`` to each interaction segment
so the contact-sheet and annotated banners show real action labels (e.g. "pick up
cup") instead of generic ``interaction_k``. Order of preference:

  1. merge a ``<clip_dir>/semantic_labels.json`` (or configured ``labels_file``) if
     present — an offline/cached label pass (list of {id, label_zh, label_en, ...}),
  2. else call a hosted VLM provider (``openai`` | ``anthropic``) if its package +
     API key are available,
  3. else skip (segments keep their generic ``interaction_k`` labels).
"""

from __future__ import annotations

import json
import logging
import os

from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")

_LABEL_KEYS = ("label_zh", "label_en", "hand", "object", "confidence")


@register("label")
class LabelStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        seg_path = ctx.clip_dir / "segments.json"
        segments = ctx.blackboard.get("segments")
        if segments is None:
            if not seg_path.exists():
                ctx.manifest.set_stage(self.name, "skipped", reason="no segments")
                return
            segments = json.loads(seg_path.read_text())

        if not any(s["label"].startswith("interaction") for s in segments):
            ctx.manifest.set_stage(self.name, "skipped", reason="no interaction segments")
            return

        labels_file = self.params.get("labels_file")
        labels_file = ctx.clip_dir / (labels_file or "semantic_labels.json")

        if labels_file.exists():
            segments = _merge_from_file(segments, labels_file)
            source = f"file:{labels_file.name}"
        else:
            provider = self.params.get("provider", "openai")
            if not _provider_available(provider):
                ctx.manifest.set_stage(self.name, "skipped",
                                       reason=f"no labels_file; VLM provider '{provider}' unavailable")
                log.info("label: skipped (no labels file, no usable VLM provider)")
                return
            try:
                from ..vlm import label_segments

                segments = label_segments(ctx.get_frames(), segments, ctx.fps, provider=provider,
                                          model=self.params.get("model"),
                                          num_keyframes=int(self.params.get("num_keyframes", 3)),
                                          verify=int(self.params.get("verify", 2)))
                source = f"vlm:{provider}"
            except Exception as exc:  # noqa: BLE001 — degrade, don't crash the run
                ctx.manifest.set_stage(self.name, "skipped", reason=f"VLM labeling failed: {exc}")
                log.warning("label: VLM labeling failed: %s", exc)
                return

        seg_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        ctx.blackboard["segments"] = segments
        n = sum(1 for s in segments if s.get("label_en"))
        ctx.manifest.set_stage(self.name, "ok", source=source, labeled=n)
        log.info("label: %d interaction segments labeled via %s", n, source)


def _merge_from_file(segments, path) -> list[dict]:
    labels = {item["id"]: item for item in json.loads(path.read_text())}
    out = []
    for s in segments:
        s = dict(s)
        lab = labels.get(s["label"])
        if lab:
            for k in _LABEL_KEYS:
                if k in lab:
                    s[k] = lab[k]
        out.append(s)
    return out


def _provider_available(provider: str) -> bool:
    if provider == "openai":
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False
