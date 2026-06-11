"""Download & cache MediaPipe Tasks model bundles.

MediaPipe 0.10.35 removed the legacy ``mp.solutions`` API; the Tasks API needs a
model bundle file on disk. We fetch the official bundles on first use and cache
them under ``~/.cache/aoe_pipeline/``.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("aoe")

CACHE_DIR = Path.home() / ".cache" / "aoe_pipeline"

_MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "selfie_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
}


def get_model(name: str) -> Path:
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'; known: {sorted(_MODELS)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if not dest.exists() or dest.stat().st_size == 0:
        log.info("downloading mediapipe model %s ...", name)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(_MODELS[name], tmp)  # noqa: S310 (trusted host)
        tmp.replace(dest)
        log.info("cached %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def hand_landmarker_model() -> Path:
    return get_model("hand_landmarker.task")


def selfie_segmenter_model() -> Path:
    return get_model("selfie_segmenter.tflite")
