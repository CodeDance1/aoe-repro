"""Optional VLM segmentation backend (substitute hook for Qwen3-VL-235B).

Samples keyframes and asks a hosted vision model to return atomic-action
boundaries as JSON. Only used when ``segment.backend: vlm``. Requires the ``vlm``
extra (``pip install -e '.[vlm]'``) and an API key in the environment.
"""

from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np

log = logging.getLogger("aoe")

_PROMPT = (
    "You are segmenting an egocentric (first-person) video into atomic action "
    "clips. The {n} frames are evenly spaced from t=0s to t={dur:.2f}s. Return ONLY "
    "a JSON array; each item: {{\"start_time\": float, \"end_time\": float, "
    "\"label\": \"short verb-object phrase\"}}. Cover the whole timeline in order."
)


def segment_with_vlm(frames, fps: float, provider: str = "anthropic",
                     model: str | None = None, num_keyframes: int = 8, **_) -> list[dict]:
    T = len(frames)
    if T == 0:
        return []
    idxs = np.linspace(0, T - 1, min(num_keyframes, T)).round().astype(int)
    dur = T / fps
    if provider == "anthropic":
        raw = _query_anthropic([frames[i] for i in idxs], dur, model)
    else:
        raise RuntimeError(f"unknown VLM provider: {provider}")
    return _to_segments(raw, T, fps)


def _query_anthropic(keyframes, dur: float, model: str | None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    content = [{"type": "text", "text": _PROMPT.format(n=len(keyframes), dur=dur)}]
    for f in keyframes:
        ok, buf = cv2.imencode(".jpg", f)
        if not ok:
            continue
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.standard_b64encode(buf.tobytes()).decode()},
        })
    msg = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text


def _to_segments(raw: str, T: int, fps: float) -> list[dict]:
    start = raw.find("[")
    end = raw.rfind("]")
    items = json.loads(raw[start : end + 1]) if start >= 0 and end > start else []
    segments = []
    for k, it in enumerate(items):
        a = int(round(float(it["start_time"]) * fps))
        b = int(round(float(it["end_time"]) * fps))
        a, b = max(0, min(a, T)), max(0, min(b, T))
        if b <= a:
            continue
        segments.append({
            "label": f"interaction_{k}:{it.get('label', '').strip()}",
            "start_frame": a, "end_frame": b,
            "start_time": round(a / fps, 3), "end_time": round(b / fps, 3),
        })
    return segments
