"""Optional VLM segmentation backend (substitute hook for Qwen3-VL-235B).

Samples an adaptive number of keyframes and asks a vision-language model to
return paper-style ``atomic_actions`` JSON. Hosted Anthropic, local
OpenAI-compatible servers (vLLM, LM Studio, llama.cpp server), and Ollama are
supported so the cloud-stage substitute can run against a local VLM.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request

import cv2
import numpy as np

log = logging.getLogger("aoe")

_PROMPT_VERSION = "paper_atomic_actions_v1"
_PROMPT = """You are the cloud atomic-action segmentation operator for an
egocentric human-manipulation data pipeline. The {n} frames are selected from a
{dur:.2f}s first-person video; each image is labeled with its timestamp in the
message text.

Return ONLY valid JSON with this schema:
{{
  "atomic_actions": [
    {{
      "start_time": 0.0,
      "end_time": 1.2,
      "verb": "hold",
      "object": "cup",
      "description": "right hand holds a cup",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    }}
  ]
}}

Guidelines:
- Segment semantic atomic hand-object interactions, not low-level optical-flow changes.
- Cover the meaningful interaction timeline in chronological order.
- Use seconds for times and image pixels for bbox. If bbox is uncertain, use null.
- Prefer concise verb/object labels; use "unknown" only when necessary.
"""


def segment_with_vlm(
    frames,
    fps: float,
    provider: str = "local_openai",
    model: str | None = None,
    num_keyframes: int = 8,
    keyframe_strategy: str = "adaptive",
    min_keyframes: int = 8,
    max_keyframes: int = 32,
    seconds_per_keyframe: float = 2.0,
    return_metadata: bool = False,
    **kwargs,
):
    T = len(frames)
    if T == 0:
        empty = ([], {"keyframes": [], "atomic_actions": []})
        return empty if return_metadata else []

    idxs = select_keyframes(
        T,
        fps,
        strategy=keyframe_strategy,
        num_keyframes=num_keyframes,
        min_keyframes=min_keyframes,
        max_keyframes=max_keyframes,
        seconds_per_keyframe=seconds_per_keyframe,
    )
    dur = T / fps
    keyframes = [frames[i] for i in idxs]
    effective_model = model
    if provider == "anthropic":
        effective_model = model or "claude-sonnet-4-6"
        raw = _query_anthropic(keyframes, idxs, fps, dur, model)
    elif provider in {"local", "local_openai", "openai_compatible"}:
        effective_model = model or os.environ.get("AOE_LOCAL_VLM_MODEL")
        raw = _query_local_openai(keyframes, idxs, fps, dur, model, **kwargs)
    elif provider == "ollama":
        effective_model = model or os.environ.get("AOE_LOCAL_VLM_MODEL") or "llava"
        raw = _query_ollama(keyframes, idxs, fps, dur, model, **kwargs)
    else:
        raise RuntimeError(f"unknown VLM provider: {provider}")

    parsed = _parse_vlm_json(raw)
    segments = _to_segments(parsed, T, fps)
    meta = {
        "provider": provider,
        "model": effective_model,
        "prompt_version": _PROMPT_VERSION,
        "keyframe_strategy": keyframe_strategy,
        "keyframes": [int(i) for i in idxs],
        "keyframe_count": int(len(idxs)),
        "atomic_actions": parsed.get("atomic_actions", []),
    }
    return (segments, meta) if return_metadata else segments


def select_keyframes(
    total_frames: int,
    fps: float,
    strategy: str = "adaptive",
    num_keyframes: int = 8,
    min_keyframes: int = 8,
    max_keyframes: int = 32,
    seconds_per_keyframe: float = 2.0,
) -> np.ndarray:
    """Paper-cloud style frame-count selection for VLM segmentation.

    ``adaptive`` scales the number of VLM images with clip duration, bounded by
    ``min_keyframes``/``max_keyframes``. ``fixed`` preserves the previous
    behavior.
    """
    if total_frames <= 0:
        return np.array([], dtype=int)
    if strategy == "fixed":
        count = int(num_keyframes)
    elif strategy == "adaptive":
        dur = total_frames / max(float(fps), 1e-6)
        count = int(math.ceil(dur / max(float(seconds_per_keyframe), 1e-6))) + 1
        count = max(int(min_keyframes), min(int(max_keyframes), count))
    else:
        raise ValueError(f"unknown keyframe_strategy: {strategy}")
    count = max(1, min(int(count), total_frames))
    return np.linspace(0, total_frames - 1, count).round().astype(int)


def _query_anthropic(keyframes, idxs, fps: float, dur: float, model: str | None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    content = [{"type": "text", "text": _prompt_with_timestamps(idxs, fps, dur)}]
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


def _query_local_openai(
    keyframes,
    idxs,
    fps: float,
    dur: float,
    model: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_tokens: int = 2048,
    **_,
) -> str:
    base_url = (base_url or os.environ.get("AOE_LOCAL_VLM_BASE_URL")
                or "http://localhost:8000/v1").rstrip("/")
    model = model or os.environ.get("AOE_LOCAL_VLM_MODEL")
    if not model:
        raise RuntimeError("local VLM model is required (set segment.vlm.model or AOE_LOCAL_VLM_MODEL)")
    api_key = api_key or os.environ.get("AOE_LOCAL_VLM_API_KEY") or "local"

    content = [{"type": "text", "text": _prompt_with_timestamps(idxs, fps, dur)}]
    for frame in keyframes:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_jpg_b64(frame)}"},
        })
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }
    data = _post_json(
        f"{base_url}/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    return data["choices"][0]["message"]["content"]


def _query_ollama(
    keyframes,
    idxs,
    fps: float,
    dur: float,
    model: str | None,
    base_url: str | None = None,
    timeout: float = 120.0,
    **_,
) -> str:
    base_url = (base_url or os.environ.get("AOE_OLLAMA_BASE_URL")
                or "http://localhost:11434").rstrip("/")
    model = model or os.environ.get("AOE_LOCAL_VLM_MODEL") or "llava"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": _prompt_with_timestamps(idxs, fps, dur),
            "images": [_jpg_b64(frame) for frame in keyframes],
        }],
        "options": {"temperature": 0},
    }
    data = _post_json(f"{base_url}/api/chat", payload, timeout=timeout)
    return data["message"]["content"]


def _prompt_with_timestamps(idxs, fps: float, dur: float) -> str:
    stamps = ", ".join(f"frame {int(i)} @ {i / fps:.2f}s" for i in idxs)
    return _PROMPT.format(n=len(idxs), dur=dur) + "\nSelected frames: " + stamps


def _jpg_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("failed to encode VLM keyframe as JPEG")
    return base64.standard_b64encode(buf.tobytes()).decode()


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 local/user-configured URL
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"local VLM request failed: {exc}") from exc


def _parse_vlm_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start_obj, end_obj = text.find("{"), text.rfind("}")
        if start_obj >= 0 and end_obj > start_obj:
            obj = json.loads(text[start_obj : end_obj + 1])
        else:
            start_arr, end_arr = text.find("["), text.rfind("]")
            if start_arr < 0 or end_arr <= start_arr:
                raise
            obj = {"atomic_actions": json.loads(text[start_arr : end_arr + 1])}
    if isinstance(obj, list):
        obj = {"atomic_actions": obj}
    if "atomic_actions" not in obj:
        raise ValueError("VLM response did not contain atomic_actions")
    return obj


def _to_segments(parsed: dict, T: int, fps: float) -> list[dict]:
    items = parsed.get("atomic_actions", [])
    segments = []
    for k, it in enumerate(items):
        a = int(round(float(it["start_time"]) * fps))
        b = int(round(float(it["end_time"]) * fps))
        a, b = max(0, min(a, T)), max(0, min(b, T))
        if b <= a:
            continue
        verb = str(it.get("verb") or "").strip()
        obj = str(it.get("object") or "").strip()
        phrase = " ".join(x for x in [verb, obj] if x) or str(it.get("label", "")).strip()
        segments.append({
            "label": f"interaction_{k}:{phrase}",
            "start_frame": a, "end_frame": b,
            "start_time": round(a / fps, 3), "end_time": round(b / fps, 3),
            "verb": verb or None,
            "object": obj or None,
            "description": it.get("description"),
            "bbox": it.get("bbox"),
            "confidence": it.get("confidence"),
        })
    return segments
