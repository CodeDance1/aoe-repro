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


_LABEL_PROMPT = (
    "You are labeling ONE atomic hand action in an egocentric (first-person) desk video. "
    "The {n} frames are the start/middle/end of a short (<1s) segment. Identify the single "
    "atomic verb-object action. Return ONLY JSON: {{\"label_zh\": \"2-6 char Chinese verb-object\", "
    "\"label_en\": \"short english verb-object phrase\", \"hand\": \"left|right|both|none\", "
    "\"object\": \"the manipulated object\", \"confidence\": 0.0-1.0}}."
)
_LABEL_KEYS = ("label_zh", "label_en", "hand", "object", "confidence")

_VERIFY_PROMPT = (
    "Adversarially verify an action label for ONE atomic hand action in an egocentric desk "
    "video. Proposed: label_zh=\"{label_zh}\", label_en=\"{label_en}\", hand={hand}, "
    "object={object}. Independently look at the {n} frames (start/middle/end) and try to REFUTE "
    "it: right hand? right object? right verb (direction of motion matters — reach vs grasp vs "
    "release)? Return ONLY JSON: {{\"agree\": true|false, \"label_zh\": \"...\", \"label_en\": "
    "\"...\", \"hand\": \"left|right|both|none\", \"reason\": \"...\"}}. If correct, return it "
    "unchanged with agree=true; if wrong, return the corrected label with agree=false."
)


def label_segments(frames, segments, fps: float, provider: str = "openai",
                   model: str | None = None, num_keyframes: int = 3, verify: int = 2,
                   **_) -> list[dict]:
    """Attach semantic labels (label_zh/label_en/hand/object) to each interaction
    segment via a hosted vision model, with optional adversarial verify voting."""
    out = []
    for s in segments:
        s = dict(s)
        if s["label"].startswith("interaction"):
            kf = _segment_keyframes(frames, s, num_keyframes)
            if kf:
                s.update(label_one(kf, provider, model, verify))
        out.append(s)
    return out


def _segment_keyframes(frames, seg, num_keyframes):
    a, b = int(seg["start_frame"]), int(seg["end_frame"])
    idxs = np.linspace(a, max(a, b - 1), min(num_keyframes, max(1, b - a))).round().astype(int)
    return [frames[i] for i in idxs if 0 <= i < len(frames)]


def label_one(keyframes, provider: str = "openai", model: str | None = None, verify: int = 2) -> dict:
    """Label one segment, then run ``verify`` adversarial votes; adopt a corrected
    label when the majority of verifiers refute the initial one."""
    initial = _parse_label(_query_label(keyframes, provider, model))
    if verify <= 0 or not initial:
        return initial
    verdicts = [v for v in
                (_parse_verify(_query_verify(keyframes, initial, provider, model)) for _ in range(verify))
                if v]
    disagree = [v for v in verdicts if v.get("agree") is False]
    if verdicts and len(disagree) > len(verdicts) / 2:
        from collections import Counter

        best_en = Counter(v.get("label_en", "") for v in disagree).most_common(1)[0][0]
        corr = next(v for v in disagree if v.get("label_en", "") == best_en)
        return {
            "label_zh": corr.get("label_zh", initial.get("label_zh")),
            "label_en": corr.get("label_en", initial.get("label_en")),
            "hand": corr.get("hand", initial.get("hand")),
            "object": initial.get("object"),
            "confidence": round(min(float(initial.get("confidence", 0.7)), 0.6), 3),
        }
    return initial


def _query_label(keyframes, provider: str, model: str | None) -> str:
    return _vision(keyframes, _LABEL_PROMPT.format(n=len(keyframes)), provider, model)


def _query_verify(keyframes, proposed: dict, provider: str, model: str | None) -> str:
    prompt = _VERIFY_PROMPT.format(
        n=len(keyframes), label_zh=proposed.get("label_zh", ""), label_en=proposed.get("label_en", ""),
        hand=proposed.get("hand", ""), object=proposed.get("object", ""))
    return _vision(keyframes, prompt, provider, model)


def _vision(keyframes, prompt: str, provider: str, model: str | None) -> str:
    if provider == "anthropic":
        return _anthropic_vision(keyframes, prompt, model or "claude-sonnet-4-6")
    if provider == "openai":
        return _openai_vision(keyframes, prompt, model or "gpt-4o-mini")
    raise RuntimeError(f"unknown VLM provider: {provider}")


def _openai_vision(keyframes, prompt: str, model: str) -> str:
    from openai import OpenAI

    content = [{"type": "text", "text": prompt}]
    for f in keyframes:
        ok, buf = cv2.imencode(".jpg", f)
        if ok:
            b64 = base64.standard_b64encode(buf.tobytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    resp = OpenAI().chat.completions.create(
        model=model, max_tokens=300, messages=[{"role": "user", "content": content}]
    )
    return resp.choices[0].message.content


def _anthropic_vision(keyframes, prompt: str, model: str) -> str:
    import anthropic

    content = [{"type": "text", "text": prompt}]
    for f in keyframes:
        ok, buf = cv2.imencode(".jpg", f)
        if ok:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                            "data": base64.standard_b64encode(buf.tobytes()).decode()}})
    msg = anthropic.Anthropic().messages.create(
        model=model, max_tokens=300, messages=[{"role": "user", "content": content}]
    )
    return msg.content[0].text


def _parse_label(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}")
    d = json.loads(raw[s : e + 1]) if (s >= 0 and e > s) else {}
    return {k: d[k] for k in _LABEL_KEYS if k in d}


def _parse_verify(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}")
    d = json.loads(raw[s : e + 1]) if (s >= 0 and e > s) else {}
    return {k: d[k] for k in ("agree", "label_zh", "label_en", "hand", "reason") if k in d}


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
