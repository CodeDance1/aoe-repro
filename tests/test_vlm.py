from __future__ import annotations

import numpy as np

from aoe_pipeline import vlm
from aoe_pipeline.vlm import _parse_vlm_json, _to_segments, select_keyframes


def test_adaptive_keyframe_count_scales_with_duration():
    short = select_keyframes(30, fps=10.0, seconds_per_keyframe=2.0, min_keyframes=4, max_keyframes=16)
    long = select_keyframes(300, fps=10.0, seconds_per_keyframe=2.0, min_keyframes=4, max_keyframes=16)

    assert len(short) == 4
    assert len(long) == 16
    assert short[0] == 0 and short[-1] == 29
    assert long[0] == 0 and long[-1] == 299


def test_parse_paper_atomic_actions_to_segments():
    raw = """
    ```json
    {
      "atomic_actions": [
        {
          "start_time": 0.0,
          "end_time": 1.5,
          "verb": "hold",
          "object": "cup",
          "description": "right hand holds cup",
          "bbox": [10, 20, 30, 40],
          "confidence": 0.9
        }
      ]
    }
    ```
    """

    parsed = _parse_vlm_json(raw)
    segments = _to_segments(parsed, T=60, fps=10.0)

    assert parsed["atomic_actions"][0]["verb"] == "hold"
    assert segments == [
        {
            "label": "interaction_0:hold cup",
            "start_frame": 0,
            "end_frame": 15,
            "start_time": 0.0,
            "end_time": 1.5,
            "verb": "hold",
            "object": "cup",
            "description": "right hand holds cup",
            "bbox": [10, 20, 30, 40],
            "confidence": 0.9,
        }
    ]


def test_local_openai_path_returns_metadata(monkeypatch):
    calls = {}

    def fake_query(keyframes, idxs, fps, dur, model, **kwargs):
        calls["keyframe_count"] = len(keyframes)
        calls["idxs"] = [int(i) for i in idxs]
        calls["model"] = model
        return """
        {"atomic_actions": [
          {"start_time": 0.0, "end_time": 0.5, "verb": "pick", "object": "block",
           "description": "hand picks block", "bbox": null, "confidence": 0.8}
        ]}
        """

    monkeypatch.setattr(vlm, "_query_local_openai", fake_query)
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(20)]

    segments, meta = vlm.segment_with_vlm(
        frames,
        fps=10.0,
        provider="local_openai",
        model="local-vlm",
        keyframe_strategy="adaptive",
        min_keyframes=4,
        max_keyframes=8,
        seconds_per_keyframe=1.0,
        return_metadata=True,
    )

    assert calls == {"keyframe_count": 4, "idxs": [0, 6, 13, 19], "model": "local-vlm"}
    assert meta["provider"] == "local_openai"
    assert meta["keyframe_count"] == 4
    assert meta["atomic_actions"][0]["verb"] == "pick"
    assert segments[0]["label"] == "interaction_0:pick block"
