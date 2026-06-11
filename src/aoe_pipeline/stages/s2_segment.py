"""Stage 2 — atomic action segmentation.

Substitute for Qwen3-VL-235B. Default backend is a no-model heuristic: combine
optical-flow motion energy with hand presence to find hand-object interaction
windows, then split long windows at motion valleys into atomic clips. A pluggable
VLM backend (``backend: vlm``) can label/segment via a hosted model instead.
"""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np

from .base import ClipContext, Stage
from .registry import register

log = logging.getLogger("aoe")


@register("segment")
class SegmentStage(Stage):
    def run(self, ctx: ClipContext) -> None:
        frames = ctx.get_frames()
        backend = self.params.get("backend", "heuristic")
        if backend == "vlm":
            segments = self._segment_vlm(ctx, frames)
        else:
            segments = self._segment_heuristic(ctx, frames)

        (ctx.clip_dir / "segments.json").write_text(json.dumps(segments, indent=2))
        ctx.blackboard["segments"] = segments
        n_inter = sum(1 for s in segments if s["label"].startswith("interaction"))
        ctx.manifest.set_stage(self.name, "ok", backend=backend,
                               num_segments=len(segments), num_interaction=n_inter)
        log.info("segment: %d segments (%d interaction) via %s", len(segments), n_inter, backend)

    # --- heuristic backend -----------------------------------------------------
    def _segment_heuristic(self, ctx: ClipContext, frames) -> list[dict]:
        thr = float(self.params.get("motion_threshold", 1.5))
        min_len = int(self.params.get("min_segment_frames", 8))
        T = len(frames)
        if T == 0:
            return []

        motion = _motion_energy(frames)
        present = _hand_presence(ctx, T)
        active = (motion > thr) & present

        # contiguous run labels, then enforce minimum length by merging.
        labels = ["interaction" if a else "idle" for a in active]
        runs = _runs(labels)
        runs = _merge_short_runs(runs, min_len)

        segments: list[dict] = []
        k = 0
        for label, lo, hi in runs:
            if label == "interaction" and (hi - lo) >= 2 * min_len:
                for a, b in _split_at_valleys(motion[lo:hi], min_len):
                    segments.append(_seg(f"interaction_{k}", lo + a, lo + b, motion, ctx.fps))
                    k += 1
            elif label == "interaction":
                segments.append(_seg(f"interaction_{k}", lo, hi, motion, ctx.fps))
                k += 1
            else:
                segments.append(_seg("idle", lo, hi, motion, ctx.fps))
        return segments

    # --- VLM backend (pluggable) ----------------------------------------------
    def _segment_vlm(self, ctx: ClipContext, frames) -> list[dict]:
        cfg = self.params.get("vlm", {}) or {}
        provider = cfg.get("provider", "anthropic")
        try:
            from ..vlm import segment_with_vlm
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"VLM backend unavailable: {exc}") from exc
        return segment_with_vlm(frames, ctx.fps, provider=provider, **cfg)


# --- helpers -------------------------------------------------------------------
def _motion_energy(frames) -> np.ndarray:
    """Mean Farneback optical-flow magnitude per frame (motion[0] = 0)."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    motion = np.zeros(len(frames), np.float32)
    for t in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            grays[t - 1], grays[t], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion[t] = float(mag.mean())
    return motion


def _hand_presence(ctx: ClipContext, T: int) -> np.ndarray:
    """Per-frame hand presence; reuse Stage 4 output if available, else True."""
    j2d = ctx.blackboard.get("joints_2d")
    if j2d is None:
        path = ctx.hands_dir / "joints_2d.npy"
        if path.exists():
            j2d = np.load(path)
    if j2d is None:
        return np.ones(T, bool)  # no hand info -> rely on motion alone
    present = np.isfinite(j2d[..., 0]).any(axis=(1, 2))
    if len(present) != T:  # be defensive about length mismatch
        out = np.ones(T, bool)
        out[: len(present)] = present
        return out
    return present


def _runs(labels: list[str]) -> list[tuple[str, int, int]]:
    """Contiguous runs as (label, start, end_exclusive)."""
    runs = []
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((labels[i], i, j))
        i = j
    return runs


def _merge_short_runs(runs, min_len: int):
    """Relabel runs shorter than min_len into the previous run, then re-merge."""
    if not runs:
        return runs
    flat = [None] * runs[-1][2]
    for label, lo, hi in runs:
        for t in range(lo, hi):
            flat[t] = label
    for label, lo, hi in runs:
        if (hi - lo) < min_len and lo > 0:
            for t in range(lo, hi):
                flat[t] = flat[lo - 1]
    return _runs(flat)


def _split_at_valleys(motion_seg: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Split a window at motion local minima into atomic [start, end) parts."""
    from scipy.signal import find_peaks

    n = len(motion_seg)
    if n < 2 * min_len:
        return [(0, n)]
    inv = -_smooth(motion_seg)
    valleys, _ = find_peaks(inv, distance=min_len)
    bounds = [0] + [int(v) for v in valleys if min_len <= v <= n - min_len] + [n]
    bounds = sorted(set(bounds))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    kern = np.ones(k) / k
    return np.convolve(x, kern, mode="same")


def _seg(label: str, lo: int, hi: int, motion: np.ndarray, fps: float) -> dict:
    return {
        "label": label,
        "start_frame": int(lo),
        "end_frame": int(hi),
        "start_time": round(lo / fps, 3),
        "end_time": round(hi / fps, 3),
        "mean_motion": round(float(motion[lo:hi].mean()) if hi > lo else 0.0, 4),
    }
