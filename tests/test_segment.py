"""Tests for the adaptive motion threshold in the segmentation stage."""

from __future__ import annotations

import numpy as np

from aoe_pipeline.stages.s2_segment import _resolve_threshold


def test_explicit_number_used_as_is():
    motion = np.array([0.1, 0.5, 1.0, 2.0])
    present = np.ones(4, bool)
    assert _resolve_threshold(motion, present, {"motion_threshold": 1.5}) == 1.5


def test_auto_is_percentile_of_clip_motion():
    motion = np.arange(1, 101, dtype=float)  # 1..100
    present = np.ones(100, bool)
    thr = _resolve_threshold(motion, present, {"motion_threshold": "auto", "motion_percentile": 65})
    assert np.isclose(thr, np.percentile(motion, 65))


def test_auto_uses_present_frames_and_ignores_zeros():
    motion = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    present = np.array([True, True, True, True, False, False])
    # present & >0 -> [1.0, 2.0]; p50 = 1.5
    thr = _resolve_threshold(motion, present, {"motion_threshold": "auto", "motion_percentile": 50})
    assert np.isclose(thr, 1.5, atol=1e-6)


def test_auto_default_percentile_is_65():
    motion = np.linspace(0.1, 1.0, 50)
    present = np.ones(50, bool)
    thr = _resolve_threshold(motion, present, {"motion_threshold": "auto"})
    assert np.isclose(thr, np.percentile(motion, 65))


def test_none_threshold_treated_as_auto():
    motion = np.linspace(0.1, 1.0, 20)
    present = np.ones(20, bool)
    thr = _resolve_threshold(motion, present, {"motion_threshold": None, "motion_percentile": 70})
    assert np.isclose(thr, np.percentile(motion, 70))
