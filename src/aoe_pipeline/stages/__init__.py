"""Importing this package registers all built-in stages."""

from __future__ import annotations

from . import (  # noqa: F401  (imported for registration side effects)
    s1_ingest,
    s2_segment,
    s2b_label,
    s3_trajectory,
    s4_hands,
    s5_augment,
    s6_qc,
)
from .base import ClipContext, Stage
from .registry import available, get_stage, register

__all__ = ["ClipContext", "Stage", "available", "get_stage", "register"]
