"""Small shared helpers."""

from __future__ import annotations

import functools
import logging

log = logging.getLogger("aoe")


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    """Best available torch device on this machine ('mps' on Apple Silicon)."""
    try:
        import torch
    except Exception:  # torch not installed yet
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
