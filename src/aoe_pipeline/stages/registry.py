"""Decorator-based stage registry.

Usage::

    @register("hands")
    class HandStage(Stage):
        ...

    cls = get_stage("hands")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycle at runtime
    from .base import Stage

_REGISTRY: dict[str, type["Stage"]] = {}


def register(name: str):
    def deco(cls: type["Stage"]) -> type["Stage"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_stage(name: str) -> type["Stage"]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown stage '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
