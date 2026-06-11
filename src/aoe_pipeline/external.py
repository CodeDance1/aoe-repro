"""Helpers for calling paper-faithful external model pipelines.

The official models used by AoE often live in their own CUDA/conda
environments. These helpers keep this lightweight repo as the orchestrator:
prepare inputs, run a configured command, then import the produced artifacts.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from string import Formatter
from typing import Any


class ExternalPipelineError(RuntimeError):
    """Raised when an external model adapter is missing or returns bad outputs."""


class _StrictFormatDict(dict):
    def __missing__(self, key: str) -> str:
        raise ExternalPipelineError(f"external command references unknown placeholder: {key}")


def render_command(command: str | list[str], values: dict[str, Any]) -> str | list[str]:
    """Format placeholders in a command string/list with path-safe values."""
    rendered = []
    fmt_values = _StrictFormatDict({k: str(v) for k, v in values.items()})
    if isinstance(command, list):
        for part in command:
            rendered.append(_format_part(str(part), fmt_values))
        return rendered
    return _format_part(command, fmt_values)


def run_external(
    command: str | list[str] | None,
    values: dict[str, Any],
    cwd: str | Path | None = None,
    shell: bool = False,
) -> None:
    """Run an external command after formatting placeholders.

    Command placeholders commonly used by stages:
    ``{video_path}``, ``{frames_dir}``, ``{clip_dir}``, ``{external_dir}``,
    ``{intrinsics_json}``, and ``{fps}``.
    """
    if not command:
        raise ExternalPipelineError(
            "paper-faithful backend requires an external command. Configure "
            "`command` in the stage params to call the official repo/environment."
        )
    rendered = render_command(command, values)
    if isinstance(rendered, str) and not shell:
        args = shlex.split(rendered)
    else:
        args = rendered
    subprocess.run(args, cwd=str(cwd) if cwd else None, shell=shell, check=True)  # noqa: S603


def require_path(path: str | Path, desc: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise ExternalPipelineError(f"expected {desc} at {p}, but it does not exist")
    return p


def _format_part(text: str, values: dict[str, str]) -> str:
    # Preflight placeholders so errors are clearer than KeyError.
    for _, field_name, _, _ in Formatter().parse(text):
        if field_name and field_name not in values:
            raise ExternalPipelineError(f"external command references unknown placeholder: {field_name}")
    return text.format_map(values)
