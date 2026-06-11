"""Subprocess adapter for faithful (heavy, own-environment) model stages.

The paper's real models need mutually conflicting environments (e.g. HaWoR is
py3.10/torch1.13/cu117 vs our base py3.12/torch2.12), so a faithful stage runs its
model **in its own conda/uv env via subprocess**, handing data off through the clip
directory. Subclasses declare how to build the command and how to load the
artifacts the external process wrote.

A faithful stage is *gated*: if its env/checkpoints aren't configured (params or
the configured paths don't exist), it records ``skipped`` and the lite pipeline
remains the source of truth — keeping CI and Mac runs green.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from .base import ClipContext, Stage

log = logging.getLogger("aoe")


class SubprocessStage(Stage):
    """Base for stages that shell out to a model living in its own environment.

    Subclasses implement:
      - ``available(ctx) -> str | None``: None if runnable, else a skip reason.
      - ``command(ctx) -> list[str]``: the full argv to execute.
      - ``load_outputs(ctx) -> dict``: parse artifacts written by the subprocess
        into blackboard entries; the returned dict is merged into the manifest
        stage info.
    """

    timeout_s: int = 3600

    def run(self, ctx: ClipContext) -> None:
        reason = self.available(ctx)
        if reason:
            ctx.manifest.set_stage(self.name, "skipped", reason=reason)
            log.info("%s: skipped (%s)", self.name, reason)
            return

        cmd = self.command(ctx)
        log.info("%s: exec %s", self.name, " ".join(map(str, cmd)))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            raise RuntimeError(
                f"{self.name} subprocess failed (exit {proc.returncode}):\n" + "\n".join(tail)
            )

        info = self.load_outputs(ctx)
        ctx.manifest.set_stage(self.name, "ok", **info)

    # --- hooks -------------------------------------------------------------
    def available(self, ctx: ClipContext) -> str | None:
        raise NotImplementedError

    def command(self, ctx: ClipContext) -> list[str]:
        raise NotImplementedError

    def load_outputs(self, ctx: ClipContext) -> dict:
        raise NotImplementedError

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def conda_run(env: str, *argv: str) -> list[str]:
        """argv prefix to run inside a named conda env (if conda is present)."""
        conda = shutil.which("conda") or shutil.which("mamba")
        if not conda:
            return list(argv)  # caller's available() should have gated this
        return [conda, "run", "--no-capture-output", "-n", env, *argv]
