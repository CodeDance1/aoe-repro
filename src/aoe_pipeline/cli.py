"""Command-line interface (Typer)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from .config import PipelineConfig
from .pipeline import Pipeline
from .stages.registry import available

app = typer.Typer(add_completion=False, help="AoE labeling pipeline (Mac-runnable reproduction)")
console = Console()


def _load_config(config: Path | None) -> PipelineConfig:
    return PipelineConfig.from_yaml(config) if config else PipelineConfig.default()


@app.command()
def run(
    video: Path = typer.Option(..., exists=True, dir_okay=False, help="Input egocentric video"),
    output_dir: Path = typer.Option(Path("output"), help="Output root directory"),
    config: Path = typer.Option(None, exists=True, help="YAML config (default: built-in)"),
    clip_id: str = typer.Option(None, help="Clip id (default: video filename stem)"),
    only: str = typer.Option(None, help="Comma-separated subset of stages to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the labeling pipeline on an egocentric video."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    cfg = _load_config(config)
    only_set = {s.strip() for s in only.split(",")} if only else None
    ctx = Pipeline(cfg).run(video, output_dir, clip_id=clip_id, only=only_set)
    console.print(f"[green]✓[/green] pipeline complete → {ctx.clip_dir}")
    for name, st in ctx.manifest.stages.items():
        color = {"ok": "green", "skipped": "yellow", "error": "red"}.get(st.status, "white")
        console.print(f"  [{color}]{st.status:8}[/{color}] {name}")


@app.command("list-stages")
def list_stages() -> None:
    """List registered pipeline stages."""
    for name in available():
        console.print(f"  - {name}")


@app.command("eval-traj")
def eval_traj(
    est: Path = typer.Option(..., exists=True, help="Estimated trajectory (.tum)"),
    gt: Path = typer.Option(..., exists=True, help="Ground-truth trajectory (.tum)"),
) -> None:
    """Camera-trajectory metrics: ATE (Sim3), ATE-S (SE3, scale-free), RPE."""
    from .eval.metrics import trajectory_metrics

    m = trajectory_metrics(str(est), str(gt))
    console.print("[bold]Trajectory metrics[/bold]")
    console.print(f"  pairs           : {m['pairs']}")
    console.print(f"  ATE  (Sim3,7DoF): {m['ate_sim3_7dof'] * 1000:.2f} mm")
    console.print(f"  ATE-S (SE3,6DoF): {m['ate_s_se3_6dof'] * 1000:.2f} mm")
    console.print(f"  RPE  (1-frame)  : {m['rpe'] * 1000:.2f} mm")


@app.command("eval-hands")
def eval_hands(
    pred: Path = typer.Option(..., exists=True, help="Predicted joints_world.npy (T,2,21,3)"),
    gt: Path = typer.Option(..., exists=True, help="GT joints .npy (T,21,3) or (T,2,21,3)"),
    slot: int = typer.Option(1, help="Hand slot for pred (0=Left, 1=Right)"),
    to_mano: bool = typer.Option(False, help="Remap MediaPipe-21 -> MANO order before scoring"),
) -> None:
    """Hand-pose metrics: MPJPE, PA-MPJPE, AUC(PCK)."""
    import numpy as np

    from .eval.joint_maps import to_mano as remap_to_mano
    from .eval.metrics import hand_metrics

    p = np.load(pred)
    g = np.load(gt)
    p = p[:, slot] if p.ndim == 4 else p           # -> (T,21,3)
    g = g[:, slot] if g.ndim == 4 else g
    n = min(len(p), len(g))
    p, g = p[:n], g[:n]
    if to_mano:
        p = remap_to_mano(p)
    m = hand_metrics(p, g)
    console.print("[bold]Hand-pose metrics[/bold]")
    if m.get("frames", 0) == 0:
        console.print("  [yellow]no comparable frames (all NaN / no overlap)[/yellow]")
        return
    console.print(f"  frames   : {m['frames']}")
    console.print(f"  MPJPE    : {m['mpjpe']:.2f}")
    console.print(f"  PA-MPJPE : {m['pa_mpjpe']:.2f}")
    console.print(f"  AUC(PCK) : {m['auc_pck']:.3f}")


@app.command()
def viz(clip_dir: Path = typer.Option(..., exists=True, help="output/<clip_id> directory")) -> None:
    """Rebuild hand overlays from a processed clip's saved arrays."""
    import numpy as np

    from .config import PipelineConfig
    from .schema import ClipManifest
    from .stages.base import ClipContext
    from .viz import draw_hand_overlay

    manifest = ClipManifest.load(clip_dir / "manifest.json")
    ctx = ClipContext(
        clip_id=manifest.clip_id, clip_dir=clip_dir, video_path=Path(manifest.video_path),
        config=PipelineConfig.default(), manifest=manifest,
    )
    j2d = np.load(clip_dir / "hands" / "joints_2d.npy")
    out = draw_hand_overlay(ctx, j2d)
    console.print(f"[green]✓[/green] overlays → {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
