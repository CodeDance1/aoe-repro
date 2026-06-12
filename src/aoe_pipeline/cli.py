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


@app.command("render-hawor-demo")
def render_hawor_demo_cmd(
    clip_dir: Path = typer.Option(..., exists=True, help="output/<clip_id> directory"),
    out: Path = typer.Option(None, help="Output MP4 path"),
    fps: float = typer.Option(30.0, help="Output frames per second"),
    size: int = typer.Option(720, help="Square output size in pixels"),
    prefer_mesh: str = typer.Option("true", help="Use MANO mesh files when available: true|false"),
) -> None:
    """Render a HaWoR-style 2x2 demo video from saved AoE outputs."""
    from .hawor_render import render_hawor_demo

    out_mp4 = out or clip_dir / "viz" / "hawor_demo.mp4"
    rendered = render_hawor_demo(
        clip_dir=clip_dir,
        out_mp4=out_mp4,
        fps=fps,
        size=size,
        prefer_mesh=_parse_bool_option(prefer_mesh, "prefer-mesh"),
    )
    console.print(f"[green]✓[/green] HaWoR demo → {rendered}")


@app.command("check-hawor-demo")
def check_hawor_demo_cmd(
    clip_dir: Path = typer.Option(..., exists=True, help="output/<clip_id> directory"),
    demo_mp4: Path = typer.Option(None, help="Rendered HaWoR-style MP4 path"),
    require_mesh: str = typer.Option("true", help="Require MANO mesh files: true|false"),
    expected_size: int = typer.Option(720, help="Expected square MP4 size in pixels"),
) -> None:
    """Validate HaWoR hybrid arrays, MANO meshes, and rendered demo MP4."""
    from .hawor_check import check_hawor_outputs

    require_mesh_bool = _parse_bool_option(require_mesh, "require-mesh")
    try:
        report = check_hawor_outputs(
            clip_dir=clip_dir,
            demo_mp4=demo_mp4,
            require_mesh=require_mesh_bool,
            expected_size=expected_size,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print("[green]✓[/green] HaWoR outputs validated")
    console.print(f"  frames_total          : {report['frames_total']}")
    console.print(f"  frames_with_world_hand: {report['frames_with_world_hand']}")
    console.print(f"  reprojection_median_px: {report['median_joint_reprojection_px']:.3f}")
    if require_mesh_bool:
        console.print(f"  frames_with_mesh      : {report['frames_with_mesh']}")
        console.print(f"  mesh_vertices         : {report['mesh_vertices']}")
        console.print(f"  mesh_faces            : {report['mesh_faces']}")
        console.print(f"  visible_mesh_vertices : {report['visible_projected_mesh_vertices']}")
    if "demo_video" in report:
        video = report["demo_video"]
        console.print(
            f"  demo_video            : {video['width']}x{video['height']}, "
            f"{video['frames']} frames"
        )


def _parse_bool_option(value: str | bool, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise typer.BadParameter(f"--{name} must be true or false")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
