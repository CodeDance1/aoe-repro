"""Visualization helpers: hand-joint overlays and camera-trajectory plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 21-joint MediaPipe hand topology.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
SLOT_COLORS = [(0, 255, 0), (0, 165, 255)]  # BGR: Left=green, Right=orange


def draw_hand_overlay(ctx, joints_2d: np.ndarray, max_frames: int | None = None) -> Path:
    """Draw 2D hand skeletons on each frame into ``viz/hands/``."""
    frames = ctx.get_frames()
    out_dir = ctx.viz_dir / "hands"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames) if max_frames is None else min(len(frames), max_frames)
    for t in range(n):
        img = frames[t].copy()
        for s in range(joints_2d.shape[1]):
            pts = joints_2d[t, s]
            if np.isnan(pts).any():
                continue
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, _ipt(pts[a]), _ipt(pts[b]), color, 2)
            for j in range(pts.shape[0]):
                cv2.circle(img, _ipt(pts[j]), 3, color, -1)
        cv2.imwrite(str(out_dir / f"hands_{t:06d}.png"), img)
    return out_dir


def plot_trajectory(poses, out_path: str | Path) -> Path:
    """3D plot of camera positions (world translation per pose)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (registers 3d projection)

    P = np.asarray(poses)
    xs, ys, zs = P[:, 0, 3], P[:, 1, 3], P[:, 2, 3]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, "-o", ms=2, lw=1)
    ax.scatter(xs[:1], ys[:1], zs[:1], c="g", s=40, label="start")
    ax.scatter(xs[-1:], ys[-1:], zs[-1:], c="r", s=40, label="end")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("camera trajectory (up to scale)")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def _ipt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


# --- HaWoR-style multi-view rendering ----------------------------------------
# World/camera frame is OpenCV convention: X right, Y down, Z forward.
# Orthographic views (each maps a 3D point -> 2D (horizontal, vertical)):
#   front: ( X, -Y)        top: ( X,  Z)        side: ( Z, -Y)
_VIEW_SPEC = {
    "front": ((0, 1.0), (1, -1.0), "X", "up (-Y)"),
    "top":   ((0, 1.0), (2, 1.0), "X", "depth (Z)"),
    "side":  ((2, 1.0), (1, -1.0), "depth (Z)", "up (-Y)"),
}
VIEWS = ("front", "top", "side")
# Matplotlib RGB (BGR SLOT_COLORS -> RGB 0..1): Left=green, Right=orange.
_MPL_COLORS = [(c[2] / 255, c[1] / 255, c[0] / 255) for c in SLOT_COLORS]


def load_tum_poses(path: str | Path) -> np.ndarray:
    """Parse a TUM trajectory file into an (T,4,4) camera->world SE(3) stack."""
    from scipy.spatial.transform import Rotation

    poses = []
    for line in Path(path).read_text().strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        _, tx, ty, tz, qx, qy, qz, qw = (float(v) for v in line.split())
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        poses.append(T)
    return np.asarray(poses)


def _project(points: np.ndarray, view: str) -> np.ndarray:
    """Orthographic projection of (...,3) world/camera points to (...,2)."""
    (uc, us), (vc, vs), _, _ = _VIEW_SPEC[view]
    p = np.asarray(points)
    return np.stack([p[..., uc] * us, p[..., vc] * vs], axis=-1)


def _clean_world(joints_world: np.ndarray, dt: float, med_window: int = 5) -> np.ndarray:
    """Viz-only: drop kinematic-outlier joints, interpolate gaps, median-smooth.

    Does not modify the saved arrays. Removes the handful of ~2.5-unit spike
    frames that monocular-VO drift / detection flips inject into world coords.
    """
    from scipy.ndimage import median_filter

    from . import qc

    flags, _ = qc.kinematic_outliers(joints_world, dt, sigma=3.0)
    jw = joints_world.astype(float).copy()
    jw[flags] = np.nan
    T, S, J, _ = jw.shape
    for s in range(S):
        for j in range(J):
            for c in range(3):
                col = jw[:, s, j, c]
                idx = np.where(np.isfinite(col))[0]
                if len(idx) < 2:
                    continue
                span = np.arange(idx[0], idx[-1] + 1)
                vals = np.interp(span, idx, col[idx])
                if len(vals) >= med_window:
                    vals = median_filter(vals, size=med_window)
                jw[span, s, j, c] = vals
    return jw


def _camera_joints_per_joint_depth(clip_dir: Path, manifest: dict) -> np.ndarray:
    """Recompute camera-frame joints sampling depth at EACH joint pixel.

    Gives the hand real within-hand depth (finger thickness) for side/top views,
    unlike the saved wrist-anchored joints which are planar. Reuses saved
    joints_2d + depth maps; re-smoothed in the camera frame.
    """
    from .stages.s4_hands import NUM_JOINTS, _smooth_series

    j2 = np.load(clip_dir / "hands" / "joints_2d.npy").astype(float)  # (T,2,21,2)
    T = j2.shape[0]
    intr = manifest["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    W, H = manifest["width"], manifest["height"]
    depth_files = sorted((clip_dir / "depth").glob("depth_*.npy"))

    joints = np.full((T, 2, NUM_JOINTS, 3), np.nan)
    for t in range(T):
        dm = np.load(depth_files[t]) if t < len(depth_files) else None
        for s in range(2):
            pts = j2[t, s]
            if not np.isfinite(pts).all():
                continue
            for j in range(NUM_JOINTS):
                u = min(max(int(round(pts[j, 0])), 0), W - 1)
                v = min(max(int(round(pts[j, 1])), 0), H - 1)
                d = float(dm[v, u]) if dm is not None else 1.0
                d = d if (np.isfinite(d) and d > 0) else 1.0
                joints[t, s, j] = (Kinv @ np.array([pts[j, 0], pts[j, 1], 1.0])) * d
    for s in range(2):
        flat = joints[:, s].reshape(T, NUM_JOINTS * 3)
        joints[:, s] = _smooth_series(flat, 5, 3).reshape(T, NUM_JOINTS, 3)
    return joints


def _to_plot(p: np.ndarray) -> np.ndarray:
    """World (X-right, Y-down, Z-forward) -> plot coords with height up.

    plot = (X, Z, -Y): x=lateral, y=depth, z=height.
    """
    p = np.asarray(p)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def render_world_scene(clip_dir: str | Path, out_mp4: str | Path | None = None,
                       fps: float | None = None, stride: int = 2, n_ghosts: int = 10,
                       hand_scale: float = 2.5, dpi: int = 120) -> Path:
    """HaWoR-style world-view emulation: 2 perspective panels with a ground plane,
    solid hands (convex-hull of the per-joint 3D joints), faded ghost hands along
    the trajectory, the camera path on the floor, and a camera frustum.

    Hands are hull-emulated (we have 21 joints, not a MANO mesh) and the world
    layout is up-to-scale (monocular VO).
    """
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import ConvexHull

    clip_dir = Path(clip_dir)
    manifest = json.loads((clip_dir / "manifest.json").read_text())
    src_fps = float(manifest.get("fps") or 30.0)
    out_fps = float(fps or src_fps / stride)

    poses = load_tum_poses(clip_dir / "trajectory.tum")
    jc = _camera_joints_per_joint_depth(clip_dir, manifest)  # (T,2,21,3) camera frame
    T = jc.shape[0]
    world = np.full_like(jc, np.nan)
    for t in range(T):
        R, tt = poses[t][:3, :3], poses[t][:3, 3]
        for s in range(2):
            if np.isfinite(jc[t, s]).all():
                world[t, s] = (R @ jc[t, s].T).T + tt
    world = _clean_world(world, 1.0 / src_fps)

    camM = _to_plot(gaussian_filter1d(poses[:, :3, 3], sigma=2, axis=0))
    worldM = _to_plot(world)

    pts = worldM.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    allp = np.vstack([pts, camM])
    mins, maxs = allp.min(0), allp.max(0)
    rng = np.maximum(maxs - mins, 1e-3)
    floor_z = mins[2] - 0.05 * rng[2]
    floor = [np.array([[mins[0], mins[1], floor_z], [maxs[0], mins[1], floor_z],
                       [maxs[0], maxs[1], floor_z], [mins[0], maxs[1], floor_z]])]

    HAND_RGB = [(0.25, 0.75, 1.0), (1.0, 0.55, 0.82)]  # cyan=Left, pink=Right
    frames_idx = list(range(0, T, stride))
    ghost_idx = frames_idx[:: max(1, len(frames_idx) // n_ghosts)]

    def draw_hand(ax, pj, color, alpha, edge=True):
        pj = pj.mean(0) + hand_scale * (pj - pj.mean(0))  # enlarge for visibility (cosmetic)
        try:
            hull = ConvexHull(pj)
            tris = [pj[s] for s in hull.simplices]
            ax.add_collection3d(Poly3DCollection(
                tris, facecolor=color, alpha=alpha,
                edgecolor=(0, 0, 0, 0.12) if edge else (0, 0, 0, 0), linewidths=0.15))
        except Exception:  # near-coplanar -> fill the palm fan instead
            palm = pj[[0, 1, 5, 9, 13, 17]]
            ax.add_collection3d(Poly3DCollection([palm], facecolor=color, alpha=alpha))
        segs = [[pj[a], pj[b]] for a, b in HAND_CONNECTIONS]
        ax.add_collection3d(Line3DCollection(segs, colors=[(0, 0, 0, 0.5 * alpha)], linewidths=0.7))

    def draw_frustum(ax, pose, scale):
        R, c = pose[:3, :3], pose[:3, 3]
        corners = (R @ (np.array([[-.5, -.5, 1], [.5, -.5, 1], [.5, .5, 1], [-.5, .5, 1]]) * scale).T).T + c
        P = _to_plot(np.vstack([c, corners]))
        segs = [[P[0], P[i]] for i in range(1, 5)] + [[P[i], P[1 + i % 4]] for i in range(1, 5)]
        ax.add_collection3d(Line3DCollection(segs, colors=[(0.5, 0.2, 0.6, 0.9)], linewidths=1.0))

    out_mp4 = Path(out_mp4) if out_mp4 else clip_dir / "viz" / "hand_views_scene.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 5.2), dpi=dpi)
    fig.patch.set_facecolor("white")
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    azims = (-72, 24)
    fig.canvas.draw()
    W, Hh = fig.canvas.get_width_height()
    W -= W % 2; Hh -= Hh % 2
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, Hh))

    for fi in frames_idx:
        for ax, az in zip(axes, azims):
            ax.cla()
            ax.set_axis_off()
            ax.set_proj_type("persp")
            ax.view_init(elev=22, azim=az)
            ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(floor_z, maxs[2])
            ax.set_box_aspect((rng[0], rng[1], maxs[2] - floor_z))
            ax.add_collection3d(Poly3DCollection(floor, facecolor=(0.62, 0.62, 0.64),
                                                 alpha=0.25, edgecolor=(0.4, 0.4, 0.4, 0.3)))
            ax.plot(camM[: fi + 1, 0], camM[: fi + 1, 1], np.full(fi + 1, floor_z),
                    color=(0.3, 0.3, 0.3), lw=0.8)
            draw_frustum(ax, poses[fi], scale=0.12 * (maxs[2] - floor_z))
            for gi in ghost_idx:
                if gi > fi:
                    break
                for s in range(2):
                    if np.isfinite(worldM[gi, s]).all():
                        draw_hand(ax, worldM[gi, s], (0.55, 0.55, 0.55), 0.10, edge=False)
            for s in range(2):
                if np.isfinite(worldM[fi, s]).all():
                    draw_hand(ax, worldM[fi, s], HAND_RGB[s], 0.7)
        fig.suptitle(f"HaWoR-style world view (emulated, up-to-scale)   t={fi / src_fps:5.2f}s",
                     fontsize=11)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.02)
        fig.canvas.draw()
        bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:Hh, :W], cv2.COLOR_RGBA2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)
    return out_mp4


def render_multiview(clip_dir: str | Path, frame: str = "world",
                     out_mp4: str | Path | None = None, fps: float | None = None,
                     dpi: int = 110, camera_depth: str = "saved") -> Path:
    """Render a Front|Top|Side orthographic animation of the 21-joint hands.

    frame='camera' uses joints_cam (clean, camera at origin); frame='world' uses
    joints_world plus the camera-trajectory trail (qualitative — up-to-scale VO).
    camera_depth='per_joint' recomputes camera-frame joints with per-joint depth
    so the hand has finger thickness in top/side (vs the planar wrist-anchored save).
    """
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clip_dir = Path(clip_dir)
    manifest = json.loads((clip_dir / "manifest.json").read_text())
    fps = float(fps or manifest.get("fps") or 30.0)

    if frame == "camera":
        if camera_depth == "per_joint":
            joints = _camera_joints_per_joint_depth(clip_dir, manifest)
            caption = "camera frame (per-joint depth) — hand 3D shape, finger thickness visible"
        else:
            joints = np.load(clip_dir / "hands" / "joints_cam.npy").astype(float)
            caption = "camera frame — hand pose relative to camera (wrist-anchored: planar)"
        cam_pos = cam_fwd = None
    else:
        joints = _clean_world(np.load(clip_dir / "hands" / "joints_world.npy").astype(float), 1.0 / fps)
        poses = load_tum_poses(clip_dir / "trajectory.tum")
        cam_pos = poses[:, :3, 3]
        cam_fwd = poses[:, :3, 2]  # camera +Z (look direction) in world
        caption = "world frame — hands + camera trail (qualitative: up-to-scale monocular VO)"

    T = joints.shape[0]
    out_mp4 = Path(out_mp4) if out_mp4 else clip_dir / "viz" / f"hand_views_{frame}.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # global per-view center + a shared half-range (equal scale across panels)
    allpts = joints.reshape(-1, 3)
    centers, half = {}, 0.0
    for v in VIEWS:
        P = _project(allpts, v)
        P = P[np.isfinite(P).all(axis=1)]
        if cam_pos is not None:
            P = np.vstack([P, _project(cam_pos, v)])
        cu = (P[:, 0].min() + P[:, 0].max()) / 2
        cv = (P[:, 1].min() + P[:, 1].max()) / 2
        centers[v] = (cu, cv)
        half = max(half, (P[:, 0].max() - P[:, 0].min()) / 2, (P[:, 1].max() - P[:, 1].min()) / 2)
    half = max(half * 1.08, 1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), dpi=dpi)
    fig.canvas.draw()
    h, w = fig.canvas.get_width_height()[::-1]
    w -= w % 2; h -= h % 2  # even dims for the codec
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for t in range(T):
        for ax, v in zip(axes, VIEWS):
            ax.cla()
            cu, cv = centers[v]
            ax.set_xlim(cu - half, cu + half)
            ax.set_ylim(cv - half, cv + half)
            ax.set_aspect("equal")
            _, _, xl, yl = _VIEW_SPEC[v]
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.set_title(f"{v} view", fontsize=10)
            ax.tick_params(labelsize=6)

            if cam_pos is not None:  # world: camera trail + current pose
                trail = _project(cam_pos[: t + 1], v)
                ax.plot(trail[:, 0], trail[:, 1], "-", color="0.6", lw=0.8, zorder=1)
                cur = _project(cam_pos[t], v)
                ax.plot(cur[0], cur[1], "^", color="0.2", ms=6, zorder=3)
                fwd = _project(cam_pos[t] + 0.3 * half * cam_fwd[t] / (np.linalg.norm(cam_fwd[t]) + 1e-9), v)
                ax.annotate("", xy=(fwd[0], fwd[1]), xytext=(cur[0], cur[1]),
                            arrowprops=dict(arrowstyle="->", color="0.2", lw=1))

            for s in range(joints.shape[1]):
                pj = _project(joints[t, s], v)
                if not np.isfinite(pj).all():
                    continue
                color = _MPL_COLORS[s % len(_MPL_COLORS)]
                for a, b in HAND_CONNECTIONS:
                    ax.plot([pj[a, 0], pj[b, 0]], [pj[a, 1], pj[b, 1]], "-", color=color, lw=1.5, zorder=4)
                ax.scatter(pj[:, 0], pj[:, 1], s=8, color=color, zorder=5)

        fig.suptitle(f"{caption}    frame {t}/{T - 1}   t={t / fps:5.2f}s", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba[:h, :w], cv2.COLOR_RGBA2BGR)
        writer.write(bgr)

    writer.release()
    plt.close(fig)
    return out_mp4
