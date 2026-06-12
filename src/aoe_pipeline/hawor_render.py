"""HaWoR-style four-panel demo video rendering.

The renderer prefers HaWoR/MANO vertices when ``hands/verts_*.npy`` and
``hands/faces.npy`` are present, and falls back to a 21-joint hull/skeleton
emulation from the standard AoE hand arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .viz import HAND_CONNECTIONS, load_tum_poses

LEFT_COLOR = (219, 88, 226)   # BGR: pink/purple
RIGHT_COLOR = (235, 190, 58)  # BGR: cyan/blue
HAND_COLORS = (LEFT_COLOR, RIGHT_COLOR)
GRID_COLOR = (96, 96, 98)
FLOOR_COLOR = (72, 72, 74)


@dataclass
class MeshBundle:
    verts_cam: np.ndarray
    verts_world: np.ndarray
    faces: np.ndarray


def render_hawor_demo(
    clip_dir: Path,
    out_mp4: Path,
    fps: float = 30.0,
    size: int = 720,
    prefer_mesh: bool = True,
) -> Path:
    """Render a square 2x2 HaWoR-style demo video for a processed clip."""
    clip_dir = Path(clip_dir)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((clip_dir / "manifest.json").read_text())
    frames = _load_frames(clip_dir)
    joints_2d = np.load(clip_dir / "hands" / "joints_2d.npy").astype(float)
    joints_cam = _load_optional(clip_dir / "hands" / "joints_cam.npy")
    joints_world = np.load(clip_dir / "hands" / "joints_world.npy").astype(float)
    meshes = load_hawor_meshes(clip_dir) if prefer_mesh else None
    poses = _load_poses_or_identity(clip_dir, len(frames))

    T = min(len(frames), joints_2d.shape[0], joints_world.shape[0])
    if meshes is not None:
        T = min(T, meshes.verts_cam.shape[0], meshes.verts_world.shape[0])
    if joints_cam is not None:
        T = min(T, joints_cam.shape[0])
    if T <= 0:
        raise ValueError(f"no frames or hand arrays available in {clip_dir}")

    size = int(size)
    if size % 2:
        size += 1
    panel = size // 2
    fps = float(fps or manifest.get("fps") or 30.0)
    intr = manifest["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])

    world_points = meshes.verts_world[:T] if meshes is not None else joints_world[:T]
    bounds = _view_bounds(world_points, poses[:T])

    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {out_mp4}")

    for t in range(T):
        input_panel = _fit_panel(frames[t], panel)
        camera_panel = render_camera_panel(
            frames[t],
            meshes.verts_cam[t] if meshes is not None else None,
            meshes.faces if meshes is not None else None,
            K,
            panel,
            joints_2d[t],
        )
        source_world = meshes.verts_world[t] if meshes is not None else joints_world[t]
        top_panel = render_world_panel(
            source_world,
            meshes.faces if meshes is not None else None,
            poses,
            t,
            panel,
            bounds["top"],
            "top",
        )
        side_panel = render_world_panel(
            source_world,
            meshes.faces if meshes is not None else None,
            poses,
            t,
            panel,
            bounds["side"],
            "side",
        )

        _label(input_panel, "Input video")
        _label(camera_panel, "Camera view")
        _label(top_panel, "Top view")
        _label(side_panel, "Side view")
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:panel, :panel] = input_panel
        canvas[:panel, panel:] = camera_panel
        canvas[panel:, :panel] = top_panel
        canvas[panel:, panel:] = side_panel
        writer.write(canvas)

    writer.release()
    return out_mp4


def load_hawor_meshes(clip_dir: Path) -> MeshBundle | None:
    hands = Path(clip_dir) / "hands"
    paths = [hands / "verts_cam.npy", hands / "verts_world.npy", hands / "faces.npy"]
    if not all(p.exists() for p in paths):
        return None
    verts_cam = np.load(paths[0]).astype(float)
    verts_world = np.load(paths[1]).astype(float)
    faces = np.load(paths[2]).astype(np.int32)
    if verts_cam.ndim != 4 or verts_world.ndim != 4 or verts_cam.shape[:2] != verts_world.shape[:2]:
        raise ValueError("verts_cam.npy and verts_world.npy must have shape (T,2,V,3)")
    if verts_cam.shape[-1] != 3 or verts_world.shape[-1] != 3:
        raise ValueError("MANO vertex arrays must end in xyz coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces.npy must have shape (F,3)")
    if len(faces) and int(faces.max()) >= verts_cam.shape[2]:
        raise ValueError("faces.npy references vertices outside verts_*.npy")
    return MeshBundle(verts_cam=verts_cam, verts_world=verts_world, faces=faces)


def render_camera_panel(
    frame: np.ndarray,
    verts_cam: np.ndarray | None,
    faces: np.ndarray | None,
    intrinsics: np.ndarray,
    size: int = 360,
    joints_2d: np.ndarray | None = None,
) -> np.ndarray:
    panel, scale, x0, y0 = _fit_panel_with_transform(frame, size)
    overlay = panel.copy()

    if verts_cam is not None and faces is not None:
        for s in range(min(2, verts_cam.shape[0])):
            verts = verts_cam[s]
            if not np.isfinite(verts).all():
                continue
            pts = _project_camera(verts, intrinsics)
            valid = np.isfinite(pts).all(axis=1) & (verts[:, 2] > 1e-6)
            pts = np.column_stack([pts[:, 0] * scale + x0, pts[:, 1] * scale + y0])
            _draw_mesh_projected(
                overlay,
                pts,
                verts,
                valid,
                faces,
                HAND_COLORS[s],
                sort_axis=2,
                reverse=True,
            )
    elif joints_2d is not None:
        for s in range(min(2, joints_2d.shape[0])):
            pts = joints_2d[s].copy()
            if not np.isfinite(pts).all():
                continue
            pts[:, 0] = pts[:, 0] * scale + x0
            pts[:, 1] = pts[:, 1] * scale + y0
            _draw_joint_hand(overlay, pts, HAND_COLORS[s], filled=True)

    return cv2.addWeighted(overlay, 0.58, panel, 0.42, 0)


def render_world_panel(
    points3d: np.ndarray,
    faces: np.ndarray | None,
    poses: np.ndarray,
    frame_idx: int,
    size: int,
    bounds: tuple[float, float, float, float],
    view: str,
) -> np.ndarray:
    panel = np.full((size, size, 3), 42, np.uint8)
    _draw_floor(panel, size)
    camera_xy = _project_world(poses[: frame_idx + 1, :3, 3], view)
    trail = _map_points(camera_xy, bounds, size)
    if len(trail) > 1:
        cv2.polylines(panel, [trail.astype(np.int32)], False, (160, 160, 160), 1, cv2.LINE_AA)
    if len(trail):
        _draw_camera_marker(panel, trail[-1], size)

    for s in range(min(2, points3d.shape[0])):
        pts3 = points3d[s]
        if not np.isfinite(pts3).all():
            continue
        pts2 = _map_points(_project_world(pts3, view), bounds, size)
        _draw_shadow(panel, pts2, size)
        if faces is not None and pts3.shape[0] > int(faces.max(initial=0)):
            _draw_mesh_projected(
                panel,
                pts2,
                pts3,
                np.ones(len(pts2), dtype=bool),
                faces,
                HAND_COLORS[s],
                sort_axis=1 if view == "top" else 0,
                reverse=True,
            )
        else:
            _draw_joint_hand(panel, pts2, HAND_COLORS[s], filled=True)
    return panel


def _load_frames(clip_dir: Path) -> list[np.ndarray]:
    files = sorted((clip_dir / "frames").glob("frame_*.png"))
    frames = [cv2.imread(str(f)) for f in files]
    return [f for f in frames if f is not None]


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path).astype(float) if path.exists() else None


def _load_poses_or_identity(clip_dir: Path, expected: int) -> np.ndarray:
    path = clip_dir / "trajectory.tum"
    if path.exists():
        poses = load_tum_poses(path)
        if len(poses) >= expected:
            return poses
    return np.repeat(np.eye(4)[None], expected, axis=0)


def _fit_panel(img: np.ndarray, size: int) -> np.ndarray:
    return _fit_panel_with_transform(img, size)[0]


def _fit_panel_with_transform(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out, scale, x0, y0


def _project_camera(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points[:, 2:3]
    uvw = (K @ points.T).T
    return uvw[:, :2] / np.where(np.abs(z) < 1e-6, np.nan, z)


def _project_world(points: np.ndarray, view: str) -> np.ndarray:
    if view == "top":
        return np.column_stack([points[..., 0].reshape(-1), points[..., 2].reshape(-1)])
    if view == "side":
        return np.column_stack([points[..., 2].reshape(-1), -points[..., 1].reshape(-1)])
    raise ValueError(f"unknown world view: {view}")


Bounds = tuple[float, float, float, float]


def _view_bounds(points: np.ndarray, poses: np.ndarray) -> dict[str, Bounds]:
    out = {}
    flat = points.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    cams = poses[:, :3, 3] if len(poses) else np.zeros((1, 3))
    for view in ("top", "side"):
        p = _project_world(flat, view) if len(flat) else np.zeros((1, 2))
        c = _project_world(cams, view)
        xy = np.vstack([p, c])
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        center = (lo + hi) / 2.0
        half = max(float((hi - lo).max()) * 0.60, 0.25)
        out[view] = (center[0] - half, center[0] + half, center[1] - half, center[1] + half)
    return out


def _map_points(points: np.ndarray, bounds: Bounds, size: int) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    x = (points[:, 0] - x0) / max(x1 - x0, 1e-9) * (size - 1)
    y = (1.0 - (points[:, 1] - y0) / max(y1 - y0, 1e-9)) * (size - 1)
    return np.column_stack([x, y]).astype(np.float32)


def _draw_mesh_2d(
    img: np.ndarray,
    pts: np.ndarray,
    valid: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    h, w = img.shape[:2]
    for tri in faces:
        if not valid[tri].all():
            continue
        poly = np.round(pts[tri]).astype(np.int32)
        outside = (
            poly[:, 0].max() < 0
            or poly[:, 1].max() < 0
            or poly[:, 0].min() >= w
            or poly[:, 1].min() >= h
        )
        if outside:
            continue
        cv2.fillConvexPoly(img, poly, color, lineType=cv2.LINE_AA)
        cv2.polylines(img, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_mesh_projected(
    img: np.ndarray,
    pts2: np.ndarray,
    pts3: np.ndarray,
    valid: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    sort_axis: int,
    reverse: bool,
) -> None:
    face_items = []
    for tri in faces:
        if not valid[tri].all():
            continue
        poly = np.round(pts2[tri]).astype(np.int32)
        if _outside_image(poly, img.shape[:2]):
            continue
        depth = float(np.nanmean(pts3[tri, sort_axis]))
        shade = _face_shade(pts3[tri])
        face_items.append((depth, shade, poly))

    face_items.sort(key=lambda item: item[0], reverse=reverse)
    for _, shade, poly in face_items:
        face_color = _shade_color(color, shade)
        edge_color = _shade_color((255, 255, 255), max(0.25, shade * 0.85))
        cv2.fillConvexPoly(img, poly, face_color, lineType=cv2.LINE_AA)
        cv2.polylines(img, [poly], True, edge_color, 1, cv2.LINE_AA)


def _outside_image(poly: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        poly[:, 0].max() < 0
        or poly[:, 1].max() < 0
        or poly[:, 0].min() >= w
        or poly[:, 1].min() >= h
    )


def _face_shade(tri3: np.ndarray) -> float:
    n = np.cross(tri3[1] - tri3[0], tri3[2] - tri3[0])
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return 0.70
    n = n / norm
    light = np.array([0.25, -0.55, 0.80])
    light = light / np.linalg.norm(light)
    return float(np.clip(0.58 + 0.36 * abs(np.dot(n, light)), 0.45, 0.96))


def _shade_color(color: tuple[int, int, int], shade: float) -> tuple[int, int, int]:
    arr = np.asarray(color, dtype=float) * shade
    return tuple(int(np.clip(v, 0, 255)) for v in arr)


def _draw_joint_hand(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    filled: bool = False,
) -> None:
    if filled:
        hull = cv2.convexHull(np.round(pts).astype(np.float32))
        cv2.fillConvexPoly(img, hull.astype(np.int32), color, lineType=cv2.LINE_AA)
    ipts = np.round(pts).astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(ipts[a]), tuple(ipts[b]), (255, 255, 255), 2, cv2.LINE_AA)
    for p in ipts:
        cv2.circle(img, tuple(p), 2, (20, 20, 20), -1, cv2.LINE_AA)


def _draw_camera_marker(img: np.ndarray, pt: np.ndarray, size: int) -> None:
    p = tuple(np.round(pt).astype(int))
    r = max(5, size // 45)
    color = (178, 72, 184)
    cv2.circle(img, p, r, color, 2, cv2.LINE_AA)
    cv2.line(img, p, (p[0], p[1] - 3 * r), color, 2, cv2.LINE_AA)
    cv2.line(img, p, (p[0] - 2 * r, p[1] + 2 * r), color, 1, cv2.LINE_AA)
    cv2.line(img, p, (p[0] + 2 * r, p[1] + 2 * r), color, 1, cv2.LINE_AA)


def _draw_floor(img: np.ndarray, size: int) -> None:
    horizon = int(size * 0.68)
    cv2.rectangle(img, (0, horizon), (size, size), FLOOR_COLOR, -1)
    spacing = max(18, size // 14)
    for y in range(horizon, size, spacing):
        cv2.line(img, (0, y), (size, y), GRID_COLOR, 1, cv2.LINE_AA)
    for x in range(0, size, spacing):
        cv2.line(img, (x, horizon), (x, size), GRID_COLOR, 1, cv2.LINE_AA)


def _draw_shadow(img: np.ndarray, pts2: np.ndarray, size: int) -> None:
    pts = pts2[np.isfinite(pts2).all(axis=1)]
    if len(pts) < 3:
        return
    shadow = pts.copy()
    floor_y = size * 0.78
    shadow[:, 1] = 0.82 * floor_y + 0.18 * shadow[:, 1]
    shadow[:, 0] = 0.92 * shadow[:, 0] + 0.08 * size / 2
    hull = cv2.convexHull(shadow.astype(np.float32)).astype(np.int32)
    layer = img.copy()
    cv2.fillConvexPoly(layer, hull, (10, 10, 10), lineType=cv2.LINE_AA)
    cv2.addWeighted(layer, 0.18, img, 0.82, 0, dst=img)


def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (13, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        img,
        text,
        (13, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
