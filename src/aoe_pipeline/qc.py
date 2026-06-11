"""Quality-control math (pure functions, no I/O) for Stage 6.

Arrays use shape (T, S, J, 3) for world joints and (T, S, J, 2) for 2D joints,
where T=frames, S=hand slots (Left/Right), J=21 joints. Missing detections are
NaN and are never flagged.
"""

from __future__ import annotations

import numpy as np


def joint_velocities(joints_world: np.ndarray, dt: float) -> np.ndarray:
    """Per-joint speed (units/sec). Shape (T, S, J); row 0 is NaN."""
    diff = joints_world[1:] - joints_world[:-1]
    vel = np.linalg.norm(diff, axis=-1) / max(dt, 1e-9)  # (T-1, S, J)
    pad = np.full((1,) + vel.shape[1:], np.nan, dtype=vel.dtype)
    return np.concatenate([pad, vel], axis=0)


def kinematic_outliers(
    joints_world: np.ndarray, dt: float, sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Flag per-joint velocity z-scores above ``sigma``.

    Returns (flags, velocities), both shape (T, S, J).
    """
    vel = joint_velocities(joints_world, dt)
    T, S, J = vel.shape
    flags = np.zeros((T, S, J), dtype=bool)
    for s in range(S):
        for j in range(J):
            v = vel[:, s, j]
            m = np.isfinite(v)
            if m.sum() < 3:
                continue
            mu, sd = v[m].mean(), v[m].std()
            if sd < 1e-9:
                continue
            z = np.abs((v - mu) / sd)
            flags[:, s, j] = np.where(np.isfinite(z), z > sigma, False)
    return flags, vel


def reprojection_error(
    joints_world: np.ndarray, joints_2d: np.ndarray, poses, K: np.ndarray
) -> np.ndarray:
    """Pixel reprojection error of world joints vs detected 2D. Shape (T, S, J)."""
    T, S, J, _ = joints_world.shape
    err = np.full((T, S, J), np.nan, dtype=np.float64)
    for t in range(T):
        P = poses[t] if (poses is not None and t < len(poses)) else np.eye(4)
        Pinv = np.linalg.inv(P)  # world -> cam
        R, tt = Pinv[:3, :3], Pinv[:3, 3]
        for s in range(S):
            Xw, d2 = joints_world[t, s], joints_2d[t, s]
            if not (np.isfinite(Xw).all() and np.isfinite(d2).all()):
                continue
            Xc = (R @ Xw.T).T + tt  # (J, 3)
            z = Xc[:, 2:3]
            z = np.where(np.abs(z) < 1e-6, 1e-6, z)
            uv = (K @ Xc.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            err[t, s] = np.linalg.norm(uv - d2, axis=1)
    return err


def reprojection_outliers(err: np.ndarray, px: float = 5.0) -> np.ndarray:
    return np.where(np.isfinite(err), err > px, False)


def frames_flagged(flags: np.ndarray) -> np.ndarray:
    """Boolean per-frame: any joint flagged in that frame. Input (T, S, J)."""
    return flags.any(axis=(1, 2))
