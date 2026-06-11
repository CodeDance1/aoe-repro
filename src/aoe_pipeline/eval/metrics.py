"""Reproduction of the paper's precision metrics.

Hand pose (units follow the inputs; report mm if joints are in mm):
  - MPJPE       mean per-joint position error
  - PA-MPJPE    Procrustes(similarity)-aligned MPJPE
  - AUC(PCK)    area under the PCK curve over a threshold sweep

Camera trajectory (via `evo`):
  - ATE   (Sim(3) / 7-DoF alignment, scale corrected)
  - ATE-S (SE(3) / 6-DoF alignment, scale-free)
  - RPE   (relative pose error, 1-frame delta)
"""

from __future__ import annotations

import numpy as np


# --- hand pose -----------------------------------------------------------------
def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint L2 error over finite joints. ``pred``/``gt``: (J, 3)."""
    d = np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)
    m = np.isfinite(d)
    return float(d[m].mean()) if m.any() else float("nan")


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity transform (R, t, s) aligning ``src`` onto ``dst`` (least squares)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-aligned MPJPE (rotation + translation + uniform scale)."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    m = np.isfinite(pred).all(1) & np.isfinite(gt).all(1)
    if m.sum() < 3:
        return float("nan")
    R, t, s = umeyama(pred[m], gt[m], with_scale=True)
    aligned = (s * (R @ pred[m].T)).T + t
    return mpjpe(aligned, gt[m])


def pck_auc(pred: np.ndarray, gt: np.ndarray, thresholds=None):
    """AUC of the PCK curve. Returns (auc, pck_curve, thresholds)."""
    if thresholds is None:
        thresholds = np.linspace(0, 50, 20)  # mm-scale default
    thresholds = np.asarray(thresholds, float)
    d = np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), np.zeros_like(thresholds), thresholds
    pck = np.array([(d <= th).mean() for th in thresholds])
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # renamed in numpy 2.0
    auc = float(trapz(pck, thresholds) / (thresholds[-1] - thresholds[0]))
    return auc, pck, thresholds


def hand_metrics(pred_seq: np.ndarray, gt_seq: np.ndarray, thresholds=None) -> dict:
    """Aggregate hand metrics over a sequence.

    ``pred_seq``/``gt_seq``: (N, J, 3) for N valid frames (already slot-matched
    and joint-aligned). Frames with any NaN are dropped pairwise.
    """
    pred_seq, gt_seq = np.asarray(pred_seq, float), np.asarray(gt_seq, float)
    mpjpes, pampjpes, all_pred, all_gt = [], [], [], []
    for p, g in zip(pred_seq, gt_seq):
        m = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if m.sum() < 3:
            continue
        mpjpes.append(mpjpe(p[m], g[m]))
        pampjpes.append(pa_mpjpe(p, g))
        all_pred.append(p[m]); all_gt.append(g[m])
    if not mpjpes:
        return {"frames": 0}
    auc, _, _ = pck_auc(np.vstack(all_pred), np.vstack(all_gt), thresholds)
    return {
        "frames": len(mpjpes),
        "mpjpe": float(np.mean(mpjpes)),
        "pa_mpjpe": float(np.mean(pampjpes)),
        "auc_pck": auc,
    }


# --- trajectory ----------------------------------------------------------------
def trajectory_metrics(est_tum: str, gt_tum: str) -> dict:
    """ATE (Sim3), ATE-S (SE3, scale-free), and RPE via evo, given TUM files."""
    import copy

    from evo.core import metrics, sync
    from evo.tools import file_interface

    ref = file_interface.read_tum_trajectory_file(gt_tum)
    est = file_interface.read_tum_trajectory_file(est_tum)
    ref, est = sync.associate_trajectories(ref, est)

    def ape(correct_scale: bool) -> float:
        e = copy.deepcopy(est)
        e.align(ref, correct_scale=correct_scale)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((ref, e))
        return float(ape_metric.get_statistic(metrics.StatisticsType.rmse))

    rpe_metric = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=1, delta_unit=metrics.Unit.frames, all_pairs=False,
    )
    e_se3 = copy.deepcopy(est)
    e_se3.align(ref, correct_scale=False)
    rpe_metric.process_data((ref, e_se3))

    return {
        "pairs": ref.num_poses,
        "ate_sim3_7dof": ape(True),
        "ate_s_se3_6dof": ape(False),
        "rpe": float(rpe_metric.get_statistic(metrics.StatisticsType.rmse)),
    }
