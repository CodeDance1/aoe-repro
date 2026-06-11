from __future__ import annotations

import numpy as np

from aoe_pipeline.eval import metrics


def test_mpjpe_zero_and_shift():
    rng = np.random.default_rng(0)
    gt = rng.uniform(-1, 1, size=(21, 3))
    assert metrics.mpjpe(gt, gt) == 0.0
    shifted = gt + np.array([3.0, 4.0, 0.0])
    assert np.isclose(metrics.mpjpe(shifted, gt), 5.0)


def test_pa_mpjpe_invariant_to_similarity():
    rng = np.random.default_rng(1)
    gt = rng.uniform(-1, 1, size=(21, 3))
    # rotate + scale + translate; PA-MPJPE should undo it -> ~0
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    pred = (2.5 * (R @ gt.T)).T + np.array([5.0, -3.0, 1.0])
    assert metrics.pa_mpjpe(pred, gt) < 1e-6


def test_pck_auc_perfect():
    gt = np.random.default_rng(2).uniform(-1, 1, size=(21, 3))
    auc, pck, _ = metrics.pck_auc(gt, gt, thresholds=np.linspace(0, 50, 20))
    assert np.isclose(auc, 1.0)


def test_umeyama_recovers_transform():
    rng = np.random.default_rng(3)
    src = rng.uniform(-1, 1, size=(30, 3))
    theta = 0.4
    R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    dst = (1.7 * (R_true @ src.T)).T + np.array([2.0, 1.0, -1.0])
    R, t, s = metrics.umeyama(src, dst, with_scale=True)
    assert np.isclose(s, 1.7, atol=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)


def _write_tum(path, positions):
    lines = []
    for t, p in enumerate(positions):
        lines.append(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 0 0 1")
    path.write_text("\n".join(lines) + "\n")


def test_trajectory_metrics_sim3_vs_se3(tmp_path):
    n = 20
    gt = np.stack([np.linspace(0, 1, n), np.sin(np.linspace(0, 3, n)), np.zeros(n)], axis=1)
    est = gt * 2.0 + np.array([1.0, -1.0, 0.5])  # scaled + translated

    gt_p = tmp_path / "gt.tum"
    est_p = tmp_path / "est.tum"
    _write_tum(gt_p, gt)
    _write_tum(est_p, est)

    m = metrics.trajectory_metrics(str(est_p), str(gt_p))
    # Sim3 alignment removes the scale -> near-zero ATE
    assert m["ate_sim3_7dof"] < 1e-6
    # SE3 (scale-free) cannot remove the 2x scale -> larger error
    assert m["ate_s_se3_6dof"] > m["ate_sim3_7dof"]
