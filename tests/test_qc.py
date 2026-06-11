from __future__ import annotations

import numpy as np

from aoe_pipeline import qc


def test_kinematic_outlier_flags_velocity_spike():
    T = 40
    jw = np.zeros((T, 2, 21, 3), np.float64)
    drift = np.linspace(0, 1, T)
    for j in range(21):
        jw[:, 0, j, 0] = drift * 0.01 * (j + 1)  # tiny smooth motion
    jw[:, 1] = np.nan                            # right hand absent -> never flagged
    jw[20, 0, 5, 0] += 0.5                        # inject a spike on one joint

    flags, vel = qc.kinematic_outliers(jw, dt=1 / 30, sigma=3.0)

    assert flags[:, 0, 5].any()           # the spiking joint is flagged
    assert flags[:, 1].sum() == 0         # NaN slot never flagged
    # a non-spiking joint should be clean
    assert flags[:, 0, 10].sum() == 0


def test_reprojection_error_consistency_and_outlier():
    K = np.array([[500.0, 0, 160], [0, 500.0, 120], [0, 0, 1.0]])
    rng = np.random.default_rng(0)
    cam = rng.uniform(-0.1, 0.1, size=(21, 3)) + np.array([0, 0, 1.0])
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    T = 5
    jw = np.full((T, 2, 21, 3), np.nan)
    j2d = np.full((T, 2, 21, 2), np.nan)
    for t in range(T):
        jw[t, 0] = cam
        j2d[t, 0] = uv
    poses = [np.eye(4) for _ in range(T)]

    err = qc.reprojection_error(jw, j2d, poses, K)
    assert np.nanmax(err[:, 0]) < 1e-6                     # consistent by construction
    assert qc.reprojection_outliers(err, px=5.0).sum() == 0

    j2d[2, 0, 3] += np.array([20.0, 20.0])                 # perturb one detection
    err2 = qc.reprojection_error(jw, j2d, poses, K)
    flags2 = qc.reprojection_outliers(err2, px=5.0)
    assert flags2[2, 0, 3]
    assert flags2.sum() == 1


def test_frames_flagged():
    flags = np.zeros((4, 2, 21), bool)
    flags[1, 0, 3] = True
    flags[3, 1, 0] = True
    ff = qc.frames_flagged(flags)
    assert ff.tolist() == [False, True, False, True]
