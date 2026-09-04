import json

import numpy as np

from src.mono_eval import (
    ate,
    evaluate,
    load_trajectories_from_jsonl,
    local_scale_ratios,
    scale_drift,
    umeyama_sim3,
)


def _random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_umeyama_sim3_recovers_known_similarity():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    s_true = 2.5
    R_true = _random_rotation(1)
    t_true = np.array([3.0, -1.0, 0.5])
    Y = (s_true * (R_true @ X.T)).T + t_true

    s, R, t = umeyama_sim3(X, Y)
    assert np.isclose(s, s_true, atol=1e-9)
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_umeyama_sim3_scale_free_recovers_scale():
    # Estimated trajectory is half the size of GT -> must be scaled by 2 to match.
    rng = np.random.default_rng(2)
    gt = np.cumsum(rng.standard_normal((40, 3)), axis=0)
    est = 0.5 * gt
    metrics = evaluate(est, gt)
    assert np.isclose(metrics["scale_ratio_vs_gt"], 2.0, atol=1e-6)
    # ATE is scale-corrected, so a pure scale error still aligns to ~0.
    assert metrics["ate_rmse"] < 1e-6


def test_ate_zero_for_identical_trajectory():
    rng = np.random.default_rng(3)
    gt = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    stats = ate(gt.copy(), gt.copy())
    assert stats["rmse"] < 1e-9
    assert np.isclose(stats["scale"], 1.0, atol=1e-9)


def test_ate_invariant_to_rigid_transform():
    rng = np.random.default_rng(4)
    gt = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    R = _random_rotation(5)
    t = np.array([10.0, -5.0, 2.0])
    est = (R @ gt.T).T + t  # rigidly displaced -> ATE must vanish after alignment
    assert ate(est, gt)["rmse"] < 1e-6


def test_scale_drift_flat_for_consistent_scale():
    rng = np.random.default_rng(6)
    gt = np.cumsum(np.abs(rng.standard_normal((60, 3))) + 0.1, axis=0)
    est = 1.3 * gt
    drift = scale_drift(est, gt, window=10)
    assert drift["windows"] > 0
    assert drift["log_std"] < 1e-6           # constant scale factor -> no drift
    assert np.isclose(drift["spread"], 1.0, atol=1e-6)


def test_scale_drift_detects_growing_scale():
    # Each GT step is unit length; est steps grow, so local scale ratio climbs.
    steps_gt = np.zeros((60, 3))
    steps_gt[:, 0] = 1.0
    growth = np.linspace(1.0, 3.0, 60).reshape(-1, 1)
    steps_est = steps_gt * growth
    gt = np.cumsum(steps_gt, axis=0)
    est = np.cumsum(steps_est, axis=0)

    ratios = local_scale_ratios(est, gt, window=10)
    assert ratios[-1] > ratios[0] + 0.5      # scale clearly drifts upward
    assert scale_drift(est, gt, window=10)["log_std"] > 0.1


def test_load_trajectories_from_jsonl_round_trip(tmp_path):
    path = tmp_path / "diag.jsonl"
    est_rows = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    gt_rows = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    with open(path, "w", encoding="utf-8") as f:
        for e, g in zip(est_rows, gt_rows):
            f.write(json.dumps({"frame": 0, "position": e, "gt_position": g}) + "\n")
        # a row missing gt_position must be skipped, not crash
        f.write(json.dumps({"frame": 99, "position": [9.0, 9.0, 9.0]}) + "\n")

    est, gt = load_trajectories_from_jsonl(str(path))
    assert est.shape == (3, 3)
    assert gt.shape == (3, 3)
    assert np.allclose(est, est_rows)
    assert np.allclose(gt, gt_rows)


def test_scale_drift_stall_stays_finite():
    # Estimator stalls (no motion) while GT keeps moving: ratio 0 windows must not
    # leak nan/inf into the metrics. Regression for the Phase-0 baseline run.
    steps_gt = np.zeros((60, 3))
    steps_gt[:, 0] = 1.0
    steps_est = steps_gt.copy()
    steps_est[30:] = 0.0  # dead second half
    gt = np.cumsum(steps_gt, axis=0)
    est = np.cumsum(steps_est, axis=0)

    drift = scale_drift(est, gt, window=10)
    assert np.isfinite(drift["log_std"])
    assert np.isfinite(drift["spread"])
    assert drift["stall_windows"] > 0

    metrics = evaluate(est, gt)
    assert np.isfinite(metrics["ate_rmse"])
    assert np.isfinite(metrics["scale_drift_log_std"])
    assert np.isfinite(metrics["scale_drift_spread"])


def test_evaluate_handles_too_few_poses():
    metrics = evaluate(np.zeros((2, 3)), np.zeros((2, 3)))
    assert metrics["n_poses"] == 2
    assert metrics["ate_rmse"] == 0.0
