"""Scale-aware trajectory evaluation for monocular VO.

Phase 0 of docs/mono_simplification_plan.md: a measuring stick that must exist
before any estimator is changed. Monocular VO has an arbitrary global scale, so
every metric here is computed under a Sim(3) (scale-free) alignment of the
estimated trajectory to ground truth, and the recovered scale is reported
*separately* so scale error is never hidden inside a scale-corrected ATE.

Pure numpy; no dataset or OpenCV dependency, so the math is unit-testable in
isolation.
"""

from __future__ import annotations

import json

import numpy as np


def umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) from X to Y: returns (s, R, t) with Y ~= s * R @ X + t.

    Umeyama (1991). X, Y are (N, 3) point sets in correspondence.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X and Y must be matching (N, 3) arrays")
    n = X.shape[0]
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    var_x = float(np.mean(np.sum(Xc * Xc, axis=1)))
    Sigma_xy = (Yc.T @ Xc) / n  # E[(y - mu_y)(x - mu_x)^T]

    U, D, Vt = np.linalg.svd(Sigma_xy)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    s = 1.0 if var_x < 1e-12 else float(np.trace(np.diag(D) @ S) / var_x)
    t = mu_y - s * R @ mu_x
    return s, R, t


def align(est: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Sim(3)-align est onto gt. Returns (aligned_est, s, R, t)."""
    s, R, t = umeyama_sim3(est, gt)
    aligned = (s * (R @ np.asarray(est, dtype=np.float64).T)).T + t
    return aligned, s, R, t


def ate(est: np.ndarray, gt: np.ndarray) -> dict:
    """Absolute trajectory error under Sim(3) alignment (scale-corrected)."""
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    aligned, s, _R, _t = align(est, gt)
    err = np.linalg.norm(aligned - gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))) if len(err) else 0.0,
        "mean": float(np.mean(err)) if len(err) else 0.0,
        "median": float(np.median(err)) if len(err) else 0.0,
        "max": float(np.max(err)) if len(err) else 0.0,
        "scale": float(s),
    }


def local_scale_ratios(est: np.ndarray, gt: np.ndarray, window: int = 10, min_gt_motion: float = 1e-3) -> np.ndarray:
    """Per-window arc-length ratio est/gt, a scale-drift probe over time.

    Path length is rotation/translation invariant, so this needs no alignment. A
    constant value means consistent scale; a trend means scale drift.
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if len(est) < 2 or len(est) != len(gt):
        return np.empty((0,), dtype=np.float64)
    d_est = np.linalg.norm(np.diff(est, axis=0), axis=1)
    d_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    window = max(1, int(window))
    ratios = []
    for start in range(0, len(d_est), window):
        seg_est = float(np.sum(d_est[start:start + window]))
        seg_gt = float(np.sum(d_gt[start:start + window]))
        if seg_gt > min_gt_motion:
            ratios.append(seg_est / seg_gt)
    return np.asarray(ratios, dtype=np.float64)


def scale_drift(est: np.ndarray, gt: np.ndarray, window: int = 10) -> dict:
    """Summarize scale-drift over time from local arc-length ratios.

    Windows where the estimate did not move at all (ratio == 0) are a tracking
    *stall*, not scale drift; they are counted separately (`stall_windows`) and
    excluded from the log-based statistics so the metrics stay finite.
    """
    ratios = local_scale_ratios(est, gt, window=window)
    if len(ratios) == 0:
        return {"windows": 0, "stall_windows": 0, "log_std": 0.0, "min": 0.0, "max": 0.0, "spread": 0.0}
    pos = ratios[ratios > 0]
    stall = int(np.sum(ratios <= 0))
    log_std = float(np.std(np.log(pos))) if len(pos) >= 2 else 0.0
    spread = float(np.max(pos) / np.min(pos)) if len(pos) else 0.0
    return {
        "windows": int(len(ratios)),
        "stall_windows": stall,
        "log_std": log_std,                     # 0 == perfectly consistent scale
        "min": float(np.min(pos)) if len(pos) else 0.0,
        "max": float(np.max(pos)) if len(pos) else 0.0,
        "spread": spread,
    }


def evaluate(est: np.ndarray, gt: np.ndarray, window: int = 10) -> dict:
    """The ~6 core numbers that diagnose drift (see plan Phase 0)."""
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]
    if n < 3:
        return {"n_poses": int(n), "ate_rmse": 0.0, "scale_ratio_vs_gt": 1.0,
                "scale_drift_log_std": 0.0, "scale_drift_spread": 0.0,
                "stall_windows": 0, "gt_path_length": 0.0}
    ate_stats = ate(est, gt)
    drift = scale_drift(est, gt, window=window)
    gt_path = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    return {
        "n_poses": int(n),
        "ate_rmse": ate_stats["rmse"],
        "scale_ratio_vs_gt": ate_stats["scale"],
        "scale_drift_log_std": drift["log_std"],
        "scale_drift_spread": drift["spread"],
        "stall_windows": drift["stall_windows"],
        "gt_path_length": gt_path,
    }


def load_trajectories_from_jsonl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read (estimated, gt) position arrays from a diagnostics JSONL run.

    Only rows carrying both 'position' and 'gt_position' are used.
    """
    est, gt = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pos = rec.get("position")
            gt_pos = rec.get("gt_position")
            if pos is None or gt_pos is None:
                continue
            est.append(pos)
            gt.append(gt_pos)
    return (
        np.asarray(est, dtype=np.float64).reshape(-1, 3),
        np.asarray(gt, dtype=np.float64).reshape(-1, 3),
    )


def format_report(metrics: dict) -> str:
    return (
        "[mono-eval] "
        f"n={metrics['n_poses']} "
        f"ATE(sim3)={metrics['ate_rmse']:.4f}m "
        f"scale_vs_gt={metrics['scale_ratio_vs_gt']:.3f} "
        f"drift_log_std={metrics['scale_drift_log_std']:.3f} "
        f"drift_spread={metrics['scale_drift_spread']:.2f} "
        f"stalls={metrics.get('stall_windows', 0)} "
        f"gt_path={metrics['gt_path_length']:.2f}m"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a mono VO diagnostics JSONL against GT.")
    parser.add_argument("jsonl", help="diagnostics JSONL written by main.py --diagnostics-log")
    parser.add_argument("--window", type=int, default=10, help="scale-drift window in frames")
    args = parser.parse_args(argv)

    est, gt = load_trajectories_from_jsonl(args.jsonl)
    if len(est) < 3:
        print(f"[mono-eval] not enough paired poses in {args.jsonl} (got {len(est)})")
        return 1
    print(format_report(evaluate(est, gt, window=args.window)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
