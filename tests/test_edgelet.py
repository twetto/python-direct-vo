import numpy as np
import sophuspy as sp

from src.edgelet import optimize_pose_2d1d, align1D, edgelet_ref_patch, detect_edgelets


K = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])


def _project(K, T_cw, pts_w):
    pc = pts_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)


def _random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(4, 8, n)])


def test_pose_recovers_from_perturbation_corners():
    pts = _random_points(60)
    T_true = np.eye(4)
    T_true[:3, 3] = [0.15, -0.1, 0.2]
    T_true[:3, :3] = sp.SO3.exp([0.03, -0.05, 0.02]).matrix()
    meas = _project(K, T_true, pts)
    is_edge = np.zeros(len(pts), dtype=bool)
    normals = np.zeros((len(pts), 2))
    T_init = np.eye(4)
    T_out, inl = optimize_pose_2d1d(K, pts, meas, is_edge, normals, T_init, max_iters=30)
    assert np.allclose(T_out, T_true, atol=1e-3)
    assert inl.all()


def test_edgelets_ignore_along_edge_slide():
    # every feature is an edgelet; corrupt each measurement by a large shift ALONG
    # the edge (perpendicular to the normal). A correct 1D point-to-line solver must
    # be unaffected and still recover the true pose.
    pts = _random_points(80, seed=1)
    T_true = np.eye(4)
    T_true[:3, 3] = [0.1, 0.05, -0.15]
    T_true[:3, :3] = sp.SO3.exp([-0.02, 0.04, -0.03]).matrix()
    meas = _project(K, T_true, pts)
    rng = np.random.default_rng(2)
    normals = rng.normal(size=(len(pts), 2))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    slid = meas + tangents * rng.uniform(-6, 6, len(pts))[:, None]  # slide along edge
    is_edge = np.ones(len(pts), dtype=bool)
    T_out, inl = optimize_pose_2d1d(K, pts, slid, is_edge, normals, np.eye(4), max_iters=40)
    assert np.allclose(T_out[:3, 3], T_true[:3, 3], atol=5e-3)
    assert np.allclose(T_out[:3, :3], T_true[:3, :3], atol=5e-3)


def test_mixed_corners_and_edgelets():
    pts = _random_points(100, seed=3)
    T_true = np.eye(4)
    T_true[:3, 3] = [-0.12, 0.08, 0.1]
    T_true[:3, :3] = sp.SO3.exp([0.05, 0.02, -0.04]).matrix()
    meas = _project(K, T_true, pts)
    is_edge = np.zeros(len(pts), dtype=bool)
    is_edge[::2] = True
    rng = np.random.default_rng(4)
    normals = np.zeros((len(pts), 2))
    en = rng.normal(size=(is_edge.sum(), 2))
    en /= np.linalg.norm(en, axis=1, keepdims=True)
    normals[is_edge] = en
    tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    m = meas.copy()
    m[is_edge] += tangents[is_edge] * rng.uniform(-5, 5, is_edge.sum())[:, None]
    T_out, inl = optimize_pose_2d1d(K, pts, m, is_edge, normals, np.eye(4), max_iters=40)
    assert np.allclose(T_out[:3, 3], T_true[:3, 3], atol=5e-3)


def test_align1d_recovers_perpendicular_shift():
    rng = np.random.default_rng(5)
    img = rng.uniform(40, 60, (80, 80)).astype(np.float32)
    # vertical edge at x=40 -> gradient normal is +x
    img[:, 40:] += 80
    img = np.clip(img, 0, 255)
    direction = np.array([1.0, 0.0])  # normal across the vertical edge
    ref, ddir = edgelet_ref_patch(img, (40.0, 40.0), direction, half=5)
    # start 2 px off across the edge; align1D should pull back to x=40
    px, conv = align1D(img, direction, ref, ddir, np.array([38.0, 40.0]), max_iter=30)
    assert conv
    assert abs(px[0] - 40.0) < 0.5
    assert abs(px[1] - 40.0) < 1e-6  # never moves along the edge (v fixed)


def test_detect_edgelets_finds_edges_and_normals():
    img = np.full((240, 320), 50, dtype=np.uint8)
    img[:, 160:] = 200  # single strong vertical edge
    mask = np.full(img.shape, 255, dtype=np.uint8)
    pos, nrm = detect_edgelets(img, mask, need=50, min_distance=8, cell=12)
    assert len(pos) > 0
    # detected points sit on the edge column and normals point across it (mostly +/-x)
    assert np.all(np.abs(pos[:, 0] - 160) <= 3)
    assert np.mean(np.abs(nrm[:, 0]) > np.abs(nrm[:, 1])) > 0.9
