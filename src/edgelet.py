"""SVO-style edgelet support for the monocular pipeline.

Three pieces, ported from rpg_svo_pro_open:
  * `detect_edgelets`   -- gradient-magnitude feature detection with edge normals,
                           to fill the feature budget where corners are absent
                           (feature_detection_utils.cpp edgeletDetector_V2).
  * `align1D`           -- inverse-compositional 1D patch alignment along the edge
                           normal (feature_alignment.cpp align1D). The along-edge
                           direction (aperture-ambiguous) is never moved.
  * `optimize_pose_2d1d`-- robust Gauss-Newton pose refinement mixing 2D point
                           residuals (corners) and 1D point-to-line residuals
                           (edgelets: e = n . (proj - meas)); pose_optimizer.cpp
                           calculateEdgeletResidual*.
"""
import cv2
import numpy as np
import scipy.ndimage as nd
import sophuspy as sp


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def optimize_pose_2d1d(
    K, pts_w, meas_px, is_edge, normals, T_cw_init,
    max_iters=10, huber_px=2.0, edge_weight=0.5, reproj_inlier_px=4.0,
):
    """Robust GN refinement of world->cam pose (left perturbation T <- exp(d) T).

    Corners contribute a 2D reprojection residual (proj - meas); edgelets contribute
    the 1D point-to-line residual n . (proj - meas), so an edge feature that slid
    along its edge under KLT constrains only the reliable perpendicular direction.
    Returns (T_cw (4,4), inlier_mask (N,)).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_w = np.asarray(pts_w, dtype=np.float64)
    meas_px = np.asarray(meas_px, dtype=np.float64)
    is_edge = np.asarray(is_edge, dtype=bool)
    normals = np.asarray(normals, dtype=np.float64)
    n = len(pts_w)
    T = sp.SE3(np.asarray(T_cw_init, dtype=np.float64))
    inl = np.ones(n, dtype=bool)
    if n == 0:
        return T.matrix(), inl

    for _ in range(max_iters):
        R = T.rotationMatrix()
        t = T.translation()
        pc = pts_w @ R.T + t
        Z = pc[:, 2]
        valid = Z > 1e-3
        Zs = np.where(valid, Z, 1.0)
        proj = np.stack([fx * pc[:, 0] / Zs + cx, fy * pc[:, 1] / Zs + cy], axis=1)
        r = proj - meas_px  # (n,2)

        # pixel Jacobian wrt pc (n,2,3), then wrt se3 left perturbation [I | -skew(pc)]
        Zi = 1.0 / Zs
        Jpc = np.zeros((n, 2, 3))
        Jpc[:, 0, 0] = fx * Zi
        Jpc[:, 0, 2] = -fx * pc[:, 0] * Zi * Zi
        Jpc[:, 1, 1] = fy * Zi
        Jpc[:, 1, 2] = -fy * pc[:, 1] * Zi * Zi
        D = np.zeros((n, 3, 6))
        D[:, :, 0:3] = np.eye(3)
        for i in range(n):
            D[i, :, 3:6] = -_skew(pc[i])
        J = Jpc @ D  # (n,2,6)

        H = np.zeros((6, 6))
        g = np.zeros(6)
        use = valid & inl
        for i in np.flatnonzero(use):
            if is_edge[i]:
                Ji = normals[i] @ J[i]          # (6,)
                ei = float(normals[i] @ r[i])   # scalar point-to-line error
                w = edge_weight
                hw = 1.0 if abs(ei) < huber_px else huber_px / abs(ei)
                H += (w * hw) * np.outer(Ji, Ji)
                g += (w * hw) * Ji * ei
            else:
                Ji = J[i]                        # (2,6)
                ei = r[i]                        # (2,)
                mag = np.linalg.norm(ei)
                hw = 1.0 if mag < huber_px else huber_px / max(mag, 1e-9)
                H += hw * (Ji.T @ Ji)
                g += hw * (Ji.T @ ei)

        try:
            delta = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
        except np.linalg.LinAlgError:
            break
        T = sp.SE3.exp(delta) * T
        if float(delta @ delta) < 1e-12:
            break

    # final inlier classification
    R = T.rotationMatrix()
    t = T.translation()
    pc = pts_w @ R.T + t
    Z = pc[:, 2]
    valid = Z > 1e-3
    Zs = np.where(valid, Z, 1.0)
    proj = np.stack([fx * pc[:, 0] / Zs + cx, fy * pc[:, 1] / Zs + cy], axis=1)
    r = proj - meas_px
    err = np.where(is_edge, np.abs(np.sum(normals * r, axis=1)), np.linalg.norm(r, axis=1))
    inl = valid & (err <= reproj_inlier_px)
    return T.matrix(), inl


def align1D(cur_img, direction, ref_patch, ref_ddir, init_px, max_iter=10):
    """Inverse-compositional 1D alignment: slide the reference patch along `direction`
    (unit edge normal) to match `cur_img` near `init_px`. Estimates a brightness
    offset too. Returns (px (2,), converged). `ref_ddir` is the reference patch's
    directional derivative 0.5*(dx*Gx + dy*Gy) precomputed by `edgelet_ref_patch`.
    """
    direction = np.asarray(direction, dtype=np.float64)
    P = ref_patch.shape[0]
    half = P // 2
    u, v = float(init_px[0]), float(init_px[1])
    mean = 0.0
    # 2x2 Hessian for [along-dir shift, brightness offset], from reference derivatives
    Jdir = ref_ddir.reshape(-1)
    ones = np.ones_like(Jdir)
    H = np.array([[np.sum(Jdir * Jdir), np.sum(Jdir * ones)],
                  [np.sum(ones * Jdir), np.sum(ones * ones)]])
    try:
        Hinv = np.linalg.inv(H + 1e-6 * np.eye(2))
    except np.linalg.LinAlgError:
        return np.array([u, v]), False
    oy, ox = np.mgrid[-half:P - half, -half:P - half]
    ox = ox.reshape(-1).astype(np.float64)
    oy = oy.reshape(-1).astype(np.float64)
    ref = ref_patch.reshape(-1).astype(np.float64)
    converged = False
    for _ in range(max_iter):
        cur = nd.map_coordinates(cur_img, [v + oy, u + ox], order=1)
        res = cur - ref - mean
        Jres = np.array([np.sum(res * Jdir), np.sum(res)])
        step = Hinv @ Jres
        u -= step[0] * direction[0]
        v -= step[0] * direction[1]
        mean -= step[1]
        if step[0] * step[0] < 0.03 * 0.03:
            converged = True
            break
    return np.array([u, v]), converged


def edgelet_ref_patch(image, px, direction, half=4):
    """Extract a reference patch and its directional derivative at `px` for align1D."""
    image = image.astype(np.float32)
    P = 2 * half
    oy, ox = np.mgrid[-half:P - half, -half:P - half]
    ox = ox.reshape(-1).astype(np.float64)
    oy = oy.reshape(-1).astype(np.float64)
    u, v = float(px[0]), float(px[1])
    patch = nd.map_coordinates(image, [v + oy, u + ox], order=1)
    gx = 0.5 * (nd.map_coordinates(image, [v + oy, u + ox + 1], order=1)
                - nd.map_coordinates(image, [v + oy, u + ox - 1], order=1))
    gy = 0.5 * (nd.map_coordinates(image, [v + oy + 1, u + ox], order=1)
                - nd.map_coordinates(image, [v + oy - 1, u + ox], order=1))
    ddir = direction[0] * gx + direction[1] * gy
    return patch.reshape(P, P), ddir.reshape(P, P)


def detect_edgelets(image, mask, need, min_distance=12, mag_thresh=None, cell=12):
    """Detect edge features where corners are scarce. Returns (positions (M,2),
    normals (M,2) unit gradient direction). Grid non-max on Scharr gradient magnitude,
    honoring `mask` (0 = excluded) so edgelets don't collide with existing features.
    """
    img = image.astype(np.float32)
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0) / 16.0
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1) / 16.0
    mag = np.sqrt(gx * gx + gy * gy)
    if mask is not None:
        mag = np.where(mask > 0, mag, 0.0)
    if mag_thresh is None:
        mag_thresh = float(np.percentile(mag[mag > 0], 80)) if np.any(mag > 0) else 0.0
    h, w = img.shape
    b = min_distance
    positions = []
    normals = []
    # one candidate per grid cell: the local gradient-magnitude maximum
    for y0 in range(b, h - b, cell):
        for x0 in range(b, w - b, cell):
            sub = mag[y0:y0 + cell, x0:x0 + cell]
            if sub.size == 0:
                continue
            iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
            m = sub[iy, ix]
            if m < mag_thresh:
                continue
            y, x = y0 + iy, x0 + ix
            g = np.array([gx[y, x], gy[y, x]], dtype=np.float64)
            ng = np.linalg.norm(g)
            if ng < 1e-6:
                continue
            positions.append([float(x), float(y)])
            normals.append((g / ng).tolist())
            if len(positions) >= need:
                return np.asarray(positions), np.asarray(normals)
    if not positions:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.asarray(positions), np.asarray(normals)
