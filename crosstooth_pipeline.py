"""
crosstooth_pipeline.py
----------------------
Geometry side of the CrossTooth Runpod worker (no torch here, so it can be
unit-tested on CPU).  Ported from the Colab notebook:

  STL -> orient (axis permutation, mirror fix, flip_z, flip_y)
      -> quadric decimation to ~16k faces           ("la decimada")
      -> [CrossTooth inference lives in handler.py]
      -> labels back onto the full-resolution mesh   ("la vuelta", KNN k=3)
      -> PLY with one RGB colour per face
"""
from __future__ import annotations

import numpy as np
import vedo
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_mesh(path: str):
    """Load an STL/PLY/OBJ and return (points float64 (N,3), faces int64 (M,3))."""
    mesh = vedo.load(path)
    if mesh is None:
        raise ValueError("vedo could not read the mesh file")
    mesh = mesh.triangulate()
    pts = np.asarray(mesh.points(), dtype=np.float64)
    faces = np.asarray(mesh.cells(), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("mesh has no triangular faces")
    if not np.isfinite(pts).all():
        raise ValueError("mesh contains non-finite vertex coordinates")
    return pts, faces


# --------------------------------------------------------------------------- #
# Orientation  ("la vuelta")
# --------------------------------------------------------------------------- #

def _edge_keys(faces: np.ndarray, n_pts: int) -> np.ndarray:
    """Undirected edge keys (one int64 per face-edge, 3 per face)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    return e[:, 0].astype(np.int64) * n_pts + e[:, 1].astype(np.int64)


def boundary_vertices(faces: np.ndarray, n_pts: int) -> np.ndarray:
    """Vertices lying on edges that belong to exactly one face (open boundary)."""
    keys = _edge_keys(faces, n_pts)
    uniq, counts = np.unique(keys, return_counts=True)
    b = uniq[counts == 1]
    return np.unique(np.concatenate([b // n_pts, b % n_pts]))


def face_normals_from_winding(pts: np.ndarray, faces: np.ndarray, normalize=True) -> np.ndarray:
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    if normalize:
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    return fn


def detect_crown_direction(p: np.ndarray, faces: np.ndarray):
    """
    Returns (+1 | -1, method, extra) : sign of Z where the crowns (occlusal
    surfaces) are, in the already-permuted frame `p`.

    1. open boundary : an intra-oral scan is an open surface whose boundary
       loop runs along the gingival / palatal cut, i.e. the side opposite to
       the crowns.  Independent of face winding.
    2. flat base     : closed meshes (printed models) have a big planar base.
    3. mean normal   : for an open surface with outward normals the total
       vector area points towards the crowns.
    """
    z = p[:, 2]
    zr = float(np.ptp(z)) + 1e-9
    bv = boundary_vertices(faces, len(p))
    extra = {"n_boundary_vertices": int(len(bv))}
    if len(bv) >= 30:
        zb = float(np.median(z[bv]))
        zc = float(np.median(z))
        extra.update({"boundary_z_median": zb, "mesh_z_median": zc})
        if abs(zc - zb) > 0.05 * zr:
            return (1 if zc > zb else -1), "boundary", extra

    fn = face_normals_from_winding(p, faces, normalize=False)      # |fn| = 2*area
    area = np.linalg.norm(fn, axis=1) + 1e-12
    total = float(area.sum())
    mean_nz = float(fn[:, 2].sum())
    extra["mean_normal_z_ratio"] = mean_nz / total
    if abs(mean_nz) > 0.10 * total:                                # open surface, consistent winding
        return (1 if mean_nz > 0 else -1), "mean_normal", extra

    nz = fn[:, 2] / area
    flat_neg = float(area[nz < -0.98].sum())
    flat_pos = float(area[nz > 0.98].sum())
    extra["flat_area_ratio_neg"] = flat_neg / total
    extra["flat_area_ratio_pos"] = flat_pos / total
    if max(flat_neg, flat_pos) > 0.08 * total:                     # closed model with a planar base
        # base normal points away from the model -> crowns on the other side
        return (1 if flat_neg > flat_pos else -1), "flat_base", extra

    extra["warning"] = "could not determine crown side; assumed +Z. Pass flip_z=true/false to force it."
    return 1, "default", extra


def detect_incisors_at_negative_y(p: np.ndarray):
    """
    The dental arch is a U: the closed end (incisors) is narrow in X, the open
    end (molars) is wide.  Returns (bool incisors_at_negative_y, extra).
    """
    y = p[:, 1]
    ylo, yhi = np.percentile(y, [1, 99])
    band = 0.15 * (yhi - ylo)
    front = p[y < ylo + band]
    back = p[y > yhi - band]

    def spread(a):
        if len(a) < 10:
            return 0.0
        q5, q95 = np.percentile(a[:, 0], [5, 95])
        return float(q95 - q5)

    s_lo, s_hi = spread(front), spread(back)
    return s_lo < s_hi, {"x_spread_at_ymin": s_lo, "x_spread_at_ymax": s_hi}


def orientation_matrix(pts: np.ndarray, faces: np.ndarray, flip_z=None, flip_y=None):
    """
    Build the 3x3 matrix T (proper rotation, det=+1) that takes the STL into
    the frame CrossTooth expects:   p_model = p_stl @ T
        X : longest extent (left-right), Y : middle (anterior-posterior),
        Z : shortest, crowns towards +Z, incisors towards -Y.
    flip_z / flip_y : None = auto-detect, True/False = force (like the notebook).
    """
    info: dict = {}
    widths = pts.max(0) - pts.min(0)
    order = np.argsort(widths)                        # [shortest, middle, longest]
    new_order = [int(order[2]), int(order[1]), int(order[0])]
    M = np.zeros((3, 3))
    for j, k in enumerate(new_order):
        M[k, j] = 1.0
    mirrored = bool(np.linalg.det(M) < 0)
    if mirrored:
        M[:, 0] *= -1.0                               # keep it a rotation, no mirror
    info["axis_order_xyz_from"] = new_order
    info["mirror_fixed"] = mirrored
    info["extents_mm"] = [float(w) for w in widths]

    p = pts @ M

    if flip_z is None:
        crown_dir, method, extra = detect_crown_direction(p, faces)
        flip_z = crown_dir < 0
        info["flip_z_method"] = method
        info["flip_z_details"] = extra
    else:
        flip_z = bool(flip_z)
        info["flip_z_method"] = "manual"
    if flip_z:
        R = np.diag([1.0, -1.0, -1.0])                # 180 deg about X
        M = M @ R
        p = p @ R
    info["flip_z"] = bool(flip_z)

    if flip_y is None:
        ok, extra = detect_incisors_at_negative_y(p)
        flip_y = not ok
        info["flip_y_method"] = "arch_width"
        info["flip_y_details"] = extra
    else:
        flip_y = bool(flip_y)
        info["flip_y_method"] = "manual"
    if flip_y:
        R = np.diag([-1.0, -1.0, 1.0])                # 180 deg about Z
        M = M @ R
        p = p @ R
    info["flip_y"] = bool(flip_y)

    info["transform"] = M.tolist()
    return M, info


# --------------------------------------------------------------------------- #
# Decimation ("la decimada")
# --------------------------------------------------------------------------- #

def decimate_mesh(pts: np.ndarray, faces: np.ndarray, target_faces=15999, max_faces=16000):
    """Quadric decimation to ~target_faces (never more than max_faces)."""
    mesh = vedo.Mesh([pts, faces])
    if mesh.ncells > target_faces:
        mesh = mesh.decimate(fraction=target_faces / mesh.ncells, method="quadric")
        tries = 0
        while mesh.ncells > max_faces and tries < 3:      # quadric can overshoot a little
            mesh = mesh.decimate(fraction=target_faces / mesh.ncells, method="quadric")
            tries += 1
        if mesh.ncells > max_faces:
            raise RuntimeError(f"decimation left {mesh.ncells} faces (> {max_faces})")
    mesh.compute_normals()
    return mesh


# --------------------------------------------------------------------------- #
# Back-projection ("la vuelta" to HD)
# --------------------------------------------------------------------------- #

def project_labels(centers_low: np.ndarray, labels_low: np.ndarray, centers_hd: np.ndarray, k=3):
    """Notebook step: KNeighborsClassifier(n_neighbors=3) on face centres."""
    if len(np.unique(labels_low)) < 2:
        return np.full(len(centers_hd), int(labels_low[0]) if len(labels_low) else 0, dtype=np.int64)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(labels_low)))
    knn.fit(centers_low, labels_low)
    return knn.predict(centers_hd).astype(np.int64)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_coloured_ply(path: str, pts: np.ndarray, faces: np.ndarray, rgb: np.ndarray, binary=True):
    """Write a PLY with one RGB colour per face (what the notebook saves)."""
    mesh = vedo.Mesh([pts, faces])
    rgba = np.hstack([rgb.astype(np.uint8), np.full((len(rgb), 1), 255, np.uint8)])
    mesh.cellcolors = rgba
    mesh.write(path, binary=binary)
    return path
