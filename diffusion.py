import numpy as np
from skimage import measure
import triangle as tr
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

# ---------- Geometry & Triangulation from mask ----------

def smooth_closed_curve(points, window_size=9):
    """
    Simple moving-average smoothing of a closed 2D curve.
    """
    if window_size < 2:
        return points.copy()

    pts = np.asarray(points)
    n = len(pts)
    k = window_size // 2

    padded = np.vstack([pts[-k:], pts, pts[:k]])
    kernel = np.ones(window_size) / window_size

    x_smooth = np.convolve(padded[:, 0], kernel, mode="valid")
    y_smooth = np.convolve(padded[:, 1], kernel, mode="valid")

    smoothed = np.column_stack([x_smooth, y_smooth])
    assert smoothed.shape[0] == n
    return smoothed

def resample_closed_curve(points, n_samples):
    """
    Resample a closed polyline to n_samples points with approximately
    equal arc-length spacing.
    """
    pts = np.asarray(points)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
    total_len = cumlen[-1]

    s_new = np.linspace(0.0, total_len, n_samples + 1)[:-1]

    x_new = np.interp(s_new, cumlen, pts[:, 0])
    y_new = np.interp(s_new, cumlen, pts[:, 1])
    return np.column_stack([x_new, y_new])

def triangulate_from_mask(
    mask,
    target_max_area=1.0,
    smooth_window=7,
    min_boundary_points=160,
    ):
    """
    Triangulate a rectangular canvas with:
      - outer rectangle boundary,
      - mask boundary as an inner constrained polygon.

    mask: 2D array (H, W), nonzero = "inside blob".

    Returns:
        tri_vertices: (Nv, 2)
        triangles:    (Nt, 3)
        mask:         (H, W) as bool
    """
    canvas = np.asarray(mask)
    h, w = canvas.shape
    mask_bool = canvas.astype(bool)

    # Extract the largest contour of the mask
    contours = measure.find_contours(mask_bool, level=0.5)
    if len(contours) == 0:
        raise RuntimeError("No contour found in mask.")
    contour = max(contours, key=len)          # (N, 2) in (row, col)
    boundary_pts = np.fliplr(contour).astype(float)  # (N, 2) in (x, y)

    # Smooth the boundary
    boundary_pts = smooth_closed_curve(boundary_pts, window_size=smooth_window)

    # Choose boundary point density to roughly match desired triangle size
    target_edge = np.sqrt(4.0 * target_max_area / np.sqrt(3.0))
    pts_closed = np.vstack([boundary_pts, boundary_pts[0]])
    seg = np.diff(pts_closed, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    perimeter = seglen.sum()
    n_boundary = int(perimeter / target_edge)
    n_boundary = max(n_boundary, min_boundary_points)

    boundary_pts = resample_closed_curve(boundary_pts, n_boundary)
    n_inner = len(boundary_pts)

    # Outer rectangular boundary
    outer_pts = np.array([
        [0.0,     0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0,     h - 1.0],
    ])
    n_outer = len(outer_pts)

    outer_segments = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ], dtype=int)

    # Inner (mask) boundary segments
    inner_indices = np.arange(n_inner, dtype=int) + n_outer
    inner_segments = np.column_stack([
        inner_indices,
        np.roll(inner_indices, -1),
    ])

    vertices = np.vstack([outer_pts, boundary_pts])
    segments = np.vstack([outer_segments, inner_segments])

    A = {"vertices": vertices, "segments": segments}
    flags = f"pqDa{target_max_area}"
    B = tr.triangulate(A, flags)

    tri_vertices = B["vertices"]
    triangles = B["triangles"]

    return tri_vertices, triangles, mask_bool

# ---------- FEM Assembly (Mass & Stiffness for Diffusion) ----------

def assemble_M_and_K(vertices, triangles, diffusion_coeff=1.0):
    """
    Assemble global mass matrix M and stiffness matrix K for
    u_t = diffusion_coeff * Δu  (P1 FEM, constant coeff).
    """
    Nv = vertices.shape[0]
    Nt = triangles.shape[0]

    I_M, J_M, V_M = [], [], []
    I_K, J_K, V_K = [], [], []

    for t in range(Nt):
        idx = triangles[t]
        x = vertices[idx, 0]
        y = vertices[idx, 1]

        mat = np.array([
            [1.0, x[0], y[0]],
            [1.0, x[1], y[1]],
            [1.0, x[2], y[2]],
        ])
        area = 0.5 * np.abs(np.linalg.det(mat))
        if area <= 0:
            continue

        # Gradients of basis functions
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

        # Local stiffness matrix (Laplace)
        Ke = diffusion_coeff * (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)

        # Local consistent mass matrix (P1)
        Me = (area / 12.0) * np.array(
            [[2, 1, 1],
             [1, 2, 1],
             [1, 1, 2]]
        )

        for a_local in range(3):
            a = idx[a_local]
            for b_local in range(3):
                b_idx = idx[b_local]

                I_K.append(a)
                J_K.append(b_idx)
                V_K.append(Ke[a_local, b_local])

                I_M.append(a)
                J_M.append(b_idx)
                V_M.append(Me[a_local, b_local])

    K = sp.coo_matrix((V_K, (I_K, J_K)), shape=(Nv, Nv)).tocsr()
    M = sp.coo_matrix((V_M, (I_M, J_M)), shape=(Nv, Nv)).tocsr()
    return M, K

# ---------- Triangle classification & barriers ----------

def classify_triangles_by_centroid(vertices, triangles, canvas_bool):
    """
    Return a boolean array 'inside_tri' of length Nt:
    True if the triangle's centroid lies inside the blob (canvas_bool==1).
    """
    h, w = canvas_bool.shape
    mask = canvas_bool.astype(bool)

    tri_pts = vertices[triangles]          # (Nt, 3, 2)
    centroids = tri_pts.mean(axis=1)       # (Nt, 2)
    cx = np.clip(np.round(centroids[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.round(centroids[:, 1]).astype(int), 0, h - 1)

    inside_tri = mask[cy, cx]
    return inside_tri

def build_edge_triangle_map(triangles):
    """
    Build a mapping: edge (i, j) -> list of incident triangle indices.
    Edges use sorted vertex pairs: (min(i,j), max(i,j)).
    """
    edge_map = {}
    for t_idx, tri in enumerate(triangles):
        a, b, c = tri
        edges = [(a, b), (b, c), (c, a)]
        for i, j in edges:
            key = (i, j) if i < j else (j, i)
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(t_idx)
    return edge_map

def find_barrier_edges(triangles, inside_tri):
    """
    Identify edges shared by two triangles of different types
    (one inside, one outside).

    Returns a list of edges as vertex index pairs.
    """
    edge_map = build_edge_triangle_map(triangles)
    barrier_edges = []

    for (i, j), tri_list in edge_map.items():
        if len(tri_list) == 2:
            t1, t2 = tri_list
            if inside_tri[t1] != inside_tri[t2]:
                barrier_edges.append((i, j))

    return barrier_edges

def weaken_diffusion_on_edges(K, barrier_edges, alpha=0.1, conserve=True):
    """
    Weaken diffusion across the given edges by factor alpha in (0,1].

    If conserve=True:
      - scale K[i, j] and K[j, i] by alpha
      - adjust K[i, i] and K[j, j] so that row sums stay zero (mass-conserving)

    If conserve=False:
      - only scale off-diagonals => introduces leakage across those edges.
    """
    K_lil = K.tolil(copy=True)

    for (i, j) in barrier_edges:
        if i == j:
            continue  # just in case

        kij_old = K_lil[i, j]
        kji_old = K_lil[j, i]

        if kij_old == 0 and kji_old == 0:
            continue

        kij_new = alpha * kij_old
        kji_new = alpha * kji_old

        delta_ij = kij_new - kij_old
        delta_ji = kji_new - kji_old

        # Update off-diagonals
        K_lil[i, j] = kij_new
        K_lil[j, i] = kji_new

        if conserve:
            # Adjust diagonals so that row sums remain unchanged
            K_lil[i, i] -= delta_ij
            K_lil[j, j] -= delta_ji

    return K_lil.tocsr()

# ---------- Batched diffusion on mask ----------

def diffuse_mask_batched(
    mask,
    u0_batch,
    total_time,
    dt=0.1,
    diffusion_coeff=1.0,
    target_max_area=1.0,
    smooth_window=7,
    min_boundary_points=160,
    barrier_alpha=0.1,
    barrier_conserve=True,
    ):
    """
    Perform diffusion on a domain defined by 'mask', for a batch of initial
    fields given in 'u0_batch'.

    Parameters
    ----------
    mask : (H, W) array-like
        Binary or boolean mask. Nonzero = "inside blob".
        Used only to locate the interface and build barrier edges.

    u0_batch : (H, W, B) array-like
        Initial conditions. First two dimensions must match mask.
        Third dimension is batch index.

    total_time : float
        Total diffusion time.

    dt : float
        Time step for implicit Euler.

    diffusion_coeff : float
        Constant diffusion coefficient D.

    target_max_area, smooth_window, min_boundary_points :
        Parameters for triangulation and boundary smoothing.

    barrier_alpha : float in (0,1]
        Factor to weaken diffusion across mask boundary.
        alpha=1.0 => no barrier; alpha<<1 => nearly impermeable.

    barrier_conserve : bool
        If True: mass-conserving barrier (no leakage across interface).
        If False: leaky barrier across mask boundary.

    Returns
    -------
    u_final_grid : (H, W, B) ndarray
        Final diffused fields at time total_time, mapped back to the grid.
    """
    mask_arr = np.asarray(mask)
    u0_arr = np.asarray(u0_batch)
    if mask_arr.ndim != 2:
        raise ValueError("mask must be 2D (H, W).")
    if u0_arr.ndim != 3:
        raise ValueError("u0_batch must be 3D (H, W, B).")

    H, W = mask_arr.shape
    if u0_arr.shape[0] != H or u0_arr.shape[1] != W:
        raise ValueError("u0_batch first two dims must match mask shape.")

    # 1) Triangulate from mask
    vertices, triangles, mask_bool = triangulate_from_mask(
        mask_arr,
        target_max_area=target_max_area,
        smooth_window=smooth_window,
        min_boundary_points=min_boundary_points,
    )

    # 2) Assemble FEM matrices
    M, K = assemble_M_and_K(vertices, triangles, diffusion_coeff=diffusion_coeff)

    # 3) Build barrier edges from mask
    inside_tri = classify_triangles_by_centroid(vertices, triangles, mask_bool)
    barrier_edges = find_barrier_edges(triangles, inside_tri)
    K = weaken_diffusion_on_edges(K, barrier_edges, alpha=barrier_alpha, conserve=barrier_conserve)

    Nv = vertices.shape[0]
    B = u0_arr.shape[2]

    # 4) Map grid initial conditions to FEM vertices (nearest-neighbor)
    x_vert = vertices[:, 0]
    y_vert = vertices[:, 1]
    px = np.clip(np.round(x_vert).astype(int), 0, W - 1)
    py = np.clip(np.round(y_vert).astype(int), 0, H - 1)

    U = np.empty((Nv, B), dtype=float)
    for b in range(B):
        U[:, b] = u0_arr[py, px, b]

    # 5) Implicit Euler time-stepping: (M + dt K) U^{n+1} = M U^n
    num_steps = int(np.round(total_time / dt))
    if num_steps < 1:
        return u0_arr.copy()

    A = M + dt * K
    A_factorized = spla.factorized(A.tocsc())

    for _ in range(num_steps):
        rhs = M @ U  # (Nv, B)
        # Solve for each batch slice
        for b in range(B):
            U[:, b] = A_factorized(rhs[:, b])

    # 6) Map final vertex values back to grid using FEM interpolation
    import matplotlib.tri as mtri

    # Triangulation object for interpolation
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    # Pixel centers: (x, y) = (column, row)
    yy, xx = np.mgrid[0:H, 0:W]  # yy[i,j] = i, xx[i,j] = j
    x_flat = xx.ravel()
    y_flat = yy.ravel()

    u_final_grid = np.empty((H, W, B), dtype=float)

    for b in range(B):
        # piecewise-linear interpolator for this batch layer
        interp = mtri.LinearTriInterpolator(triang, U[:, b])

        # evaluate at all pixel centers
        z = interp(x_flat, y_flat)   # masked array (mask=True outside triangulation)
        z = z.reshape(H, W)

        # start from original grid values
        layer = u0_arr[:, :, b].copy()

        if np.ma.isMaskedArray(z):
            # keep original where outside triangulation, use interpolated where valid
            vals = z.filled(0.0)
            mask_valid = ~z.mask
            layer[mask_valid] = vals[mask_valid]
        else:
            # no mask: everything covered by triangulation
            layer[:, :] = z

        u_final_grid[:, :, b] = layer

    return u_final_grid