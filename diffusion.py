import numpy as np
from skimage import measure
import triangle as tr
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------- Geometry & Triangulation ----------

def make_curvy_blob_canvas(
    height=200, width=200,
    base_radius=40,
    amplitude=10,
    frequency=4,
    noise_scale=0,
    ):
    """
    Create a binary canvas with background 0 and a single curvy blob of 1s in the center.
    """
    y = np.arange(height)
    x = np.arange(width)
    xx, yy = np.meshgrid(x, y)

    cx, cy = width // 2, height // 2

    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    num_theta_samples = 360
    theta_samples = np.linspace(-np.pi, np.pi, num_theta_samples, endpoint=False)
    noise_samples = np.random.randn(num_theta_samples) * noise_scale

    theta_flat = theta.ravel()
    theta_idx = (theta_flat - (-np.pi)) / (2 * np.pi) * num_theta_samples
    theta_idx %= num_theta_samples
    theta_idx_floor = np.floor(theta_idx).astype(int)
    theta_idx_ceil = (theta_idx_floor + 1) % num_theta_samples
    t = theta_idx - theta_idx_floor
    noise_flat = (1 - t) * noise_samples[theta_idx_floor] + t * noise_samples[theta_idx_ceil]
    noise = noise_flat.reshape(theta.shape)

    radius = base_radius + amplitude * np.sin(frequency * theta) + noise
    radius = np.clip(radius, base_radius * 0.4, None)

    blob = r <= radius

    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[blob] = 1
    return canvas

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

def triangulate_canvas_uniform(
    height=200,
    width=200,
    base_radius=40,
    amplitude=10,
    frequency=4,
    noise_scale=3.0,
    smooth_window=9,
    target_max_area=40.0,
    min_boundary_points=120,
    ):
    """
    Triangulate the entire rectangular canvas with:
      - blob boundary enforced as constrained edges,
      - roughly uniform triangle sizes.
    Returns:
        tri_vertices: (Nv, 2)
        triangles:    (Nt, 3)
        canvas:       (H, W)
    """
    canvas = make_curvy_blob_canvas(
        height=height, width=width,
        base_radius=base_radius,
        amplitude=amplitude,
        frequency=frequency,
        noise_scale=noise_scale,
    )
    h, w = canvas.shape
    mask = canvas.astype(bool)

    contours = measure.find_contours(mask, level=0.5)
    if len(contours) == 0:
        raise RuntimeError("No contour found in blob mask.")
    contour = max(contours, key=len)          # (N, 2) in (row, col)
    boundary_pts = np.fliplr(contour).astype(float)  # (N, 2) in (x, y)

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

    return tri_vertices, triangles, canvas

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

# ---------- Helper: classify triangles + derive better initial u0 ----------

def classify_triangles_by_centroid(vertices, triangles, canvas):
    """
    Return a boolean array 'inside_tri' of length Nt:
    True if the triangle's centroid lies inside the blob (canvas==1).
    """
    h, w = canvas.shape
    mask = canvas.astype(bool)

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

def make_initial_condition_from_triangles(vertices, triangles, canvas,
                                          majority_threshold=0.5):
    """
    Use triangle centroids to decide inside/outside, then assign nodal values
    based on how many incident triangles are inside.

    majority_threshold in [0,1]: vertex is "inside" if
        (# incident inside triangles / # incident triangles) > threshold
    """
    Nv = vertices.shape[0]
    Nt = triangles.shape[0]

    inside_tri = classify_triangles_by_centroid(vertices, triangles, canvas)
    # Identify barrier edges
    barrier_edges = find_barrier_edges(triangles, inside_tri)

    print(f"Found {len(barrier_edges)} barrier edges.")

    # Build vertex -> incident triangle lists
    incident_counts_total = np.zeros(Nv, dtype=int)
    incident_counts_inside = np.zeros(Nv, dtype=int)

    for t in range(Nt):
        tri = triangles[t]
        inside = inside_tri[t]
        for v in tri:
            incident_counts_total[v] += 1
            if inside:
                incident_counts_inside[v] += 1

    # Avoid division by zero (shouldn't happen, but just in case)
    valid = incident_counts_total > 0
    ratio = np.zeros(Nv, dtype=float)
    ratio[valid] = incident_counts_inside[valid] / incident_counts_total[valid]

    # Majority rule
    inside_vertex = ratio > majority_threshold

    u0 = np.zeros(Nv, dtype=float)
    u0[inside_vertex] = 1.0
    return u0, inside_tri

def weaken_diffusion_on_edges(K, barrier_edges, alpha=0.1):
    """
    Weaken diffusion across the given edges by factor alpha in (0,1].

    For each edge (i, j), we:
      - scale K[i, j] and K[j, i] by alpha
      - adjust K[i, i] and K[j, j] so that row sums stay zero (mass-conserving)
    """
    # Work in LIL for efficient element-wise updates
    K_lil = K.tolil(copy=True)

    for (i, j) in barrier_edges:
        if i == j:
            continue  # just in case

        kij_old = K_lil[i, j]
        kji_old = K_lil[j, i]

        # If there's no coupling, nothing to scale
        if kij_old == 0 and kji_old == 0:
            continue

        kij_new = alpha * kij_old
        kji_new = alpha * kji_old

        delta_ij = kij_new - kij_old  # = (alpha - 1) * kij_old
        delta_ji = kji_new - kji_old  # = (alpha - 1) * kji_old

        # Update off-diagonals
        K_lil[i, j] = kij_new
        K_lil[j, i] = kji_new

        # Adjust diagonals so that row sums remain unchanged
        K_lil[i, i] -= delta_ij
        K_lil[j, j] -= delta_ji

    return K_lil.tocsr()

# ---------- Time-stepping diffusion u_t = D Δu ----------

if __name__ == "__main__":
    # 1) Triangulate the canvas with the blob
    tri_vertices, triangles, canvas = triangulate_canvas_uniform(
        height=200, width=200,
        base_radius=40, amplitude=10, frequency=4, noise_scale=3.0,
        smooth_window=7,
        target_max_area=1.0, 
        min_boundary_points=160
    )

    # --- Plot 0: mask + triangulation overlay ---
    fig_mask, ax_mask = plt.subplots()
    ax_mask.imshow(canvas, cmap="gray", origin="upper")
    ax_mask.triplot(
        tri_vertices[:, 0],
        tri_vertices[:, 1],
        triangles,
        linewidth=0.4,
    )
    ax_mask.set_title("Mask with triangulation overlay")
    ax_mask.set_aspect("equal")
    ax_mask.axis("off")
    plt.tight_layout()
    plt.show()

    # 2) Assemble FEM matrices for diffusion u_t = D Δu
    D = 1.0
    M, K = assemble_M_and_K(tri_vertices, triangles, diffusion_coeff=D)

    # 3) Initial condition from triangle-based classification
    u0, inside_tri = make_initial_condition_from_triangles(
        tri_vertices, triangles, canvas,
        majority_threshold=0.6  # tweak: 0.5..0.7 works well
    )

    # 3b) Build list of edges between inside / outside triangles
    barrier_edges = find_barrier_edges(triangles, inside_tri)
    print(f"Found {len(barrier_edges)} barrier edges.")

    # 3c) Weaken diffusion across those edges by factor alpha
    barrier_alpha = 0.01  # e.g. 0.1 = strong but leaky barrier
    K = weaken_diffusion_on_edges(K, barrier_edges, alpha=barrier_alpha)

    # 4) Implicit time-stepping: (M + dt K) u^{n+1} = M u^n
    dt = 0.1
    times_to_save = [0, 20, 40, 60, 200, 1000]

    snapshots = {}
    current_time = 0.0

    A = M + dt * K
    A_factorized = spla.factorized(A.tocsc())

    u = u0.copy()

    # Save the initial state
    snapshots[0] = u.copy()

    # --- SIMULATE UNTIL LAST REQUESTED TIME ---
    target_T = times_to_save[-1]

    while current_time < target_T + 1e-9:
        rhs = M @ u
        u = A_factorized(rhs)
        current_time += dt

        # Check if we crossed a requested output time
        for t_target in times_to_save:
            # Save only once per target time
            if (t_target not in snapshots) and (current_time >= t_target):
                snapshots[t_target] = u.copy()
                print(f"Saved snapshot at t={t_target:.2f}")

    # snapshots dict now contains: u(t=0), u(t=20), u(t=40), ... etc.

    # --- PLOT THEM ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for ax, t in zip(axes, times_to_save):
        ui = snapshots[t]
        ax.tripcolor(
            tri_vertices[:, 0], tri_vertices[:, 1], triangles, ui,
            shading="gouraud"
        )
        ax.set_title(f"t = {t}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    plt.tight_layout()
    plt.show()