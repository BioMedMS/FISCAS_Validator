import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt



def _build_pixel_laplacian(mask, diffusion_coeff=1.0, barrier_alpha=0.1):
    """
    Build a sparse weighted graph Laplacian on a 2D grid (4-neighborhood),
    with reduced coupling across label boundaries.

    mask: (H, W) integer-like labels. Background is typically 0, but treated
          like any other label for boundary detection.

    For each adjacent pair (p, q):
      w = diffusion_coeff
      if mask[p] != mask[q]: w *= barrier_alpha

    Returns:
        K: (N, N) sparse CSR matrix, N=H*W
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask must be 2D (H, W).")
    H, W = m.shape
    N = H * W

    # Map (y, x) -> linear index
    def idx(y, x):
        return y * W + x

    I = []
    J = []
    V = []

    def add_edge(a, b, w):
        # Laplacian contributions for an undirected edge:
        # K[a,a] += w; K[b,b] += w; K[a,b] -= w; K[b,a] -= w
        I.extend([a, b, a, b])
        J.extend([a, b, b, a])
        V.extend([w, w, -w, -w])

    D = float(diffusion_coeff)
    alpha = float(barrier_alpha)

    # Horizontal edges
    for y in range(H):
        for x in range(W - 1):
            a = idx(y, x)
            b = idx(y, x + 1)
            w = D * (alpha if m[y, x] != m[y, x + 1] else 1.0)
            if w != 0.0:
                add_edge(a, b, w)

    # Vertical edges
    for y in range(H - 1):
        for x in range(W):
            a = idx(y, x)
            b = idx(y + 1, x)
            w = D * (alpha if m[y, x] != m[y + 1, x] else 1.0)
            if w != 0.0:
                add_edge(a, b, w)

    K = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return K


def _outer_boundary_indices(H, W):
    """Linear indices for the outer image boundary."""
    if H == 1 and W == 1:
        return np.array([0], dtype=int)

    top = np.arange(0, W, dtype=int)
    bottom = np.arange((H - 1) * W, H * W, dtype=int)
    left = np.arange(0, H * W, W, dtype=int)
    right = np.arange(W - 1, H * W, W, dtype=int)
    return np.unique(np.concatenate([top, bottom, left, right]))


def diffuse_mask_batched(
    mask,
    u0_batch,
    total_time,
    dt=0.1,
    diffusion_coeff=1.0,     # can be scalar or (B,)
    barrier_alpha=0.1,       # can be scalar or (B,)
    leaky_outer_boundary=True,
):
    """
    Pixel-based diffusion with barriers on label boundaries.

    mask : (H, W) label/index mask. Background should be 0.
    u0_batch : (H, W, B) initial fields

    diffusion_coeff: float or array-like of shape (B,)
    barrier_alpha:   float or array-like of shape (B,)
    """
    m = np.asarray(mask)
    u0 = np.asarray(u0_batch)

    if m.ndim != 2:
        raise ValueError("mask must be 2D (H, W).")
    if u0.ndim != 3:
        raise ValueError("u0_batch must be 3D (H, W, B).")

    H, W = m.shape
    if u0.shape[0] != H or u0.shape[1] != W:
        raise ValueError("u0_batch first two dims must match mask shape.")

    if total_time <= dt:
        return u0.copy()

    N = H * W
    B = u0.shape[2]

    # Normalize coeffs to per-layer arrays of length B
    D = np.asarray(diffusion_coeff, dtype=float)
    Apha = np.asarray(barrier_alpha, dtype=float)

    if D.ndim == 0:
        D = np.full(B, float(D), dtype=float)
    elif D.shape != (B,):
        raise ValueError(f"diffusion_coeff must be scalar or shape (B,), got {D.shape}")

    if Apha.ndim == 0:
        Apha = np.full(B, float(Apha), dtype=float)
    elif Apha.shape != (B,):
        raise ValueError(f"barrier_alpha must be scalar or shape (B,), got {Apha.shape}")

    boundary = _outer_boundary_indices(H, W) if leaky_outer_boundary else None
    num_steps = int(np.round(total_time / dt))

    out = np.empty_like(u0, dtype=float)

    # Diffuse each layer independently with its own matrix
    for b in range(B):
        # Build stiffness (Laplacian) for this layer
        K = _build_pixel_laplacian(
            m,
            diffusion_coeff=D[b],
            barrier_alpha=Apha[b],
        )

        # Implicit Euler: (I + dt K) u^{n+1} = u^n
        I = sp.identity(N, format="csr")
        A_mat = (I + dt * K).tocsc()
        solve = spla.factorized(A_mat)

        U = u0[..., b].reshape(N).astype(float, copy=True)

        for _ in range(num_steps):
            U = solve(U)
            if boundary is not None:
                U[boundary] = 0.0

        out[..., b] = U.reshape(H, W)

    for b in range(B):
        plt.figure(figsize=(5, 5))
        plt.imshow(out[..., b], cmap="viridis")
        plt.colorbar()
        plt.title(f"Diffused layer {b}")
        plt.axis("off")
        plt.savefig(f"diffused_layer_{b}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return out