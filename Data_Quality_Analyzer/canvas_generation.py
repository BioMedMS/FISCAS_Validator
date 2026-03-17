import copy
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.sparse as sp

import cupy as cp
import cupyx.scipy.sparse as cpx_sp
import cupyx.scipy.sparse.linalg as cpx_spla


def generate_msi_from_existing_mask(
    mask_canvas,
    ms_spacing,
    mic_spacing,
    diffusion_total_time,
    diffusion_dt,
    diffusion_coeff,
    barrier_alpha,
    leaky_outer_boundary,
    intensities,
    sigmas,
    cell_type_name="generic_cell"
    ):
    if mask_canvas.ndim != 2:
        raise ValueError("mask_canvas must be a 2D labeled mask")

    cell_ids = np.unique(mask_canvas)
    cell_ids = cell_ids[cell_ids > 0]

    if len(intensities) != len(cell_ids):
        raise ValueError(
            f"Number of intensities ({len(intensities)}) must equal number of cells ({len(cell_ids)})"
        )

    if len(sigmas) != len(cell_ids):
        raise ValueError(
            f"Number of sigmas ({len(sigmas)}) must equal number of cells ({len(cell_ids)})"
        )

    scaling_factor = int(ms_spacing / mic_spacing)

    if scaling_factor < 1:
        raise ValueError("ms_spacing must be >= mic_spacing")

    mic_spacing = ms_spacing / scaling_factor
    mic_to_ms = mic_spacing / ms_spacing
    ms_to_mic = ms_spacing / mic_spacing

    H_mic, W_mic = mask_canvas.shape

    cell_canvas = np.zeros((H_mic, W_mic), dtype=np.float32)

    all_cell_intensities = {}
    cell_labels = {}
    msi_coordinates = {}

    for idx, cell_id in enumerate(cell_ids):
        ys, xs = np.where(mask_canvas == cell_id)

        if len(xs) == 0 or len(ys) == 0:
            continue

        y0 = ys.min()
        y1 = ys.max() + 1
        x0 = xs.min()
        x1 = xs.max() + 1

        local_region = mask_canvas[y0:y1, x0:x1]

        cell_mask = np.zeros_like(local_region, dtype=np.uint16)
        cell_mask[local_region == cell_id] = cell_id

        local_mask = copy.deepcopy(cell_mask)
        coords = np.argwhere(local_mask > 0)

        sigma = sigmas[idx]
        factor = intensities[idx]

        intensities_map = np.zeros_like(local_mask, dtype=np.float32)

        if len(coords) > 0:
            center_of_mass = coords.mean(axis=0)
            cy_local, cx_local = center_of_mass

            for y, x in coords:
                distance = np.sqrt((y - cy_local) ** 2 + (x - cx_local) ** 2) * mic_spacing
                intensities_map[y, x] = np.exp(-(distance / sigma) ** 2) + 0.1

        if np.sum(intensities_map) > 0:
            msi_cell = intensities_map / np.sum(intensities_map) * factor
        else:
            msi_cell = intensities_map * factor

        if msi_cell.shape != cell_mask.shape:
            raise ValueError(
                f"Generated cell shape {msi_cell.shape} does not match mask shape {cell_mask.shape}"
            )

        cell_canvas[y0:y1, x0:x1] += msi_cell

        cx = xs.mean()
        cy = ys.mean()

        msi_coordinates[cell_id] = (
            int(round(cx * mic_to_ms)),
            int(round(cy * mic_to_ms))
        )

        all_cell_intensities[cell_id] = float(np.sum(msi_cell))
        cell_labels[cell_id] = cell_type_name

    diffusion_canvas = diffuse_mask_2d(
        mask=mask_canvas,
        u0=cell_canvas,
        total_time=diffusion_total_time,
        dt=diffusion_dt,
        diffusion_coeff=diffusion_coeff,
        barrier_alpha=barrier_alpha,
        leaky_outer_boundary=leaky_outer_boundary
    )

    block = int(ms_to_mic)

    reduced_msi = skimage.measure.block_reduce(
        diffusion_canvas,
        block_size=(block, block),
        func=np.sum
    ).astype(np.float32)

    ms_canvas = reduced_msi

    return (
        ms_canvas,
        cell_canvas,
        diffusion_canvas,
        all_cell_intensities,
        cell_labels,
        msi_coordinates
    )


def _build_pixel_laplacian(mask, diffusion_coeff=1.0, barrier_alpha=0.1):
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask must be 2D (H, W).")

    H, W = m.shape
    N = H * W

    D = np.float32(diffusion_coeff)
    alpha = np.float32(barrier_alpha)

    # ----- horizontal neighbor pairs -----
    # left pixel ids and right pixel ids
    a_h = (np.arange(H, dtype=np.int32)[:, None] * W + np.arange(W - 1, dtype=np.int32)[None, :]).ravel()
    b_h = a_h + 1

    same_h = (m[:, :-1] == m[:, 1:]).ravel()
    w_h = np.where(same_h, D, D * alpha).astype(np.float32)

    # ----- vertical neighbor pairs -----
    a_v = (np.arange(H - 1, dtype=np.int32)[:, None] * W + np.arange(W, dtype=np.int32)[None, :]).ravel()
    b_v = a_v + W

    same_v = (m[:-1, :] == m[1:, :]).ravel()
    w_v = np.where(same_v, D, D * alpha).astype(np.float32)

    # combine horizontal + vertical edges
    a = np.concatenate([a_h, a_v])
    b = np.concatenate([b_h, b_v])
    w = np.concatenate([w_h, w_v]).astype(np.float32, copy=False)

    # each undirected edge contributes:
    # (a,a)+=w, (b,b)+=w, (a,b)+=-w, (b,a)+=-w
    rows = np.concatenate([a, b, a, b]).astype(np.int32, copy=False)
    cols = np.concatenate([a, b, b, a]).astype(np.int32, copy=False)
    vals = np.concatenate([w, w, -w, -w]).astype(np.float32, copy=False)

    K = sp.coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32).tocsr()
    K.sum_duplicates()
    K.eliminate_zeros()
    return K

def _outer_boundary_indices(H, W):
    if H == 1 and W == 1:
        return np.array([0], dtype=int)

    top = np.arange(0, W, dtype=int)
    bottom = np.arange((H - 1) * W, H * W, dtype=int)
    left = np.arange(0, H * W, W, dtype=int)
    right = np.arange(W - 1, H * W, W, dtype=int)

    return np.unique(np.concatenate([top, bottom, left, right]))

def diffuse_mask_2d(
    mask,
    u0,
    total_time,
    dt=0.1,
    diffusion_coeff=1.0,
    barrier_alpha=0.1,
    leaky_outer_boundary=True,
    cg_tol=1e-6,
    cg_maxiter=1000,
    ):
    m = np.asarray(mask)
    u0 = np.asarray(u0)

    if m.ndim != 2:
        raise ValueError("mask must be 2D (H, W).")

    if u0.ndim != 2:
        raise ValueError("u0 must be 2D (H, W).")

    H, W = m.shape

    if u0.shape != (H, W):
        raise ValueError("u0 shape must match mask shape.")

    if total_time <= dt:
        return u0.copy()

    N = H * W
    boundary = _outer_boundary_indices(H, W) if leaky_outer_boundary else None
    num_steps = int(np.round(total_time / dt))

    # Build sparse operator on CPU exactly as before
    K_cpu = _build_pixel_laplacian(
        m,
        diffusion_coeff=diffusion_coeff,
        barrier_alpha=barrier_alpha
    )

    I_cpu = sp.identity(N, format="csr", dtype=np.float32)
    A_cpu = (I_cpu + np.float32(dt) * K_cpu).tocsr()

    del K_cpu

    # Move matrix once to GPU
    A_gpu = cpx_sp.csr_matrix(A_cpu)

    del A_cpu
    del I_cpu
    import gc
    gc.collect()

    # Simple Jacobi preconditioner on GPU
    diag = A_gpu.diagonal()
    diag = cp.where(diag != 0, diag, 1.0)

    def precond(x):
        return x / diag

    M_gpu = cpx_spla.LinearOperator(
        shape=A_gpu.shape,
        matvec=precond,
        dtype=A_gpu.dtype
    )

    # Move initial condition to GPU
    U_gpu = cp.asarray(u0.reshape(N), dtype=cp.float32)

    for step in range(num_steps):
        U_gpu, info = cpx_spla.cg(
            A_gpu,
            U_gpu,
            x0=U_gpu,
            rtol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=M_gpu,
        )

        if info != 0:
            raise RuntimeError(
                f"GPU CG failed at step {step} with info={info}. "
                f"Try larger cg_maxiter or looser cg_tol."
            )

        if boundary is not None:
            U_gpu[boundary] = 0.0

    out = cp.asnumpy(U_gpu).reshape(H, W)
    return out

if __name__ == "__main__":
    mask_canvas = np.load("input/seg.npy")

    cell_ids = np.unique(mask_canvas)
    cell_ids = cell_ids[cell_ids > 0]
    n_cells = len(cell_ids)

    intensities = np.full(n_cells, 10.0)
    sigmas = np.full(n_cells, 40.0)

    (
        ms_canvas,
        cell_canvas,
        diffusion_canvas,
        all_cell_intensities,
        cell_labels,
        msi_coordinates
    ) = generate_msi_from_existing_mask(
        mask_canvas=mask_canvas,
        ms_spacing=20,
        mic_spacing=1,
        diffusion_total_time=10,
        diffusion_dt=0.1,
        diffusion_coeff=1.0,
        barrier_alpha=0.1,
        leaky_outer_boundary=True,
        intensities=intensities,
        sigmas=sigmas
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Initial cell canvas")
    plt.imshow(cell_canvas, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Diffusion canvas")
    plt.imshow(diffusion_canvas, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Reduced MSI canvas")
    plt.imshow(ms_canvas, cmap="viridis")
    plt.colorbar()

    plt.tight_layout()
    plt.show()