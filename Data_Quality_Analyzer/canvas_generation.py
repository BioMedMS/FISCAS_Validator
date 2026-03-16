import copy
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def generate_msi_and_plot_from_existing_mask(
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

    cell_canvas = np.zeros(
        (H_mic, W_mic, 1),
        dtype=np.float32
    )

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

        intensities_map = np.zeros_like(local_mask, dtype=float)

        if len(coords) > 0:
            center_of_mass = coords.mean(axis=0)
            cx_local, cy_local = center_of_mass

            for x, y in coords:
                distance = np.sqrt((x - cx_local) ** 2 + (y - cy_local) ** 2) * mic_spacing
                intensities_map[x, y] = np.exp(-(distance / sigma) ** 2) + 0.1

        if np.sum(intensities_map) > 0:
            msi_cell = intensities_map / np.sum(intensities_map) * factor
        else:
            msi_cell = intensities_map * factor

        if msi_cell.shape != cell_mask.shape:
            raise ValueError(
                f"Generated cell shape {msi_cell.shape} does not match mask shape {cell_mask.shape}"
            )

        cell_canvas[y0:y1, x0:x1, 0] += msi_cell

        cx = xs.mean()
        cy = ys.mean()

        msi_coordinates[cell_id] = (
            int(round(cx * mic_to_ms)),
            int(round(cy * mic_to_ms))
        )

        all_cell_intensities[cell_id] = float(np.sum(msi_cell))
        cell_labels[cell_id] = cell_type_name

    diffusion_canvas = diffuse_mask_batched(
        mask=mask_canvas,
        u0_batch=cell_canvas,
        total_time=diffusion_total_time,
        dt=diffusion_dt,
        diffusion_coeff=diffusion_coeff,
        barrier_alpha=barrier_alpha,
        leaky_outer_boundary=leaky_outer_boundary
    )

    block = int(ms_to_mic)

    reduced_msi = skimage.measure.block_reduce(
        diffusion_canvas[:, :, 0],
        block_size=(block, block),
        func=np.sum
    ).astype(np.float32)

    ms_canvas = reduced_msi[:, :, np.newaxis]

    f_mask_canvas = plt.figure()

    masked_canvas = np.ma.masked_where(
        mask_canvas == 0,
        mask_canvas
    )

    my_cmap = plt.cm.prism.copy()
    my_cmap.set_bad(color="black")

    plt.imshow(masked_canvas, cmap=my_cmap)
    plt.axis("off")

    f_diffusion_canvas = plt.figure()
    f_diffusion_canvas.canvas.manager.set_window_title(
        "MSI Canvas After Diffusion"
    )

    plt.imshow(
        diffusion_canvas[:, :, 0],
        vmax=np.percentile(diffusion_canvas[:, :, 0], 99.0),
        cmap="inferno"
    )
    plt.axis("off")

    f_msi_canvas = plt.figure()
    f_msi_canvas.canvas.manager.set_window_title(
        "MSI Canvas Target Resolution"
    )

    plt.imshow(
        ms_canvas[:, :, 0],
        vmax=np.percentile(ms_canvas[:, :, 0], 99.0),
        cmap="inferno"
    )
    plt.axis("off")

    return (
        f_mask_canvas,
        f_diffusion_canvas,
        f_msi_canvas,
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

    def idx(y, x):
        return y * W + x

    I = []
    J = []
    V = []

    def add_edge(a, b, w):
        I.extend([a, b, a, b])
        J.extend([a, b, b, a])
        V.extend([w, w, -w, -w])

    D = float(diffusion_coeff)
    alpha = float(barrier_alpha)

    for y in range(H):
        for x in range(W - 1):
            a = idx(y, x)
            b = idx(y, x + 1)
            w = D * (alpha if m[y, x] != m[y, x + 1] else 1.0)
            if w != 0.0:
                add_edge(a, b, w)

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
        diffusion_coeff=1.0,
        barrier_alpha=0.1,
        leaky_outer_boundary=True
    ):
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

    D = np.asarray(diffusion_coeff, dtype=float)
    Alpha = np.asarray(barrier_alpha, dtype=float)

    if D.ndim == 0:
        D = np.full(B, float(D), dtype=float)
    elif D.shape != (B,):
        raise ValueError(f"diffusion_coeff must be scalar or shape (B,), got {D.shape}")

    if Alpha.ndim == 0:
        Alpha = np.full(B, float(Alpha), dtype=float)
    elif Alpha.shape != (B,):
        raise ValueError(f"barrier_alpha must be scalar or shape (B,), got {Alpha.shape}")

    boundary = _outer_boundary_indices(H, W) if leaky_outer_boundary else None
    num_steps = int(np.round(total_time / dt))
    out = np.empty_like(u0, dtype=float)

    for b in range(B):
        K = _build_pixel_laplacian(
            m,
            diffusion_coeff=D[b],
            barrier_alpha=Alpha[b]
        )

        I = sp.identity(N, format="csr")
        A_mat = (I + dt * K).tocsc()
        solve = spla.factorized(A_mat)
        U = u0[..., b].reshape(N).astype(float, copy=True)

        for _ in range(num_steps):
            U = solve(U)
            if boundary is not None:
                U[boundary] = 0.0

        out[..., b] = U.reshape(H, W)

    return out


if __name__ == "__main__":
    mask_canvas = np.load("input/seg.npy")

    cell_ids = np.unique(mask_canvas)
    cell_ids = cell_ids[cell_ids > 0]
    n_cells = len(cell_ids)

    intensities = np.full(n_cells, 10.0)
    sigmas = np.full(n_cells, 40.0)

    (
        f_mask_canvas,
        f_diffusion_canvas,
        f_msi_canvas,
        ms_canvas,
        cell_canvas,
        diffusion_canvas,
        all_cell_intensities,
        cell_labels,
        msi_coordinates
    ) = generate_msi_and_plot_from_existing_mask(
        mask_canvas=mask_canvas,
        ms_spacing=20,
        mic_spacing=1,
        diffusion_total_time=10,
        diffusion_dt=10,
        diffusion_coeff=1.0,
        barrier_alpha=0.1,
        leaky_outer_boundary=True,
        intensities=intensities,
        sigmas=sigmas
    )

    print("Cells detected:", len(msi_coordinates))
    print("Cell intensities:", all_cell_intensities)

    plt.show()