import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import gaussian_filter, zoom


def _block_reduce_sum_exact(arr, block):
    """
    Downsample by summing blocks. Pads the array so dimensions are divisible by block.
    Returns the downsampled array and the original shape.
    """
    if block <= 1:
        return arr.astype(np.float32, copy=True), arr.shape

    H, W = arr.shape
    pad_H = int(np.ceil(H / block) * block)
    pad_W = int(np.ceil(W / block) * block)

    padded = np.zeros((pad_H, pad_W), dtype=np.float32)
    padded[:H, :W] = arr.astype(np.float32, copy=False)

    low = skimage.measure.block_reduce(
        padded,
        block_size=(block, block),
        func=np.sum,
    ).astype(np.float32)

    return low, (H, W)


def _upsample_sum_preserving(low, block, original_shape):
    """
    Upsample low-resolution block sums back to full resolution.

    Because low contains sums per low-res pixel, naive zoom would inflate total
    intensity by approximately block**2. Dividing by block**2 preserves total mass.
    """
    if block <= 1:
        return low[:original_shape[0], :original_shape[1]].astype(np.float32, copy=False)

    high = zoom(
        low,
        zoom=block,
        order=1,
    ).astype(np.float32)

    high = high / float(block * block)

    H, W = original_shape
    return high[:H, :W].astype(np.float32, copy=False)


def _apply_mask_barrier(diffused, u0, mask, barrier_alpha, preserve_total=False):
    """
    Apply a foreground/background permeability barrier after fast Gaussian diffusion.

    Interpretation:
        barrier_alpha = 0.0 -> no permeability into background
        barrier_alpha = 1.0 -> full free diffusion into background

    The fast diffusion approximation itself is a global Gaussian filter. This
    post-processing step suppresses the background part of the diffused signal.

    Important:
        preserve_total=False is recommended for this MALDI simulation pipeline.
        Otherwise blocked background mass is redistributed into foreground cells,
        which can artificially amplify cell signal.
    """
    result = np.asarray(diffused, dtype=np.float32).copy()
    mask = np.asarray(mask)

    if result.shape != mask.shape:
        raise ValueError("diffused and mask must have the same shape.")

    alpha = float(np.clip(barrier_alpha, 0.0, 1.0))

    if alpha >= 1.0:
        return result.astype(np.float32, copy=False)

    foreground = mask > 0

    if not np.any(foreground):
        return result.astype(np.float32, copy=False)

    background = ~foreground

    result[background] *= alpha

    return result.astype(np.float32, copy=False)


def diffuse_mask_2d_fast_downsampled(
    u0,
    diffusion_coeff,
    total_time,
    mic_spacing=1.0,
    downsample=8,
    max_sigma_lowres=15.0,
    leaky_outer_boundary=False,
    preserve_total=True,
):
    u0 = np.asarray(u0, dtype=np.float32)

    if u0.ndim != 2:
        raise ValueError("u0 must be a 2D array.")

    input_sum = float(np.sum(u0))
    input_max = float(np.max(u0)) if u0.size else 0.0

    print(
        "[diffusion input]",
        "sum =", input_sum,
        "max =", input_max,
        "D =", float(diffusion_coeff),
        "T =", float(total_time),
        "D*T =", float(diffusion_coeff) * float(total_time),
    )

    if input_sum <= 0:
        print("[diffusion] input is empty; returning input")
        return u0.copy()

    if total_time <= 0 or diffusion_coeff <= 0:
        print("[diffusion] zero diffusion; returning input")
        return u0.copy()

    if mic_spacing <= 0:
        raise ValueError("mic_spacing must be positive.")

    if downsample < 1:
        raise ValueError("downsample must be >= 1.")

    downsample = int(downsample)

    effective_diffusion = float(diffusion_coeff) * float(total_time)

    sigma_full_px = np.sqrt(2.0 * effective_diffusion) / float(mic_spacing)
    sigma_low_px_uncapped = sigma_full_px / float(downsample)

    sigma_low_px = sigma_low_px_uncapped
    if max_sigma_lowres is not None:
        sigma_low_px = min(float(sigma_low_px), float(max_sigma_lowres))

    print(
        "[diffusion sigma]",
        "sigma_full_px =", sigma_full_px,
        "sigma_low_uncapped =", sigma_low_px_uncapped,
        "sigma_low_used =", sigma_low_px,
        "downsample =", downsample,
    )

    low, original_shape = _block_reduce_sum_exact(u0, downsample)

    print(
        "[low before filter]",
        "shape =", low.shape,
        "sum =", float(low.sum()),
        "max =", float(low.max()),
    )

    mode = "constant" if leaky_outer_boundary else "reflect"

    low_diffused = gaussian_filter(
        low,
        sigma=sigma_low_px,
        mode=mode,
        cval=0.0,
    ).astype(np.float32)

    print(
        "[low after filter]",
        "sum =", float(low_diffused.sum()),
        "max =", float(low_diffused.max()),
        "mode =", mode,
    )

    high = _upsample_sum_preserving(
        low_diffused,
        block=downsample,
        original_shape=original_shape,
    )

    output_sum = float(np.sum(high))
    output_max = float(np.max(high)) if high.size else 0.0

    print(
        "[high after upsample before rescale]",
        "sum =", output_sum,
        "max =", output_max,
    )

    if preserve_total:
        if output_sum > 0:
            high *= input_sum / output_sum
        else:
            print("[diffusion WARNING] output sum is zero; returning input")
            return u0.copy()

    print(
        "[diffusion output before barrier]",
        "sum =", float(high.sum()),
        "max =", float(high.max()),
    )

    return high.astype(np.float32, copy=False)


def diffuse_mask_2d(
    mask,
    u0,
    total_time,
    dt=0.1,
    diffusion_coeff=1.0,
    barrier_alpha=0.0,
    leaky_outer_boundary=False,
    cg_tol=None,
    cg_maxiter=None,
    use_gpu=True,
    mic_spacing=1.0,
    downsample=8,
    max_sigma_lowres=10.0,
):
    """
    Fast approximate masked diffusion.

    The actual diffusion is approximated with a downsampled Gaussian filter.
    A mask-based foreground/background barrier is then applied.

    barrier_alpha:
        0.0 -> no permeability into background
        1.0 -> full free Gaussian diffusion into background
    """
    m = np.asarray(mask)
    u0 = np.asarray(u0, dtype=np.float32)

    if m.ndim != 2:
        raise ValueError("mask must be 2D (H, W).")

    if u0.ndim != 2:
        raise ValueError("u0 must be 2D (H, W).")

    if u0.shape != m.shape:
        raise ValueError("u0 shape must match mask shape.")

    if total_time <= 0 or diffusion_coeff <= 0:
        return u0.copy()

    diffused = diffuse_mask_2d_fast_downsampled(
        u0=u0,
        diffusion_coeff=diffusion_coeff,
        total_time=total_time,
        mic_spacing=mic_spacing,
        downsample=downsample,
        max_sigma_lowres=max_sigma_lowres,
        leaky_outer_boundary=leaky_outer_boundary,
        preserve_total=True,
    )

    diffused = _apply_mask_barrier(
        diffused=diffused,
        u0=u0,
        mask=m,
        barrier_alpha=barrier_alpha,
        preserve_total=False,
    )

    print(
        "[diffusion output after barrier]",
        "barrier_alpha =", float(np.clip(barrier_alpha, 0.0, 1.0)),
        "sum =", float(diffused.sum()),
        "max =", float(diffused.max()) if diffused.size else 0.0,
    )

    return diffused.astype(np.float32, copy=False)


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
    cell_type_name="generic_cell",
    use_gpu=True,
    diffusion_downsample=8,
    max_sigma_lowres=10.0,
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
        local_mask = local_region == cell_id

        coords = np.argwhere(local_mask)

        sigma = float(sigmas[idx])
        factor = float(intensities[idx])

        intensities_map = np.zeros(local_mask.shape, dtype=np.float32)

        if coords.shape[0] > 0:
            cy_local, cx_local = coords.mean(axis=0)

            yy = coords[:, 0].astype(np.float32)
            xx = coords[:, 1].astype(np.float32)

            distance = np.sqrt(
                (yy - cy_local) ** 2 + (xx - cx_local) ** 2
            ) * float(mic_spacing)

            values = np.exp(-((distance / sigma) ** 2)) + 0.1
            intensities_map[coords[:, 0], coords[:, 1]] = values.astype(np.float32)

        total = float(np.sum(intensities_map))

        if total > 0:
            msi_cell = intensities_map / total * factor
        else:
            msi_cell = intensities_map * factor

        cell_canvas[y0:y1, x0:x1] += msi_cell

        cx = xs.mean()
        cy = ys.mean()

        msi_coordinates[cell_id] = (
            int(round(cx * mic_to_ms)),
            int(round(cy * mic_to_ms)),
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
        leaky_outer_boundary=leaky_outer_boundary,
        use_gpu=use_gpu,
        mic_spacing=mic_spacing,
        downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
    )

    block = int(ms_to_mic)

    reduced_msi = skimage.measure.block_reduce(
        diffusion_canvas,
        block_size=(block, block),
        func=np.sum,
    ).astype(np.float32)

    ms_canvas = reduced_msi

    return (
        ms_canvas,
        cell_canvas,
        diffusion_canvas,
        all_cell_intensities,
        cell_labels,
        msi_coordinates,
    )


def generate_msi_from_mask_slice(
    mask_slice,
    sigmas,
    intensities,
    diffusion_total_time,
    diffusion_coefficient,
    diffusion_barrier_alpha,
    mic_spacing=1.0,
    leaky_outer_boundary=False,
    use_gpu=True,
    ms_spacing=None,
    diffusion_downsample=8,
    max_sigma_lowres=10.0,
):
    if mask_slice.ndim != 2:
        raise ValueError("mask_slice must be a 2D labeled mask")

    cell_ids = np.unique(mask_slice)
    cell_ids = cell_ids[cell_ids > 0]

    if len(sigmas) != len(cell_ids):
        raise ValueError(
            f"Number of sigmas ({len(sigmas)}) must equal number of cells ({len(cell_ids)})"
        )

    if len(intensities) != len(cell_ids):
        raise ValueError(
            f"Number of intensities ({len(intensities)}) must equal number of cells ({len(cell_ids)})"
        )

    cell_canvas = np.zeros_like(mask_slice, dtype=np.float32)

    for idx, cell_id in enumerate(cell_ids):
        sigma = float(sigmas[idx])
        factor = float(intensities[idx])

        ys, xs = np.where(mask_slice == cell_id)

        if xs.size == 0:
            continue

        y0 = ys.min()
        y1 = ys.max() + 1
        x0 = xs.min()
        x1 = xs.max() + 1

        local_region = mask_slice[y0:y1, x0:x1]
        local_mask = local_region == cell_id
        coords = np.argwhere(local_mask)

        if coords.shape[0] == 0:
            continue

        cy, cx = coords.mean(axis=0)

        yy = coords[:, 0].astype(np.float32)
        xx = coords[:, 1].astype(np.float32)

        distance = np.sqrt(
            (yy - cy) ** 2 + (xx - cx) ** 2
        ) * float(mic_spacing)

        values = np.exp(-((distance / sigma) ** 2)) + 0.1

        intensities_map = np.zeros(local_mask.shape, dtype=np.float32)
        intensities_map[coords[:, 0], coords[:, 1]] = values.astype(np.float32)

        total = float(intensities_map.sum())

        if total > 0:
            intensities_map = intensities_map / total * factor

        cell_canvas[y0:y1, x0:x1] += intensities_map

    diffusion_canvas = diffuse_mask_2d(
        mask=mask_slice,
        u0=cell_canvas,
        total_time=diffusion_total_time,
        diffusion_coeff=diffusion_coefficient,
        barrier_alpha=diffusion_barrier_alpha,
        leaky_outer_boundary=leaky_outer_boundary,
        use_gpu=use_gpu,
        mic_spacing=mic_spacing,
        downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
    )

    if ms_spacing is not None:
        scaling_factor = int(ms_spacing / mic_spacing)

        if scaling_factor < 1:
            raise ValueError("ms_spacing must be >= mic_spacing")

        ms_canvas = skimage.measure.block_reduce(
            diffusion_canvas,
            block_size=(scaling_factor, scaling_factor),
            func=np.sum,
        ).astype(np.float32)

        return ms_canvas

    return diffusion_canvas


if __name__ == "__main__":
    mask_canvas = np.load("input/seg.npy")[1500:2000, 1500:2000]

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
        msi_coordinates,
    ) = generate_msi_from_existing_mask(
        mask_canvas=mask_canvas,
        ms_spacing=20,
        mic_spacing=1,
        diffusion_total_time=1,
        diffusion_dt=0.1,
        diffusion_coeff=265041.87486068613,
        barrier_alpha=0.0,
        leaky_outer_boundary=False,
        intensities=intensities,
        sigmas=sigmas,
        use_gpu=False,
        diffusion_downsample=8,
        max_sigma_lowres=10.0,
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Initial cell canvas")
    plt.imshow(cell_canvas, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Barrier-aware diffusion canvas")
    plt.imshow(diffusion_canvas, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Reduced MSI canvas")
    plt.imshow(ms_canvas, cmap="viridis")
    plt.colorbar()

    plt.tight_layout()
    plt.show()