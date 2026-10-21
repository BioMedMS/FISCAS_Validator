import contextlib
import io

import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import gaussian_filter, zoom, distance_transform_edt


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
    target_canvas=None,
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



def _choose_cell_center_from_mask_and_target(local_mask, target_patch=None, scaling_factor=None):
    """
    Pick a center for one cell using both the cell geometry and the target.

    Strategy:
      1. Restrict the search to the cell pixels.
      2. If a target patch is available, map it to the cell grid and find the cell's
         global maximum inside the target.
      3. If several pixels are tied / nearly tied, prefer the flattest point
         (smallest gradient magnitude) and then the most interior point
         (largest distance-to-boundary).
      4. Fall back to the mask interior maximum if no target support is available.
    """
    local_mask = np.asarray(local_mask, dtype=bool)
    coords = np.argwhere(local_mask)

    if coords.shape[0] == 0:
        return None

    interior_distance = distance_transform_edt(local_mask).astype(np.float32)

    if target_patch is None:
        best_idx = np.argmax(interior_distance[coords[:, 0], coords[:, 1]])
        cy, cx = coords[best_idx]
        return float(cy), float(cx)

    target_patch = np.asarray(target_patch, dtype=np.float32)

    if target_patch.ndim != 2 or target_patch.size == 0:
        best_idx = np.argmax(interior_distance[coords[:, 0], coords[:, 1]])
        cy, cx = coords[best_idx]
        return float(cy), float(cx)

    if scaling_factor is None or scaling_factor <= 1:
        sf = 1
    else:
        sf = int(scaling_factor)

    # Paint the target values onto the high-resolution local cell grid.
    hr_target = np.repeat(np.repeat(target_patch, sf, axis=0), sf, axis=1)
    hr_target = hr_target[:local_mask.shape[0], :local_mask.shape[1]]

    if hr_target.shape != local_mask.shape:
        padded = np.zeros(local_mask.shape, dtype=np.float32)
        padded[:hr_target.shape[0], :hr_target.shape[1]] = hr_target
        hr_target = padded

    masked_target = np.where(local_mask, hr_target, -np.inf)

    finite_values = masked_target[np.isfinite(masked_target)]
    if finite_values.size == 0:
        best_idx = np.argmax(interior_distance[coords[:, 0], coords[:, 1]])
        cy, cx = coords[best_idx]
        return float(cy), float(cx)

    max_value = float(np.max(finite_values))

    if not np.isfinite(max_value):
        best_idx = np.argmax(interior_distance[coords[:, 0], coords[:, 1]])
        cy, cx = coords[best_idx]
        return float(cy), float(cx)

    # Allow a tiny tolerance so flat maxima / quantized peaks are handled robustly.
    tol = max(1e-8, abs(max_value) * 1e-6)
    peak_mask = local_mask & (hr_target >= (max_value - tol))
    peak_coords = np.argwhere(peak_mask)

    if peak_coords.shape[0] == 0:
        peak_coords = coords

    gy, gx = np.gradient(np.where(local_mask, hr_target, 0.0).astype(np.float32))
    grad_mag = np.sqrt(gx * gx + gy * gy)

    peak_grad = grad_mag[peak_coords[:, 0], peak_coords[:, 1]]
    peak_dist = interior_distance[peak_coords[:, 0], peak_coords[:, 1]]

    order = np.lexsort((-peak_dist, peak_grad))
    cy, cx = peak_coords[order[0]]

    return float(cy), float(cx)


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
    target_canvas=None,
    local_diffusion=True,
    diffusion_crop_n_sigma=4.0,
    local_diffusion_verbose=False,
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
            local_target = None
            local_sf = None

            if target_canvas is not None:
                local_sf = int(ms_spacing / mic_spacing)
                ty0 = y0 // local_sf
                ty1 = int(np.ceil(y1 / local_sf))
                tx0 = x0 // local_sf
                tx1 = int(np.ceil(x1 / local_sf))
                local_target = np.asarray(target_canvas[ty0:ty1, tx0:tx1], dtype=np.float32)

            center = _choose_cell_center_from_mask_and_target(
                local_mask=local_mask,
                target_patch=local_target,
                scaling_factor=local_sf,
            )

            if center is None:
                cy_local, cx_local = coords.mean(axis=0)
            else:
                cy_local, cx_local = center

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

        local_target = None
        local_sf = None

        if target_canvas is not None:
            local_sf = int(ms_spacing / mic_spacing)
            ty0 = y0 // local_sf
            ty1 = int(np.ceil(y1 / local_sf))
            tx0 = x0 // local_sf
            tx1 = int(np.ceil(x1 / local_sf))
            local_target = np.asarray(target_canvas[ty0:ty1, tx0:tx1], dtype=np.float32)

        center = _choose_cell_center_from_mask_and_target(
            local_mask=local_mask,
            target_patch=local_target,
            scaling_factor=local_sf,
        )

        if center is None:
            cx = xs.mean()
            cy = ys.mean()
        else:
            cy_local, cx_local = center
            cx = x0 + cx_local
            cy = y0 + cy_local

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



def estimate_diffusion_effect_radius_px(
    diffusion_coeff,
    total_time,
    mic_spacing=1.0,
    diffusion_downsample=8,
    max_sigma_lowres=10.0,
    n_sigma=4.0,
):
    """
    Estimate the high-resolution crop radius needed to capture the visible
    Gaussian diffusion effect for local diffusion.

    The full diffusion path computes sigma_full_px = sqrt(2 * D * T) / mic_spacing,
    then optionally caps sigma on the temporary low-resolution grid. This helper
    mirrors that cap and converts the usable sigma back to high-resolution pixels.
    """
    if total_time <= 0 or diffusion_coeff <= 0:
        return 0

    if mic_spacing <= 0:
        raise ValueError("mic_spacing must be positive.")

    downsample = int(max(1, diffusion_downsample))
    sigma_full_px = np.sqrt(2.0 * float(diffusion_coeff) * float(total_time)) / float(mic_spacing)

    if max_sigma_lowres is not None:
        sigma_full_px = min(sigma_full_px, float(max_sigma_lowres) * float(downsample))

    # Add a small numerical guard for low-res filtering / upsampling support.
    return int(np.ceil(float(n_sigma) * sigma_full_px + 2 * downsample))


def _aligned_crop_box_from_bbox(
    y0,
    y1,
    x0,
    x1,
    image_shape,
    margin,
    alignment,
):
    """
    Build a crop around a high-resolution cell bbox, aligned to MSI block
    boundaries so reduced local crops stitch back exactly into MSI coordinates.
    """
    H, W = image_shape
    alignment = int(max(1, alignment))
    margin = int(max(0, margin))

    top = max(0, int(y0) - margin)
    left = max(0, int(x0) - margin)
    bottom = min(H, int(y1) + margin)
    right = min(W, int(x1) + margin)

    top = (top // alignment) * alignment
    left = (left // alignment) * alignment
    bottom = int(np.ceil(bottom / alignment) * alignment)
    right = int(np.ceil(right / alignment) * alignment)

    bottom = min(H, max(bottom, top + alignment))
    right = min(W, max(right, left + alignment))

    return top, bottom, left, right


def _render_one_cell_source_patch(
    mask_patch,
    cell_id,
    sigma,
    intensity,
    mic_spacing=1.0,
    target_patch=None,
    scaling_factor=None,
):
    """
    Render one cell's pre-diffusion source field inside a local crop.
    The returned patch has the same shape as mask_patch and contains only this
    cell's contribution.
    """
    local_mask = mask_patch == int(cell_id)
    coords = np.argwhere(local_mask)
    source = np.zeros(mask_patch.shape, dtype=np.float32)

    if coords.shape[0] == 0 or float(intensity) <= 0:
        return source

    center = _choose_cell_center_from_mask_and_target(
        local_mask=local_mask,
        target_patch=target_patch,
        scaling_factor=scaling_factor,
    )

    if center is None:
        cy, cx = coords.mean(axis=0)
    else:
        cy, cx = center

    yy = coords[:, 0].astype(np.float32)
    xx = coords[:, 1].astype(np.float32)

    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * float(mic_spacing)
    sigma = max(float(sigma), 1e-6)
    values = np.exp(-((distance / sigma) ** 2)) + 0.1
    source[coords[:, 0], coords[:, 1]] = values.astype(np.float32)

    total = float(source.sum())
    if total > 0:
        source *= float(intensity) / total

    return source.astype(np.float32, copy=False)


def _block_reduce_local_to_msi(diffused_patch, scaling_factor):
    low, _ = _block_reduce_sum_exact(
        np.asarray(diffused_patch, dtype=np.float32),
        int(scaling_factor),
    )
    return low.astype(np.float32, copy=False)


def generate_msi_from_mask_slice_local_diffusion(
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
    target_canvas=None,
    diffusion_crop_n_sigma=4.0,
    verbose=False,
):
    """
    Generate an MSI canvas by diffusing each cell in a padded local crop and
    stitching the reduced local MSI patches into the final image.

    This avoids allocating and blurring a full high-resolution diffusion canvas.
    It is intended to approximate the full-canvas path when the local crop radius
    captures the practical Gaussian support. Increase diffusion_crop_n_sigma for
    higher fidelity; 4.0 is usually a good speed/accuracy tradeoff.
    """
    if ms_spacing is None:
        raise ValueError("local diffusion requires ms_spacing so local crops can be stitched into MSI space.")

    mask_slice = np.asarray(mask_slice)
    if mask_slice.ndim != 2:
        raise ValueError("mask_slice must be a 2D labeled mask")

    scaling_factor = int(ms_spacing / mic_spacing)
    if scaling_factor < 1:
        raise ValueError("ms_spacing must be >= mic_spacing")

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

    H, W = mask_slice.shape
    out_h = int(np.ceil(H / scaling_factor))
    out_w = int(np.ceil(W / scaling_factor))
    ms_canvas = np.zeros((out_h, out_w), dtype=np.float32)

    diffusion_margin = estimate_diffusion_effect_radius_px(
        diffusion_coeff=diffusion_coefficient,
        total_time=diffusion_total_time,
        mic_spacing=mic_spacing,
        diffusion_downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
        n_sigma=diffusion_crop_n_sigma,
    )

    # Ensure the crop has at least one MSI-pixel collar, even for tiny mu values.
    diffusion_margin = max(int(diffusion_margin), int(scaling_factor))

    if verbose:
        print(
            "[local diffusion]",
            "cells =", len(cell_ids),
            "margin_px =", diffusion_margin,
            "scaling_factor =", scaling_factor,
        )

    for idx, cell_id in enumerate(cell_ids):
        sigma = float(sigmas[idx])
        intensity = float(intensities[idx])

        if intensity <= 0:
            continue

        ys, xs = np.where(mask_slice == int(cell_id))
        if xs.size == 0:
            continue

        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1

        top, bottom, left, right = _aligned_crop_box_from_bbox(
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            image_shape=mask_slice.shape,
            margin=diffusion_margin,
            alignment=scaling_factor,
        )

        mask_patch = mask_slice[top:bottom, left:right]

        local_target = None
        if target_canvas is not None:
            ty0 = top // scaling_factor
            ty1 = int(np.ceil(bottom / scaling_factor))
            tx0 = left // scaling_factor
            tx1 = int(np.ceil(right / scaling_factor))
            local_target = np.asarray(target_canvas[ty0:ty1, tx0:tx1], dtype=np.float32)

        source_patch = _render_one_cell_source_patch(
            mask_patch=mask_patch,
            cell_id=int(cell_id),
            sigma=sigma,
            intensity=intensity,
            mic_spacing=mic_spacing,
            target_patch=local_target,
            scaling_factor=scaling_factor,
        )

        if diffusion_total_time <= 0 or diffusion_coefficient <= 0:
            diffused_patch = source_patch
        else:
            # Suppress per-cell debug prints from the full diffusion helper.
            if verbose:
                diffused_patch = diffuse_mask_2d(
                    mask=mask_patch,
                    u0=source_patch,
                    total_time=diffusion_total_time,
                    diffusion_coeff=diffusion_coefficient,
                    barrier_alpha=diffusion_barrier_alpha,
                    leaky_outer_boundary=True,
                    use_gpu=use_gpu,
                    mic_spacing=mic_spacing,
                    downsample=diffusion_downsample,
                    max_sigma_lowres=max_sigma_lowres,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    diffused_patch = diffuse_mask_2d(
                        mask=mask_patch,
                        u0=source_patch,
                        total_time=diffusion_total_time,
                        diffusion_coeff=diffusion_coefficient,
                        barrier_alpha=diffusion_barrier_alpha,
                        leaky_outer_boundary=True,
                        use_gpu=use_gpu,
                        mic_spacing=mic_spacing,
                        downsample=diffusion_downsample,
                        max_sigma_lowres=max_sigma_lowres,
                    )

        local_ms = _block_reduce_local_to_msi(
            diffused_patch=diffused_patch,
            scaling_factor=scaling_factor,
        )

        top_ms = top // scaling_factor
        left_ms = left // scaling_factor
        bottom_ms = min(top_ms + local_ms.shape[0], ms_canvas.shape[0])
        right_ms = min(left_ms + local_ms.shape[1], ms_canvas.shape[1])

        h = bottom_ms - top_ms
        w = right_ms - left_ms
        if h > 0 and w > 0:
            ms_canvas[top_ms:bottom_ms, left_ms:right_ms] += local_ms[:h, :w]

    return ms_canvas.astype(np.float32, copy=False)

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
    target_canvas=None,
    local_diffusion=True,
    diffusion_crop_n_sigma=4.0,
    local_diffusion_verbose=False,
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

    if (
        local_diffusion
        and ms_spacing is not None
        and diffusion_total_time > 0
        and diffusion_coefficient > 0
    ):
        return generate_msi_from_mask_slice_local_diffusion(
            mask_slice=mask_slice,
            sigmas=sigmas,
            intensities=intensities,
            diffusion_total_time=diffusion_total_time,
            diffusion_coefficient=diffusion_coefficient,
            diffusion_barrier_alpha=diffusion_barrier_alpha,
            mic_spacing=mic_spacing,
            leaky_outer_boundary=leaky_outer_boundary,
            use_gpu=use_gpu,
            ms_spacing=ms_spacing,
            diffusion_downsample=diffusion_downsample,
            max_sigma_lowres=max_sigma_lowres,
            target_canvas=target_canvas,
            diffusion_crop_n_sigma=diffusion_crop_n_sigma,
            verbose=local_diffusion_verbose,
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

        local_target = None
        local_sf = None

        if target_canvas is not None and ms_spacing is not None:
            local_sf = int(ms_spacing / mic_spacing)
            ty0 = y0 // local_sf
            ty1 = int(np.ceil(y1 / local_sf))
            tx0 = x0 // local_sf
            tx1 = int(np.ceil(x1 / local_sf))
            local_target = np.asarray(target_canvas[ty0:ty1, tx0:tx1], dtype=np.float32)

        center = _choose_cell_center_from_mask_and_target(
            local_mask=local_mask,
            target_patch=local_target,
            scaling_factor=local_sf,
        )

        if center is None:
            cy, cx = coords.mean(axis=0)
        else:
            cy, cx = center

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