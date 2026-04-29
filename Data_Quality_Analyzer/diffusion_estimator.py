import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit


def estimate_diffusion_parameter(msi, msi_generated, canvas=None, scaling_factor=None, eps=1e-12):
    """
    Compare target MSI and generated MSI by channel.

    Returns one leakage/mismatch score per channel.

    A value near 0 means generated total signal matches target total.
    A value near 1 means generated total signal is badly below target total.

    Parameters
    ----------
    msi:
        Target MSI array, shape (H, W, C).
    msi_generated:
        Generated MSI array, shape (H, W, C).
    canvas:
        Kept for API compatibility. Not used.
    scaling_factor:
        Kept for API compatibility. Not used.
    eps:
        Numerical safety value.
    """
    target = np.asarray(msi, dtype=np.float64)
    generated = np.asarray(msi_generated, dtype=np.float64)

    if target.ndim == 2:
        target = target[:, :, None]

    if generated.ndim == 2:
        generated = generated[:, :, None]

    if target.shape != generated.shape:
        raise ValueError(
            f"msi and msi_generated must have the same shape. "
            f"Got {target.shape} and {generated.shape}."
        )

    scores = []

    for c in range(target.shape[-1]):
        total_signal_gt = float(np.sum(target[:, :, c]))
        total_signal_sim = float(np.sum(generated[:, :, c]))

        if total_signal_gt <= eps:
            scores.append(0.0)
            continue

        leaked_signal_percentage = (total_signal_gt - total_signal_sim) / total_signal_gt

        if leaked_signal_percentage >= 0.5:
            score = 1.0
        else:
            score = leaked_signal_percentage * 2.0

        scores.append(float(np.clip(score, 0.0, 1.0)))

    return scores


def _gaussian_halo_model(r, amplitude, sigma):
    """
    Outside-cell halo approximation.

    excess(r) = amplitude * exp(-r^2 / (2 sigma^2))

    The generator uses:
        sigma_px = sqrt(2 * D * T)

    So:
        D * T = sigma_px^2 / 2
    """
    return amplitude * np.exp(-(r ** 2) / (2.0 * sigma ** 2))


def _as_labeled_mask(patch_mask):
    """
    Convert patch_mask into a labeled integer mask.

    If patch_mask is boolean:
        True is treated as one foreground object.

    If patch_mask is integer-labeled:
        labels > 0 are treated as individual cells.
    """
    mask = np.asarray(patch_mask)

    if mask.ndim != 2:
        raise ValueError("patch_mask must be a 2D array.")

    if mask.dtype == bool:
        labeled = np.zeros(mask.shape, dtype=np.int32)
        labeled[mask] = 1
        return labeled

    return mask.astype(np.int32, copy=False)


def _robust_background(values, fallback, method="median"):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return float(fallback)

    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback)

    if method == "mean":
        return float(np.mean(values))

    return float(np.median(values))


def _fit_single_cell_halo(
    u,
    labeled_mask,
    cell_id,
    global_background,
    max_radius_px,
    background_inner_radius_px,
    background_outer_radius_px,
    min_ring_pixels,
    min_points,
    min_excess_fraction,
):
    """
    Estimate halo sigma for one cell by fitting positive outside-cell excess
    as a function of distance from the cell boundary.
    """
    cell = labeled_mask == cell_id

    if not np.any(cell):
        return None

    # Distance from each outside pixel to the nearest pixel of this cell.
    dist_outside = distance_transform_edt(~cell)

    outside_this_cell = ~cell

    # Use only true background pixels for halo/background estimation.
    # This avoids measuring other cells as "halo".
    true_background = labeled_mask == 0

    background_annulus = (
        true_background
        & outside_this_cell
        & (dist_outside >= background_inner_radius_px)
        & (dist_outside <= background_outer_radius_px)
    )

    local_background = _robust_background(
        u[background_annulus],
        fallback=global_background,
        method="median",
    )

    radii = []
    excess_means = []
    ring_counts = []

    for r in range(1, int(max_radius_px) + 1):
        ring = (
            true_background
            & outside_this_cell
            & (dist_outside >= r - 1)
            & (dist_outside < r)
        )

        count = int(np.count_nonzero(ring))

        if count < min_ring_pixels:
            continue

        ring_mean = float(np.mean(u[ring]))
        excess = ring_mean - local_background

        radii.append(float(r - 0.5))
        excess_means.append(float(excess))
        ring_counts.append(count)

    if len(radii) < min_points:
        return None

    radii = np.asarray(radii, dtype=np.float64)
    excess_means = np.asarray(excess_means, dtype=np.float64)
    ring_counts = np.asarray(ring_counts, dtype=np.float64)

    finite = np.isfinite(radii) & np.isfinite(excess_means)

    radii = radii[finite]
    excess_means = excess_means[finite]
    ring_counts = ring_counts[finite]

    if radii.size < min_points:
        return None

    peak_excess = float(np.max(excess_means))

    if peak_excess <= 0:
        return None

    # Keep only meaningful positive halo points.
    keep = excess_means > (min_excess_fraction * peak_excess)

    if np.count_nonzero(keep) < min_points:
        return None

    r_fit = radii[keep]
    y_fit = excess_means[keep]
    w_fit = ring_counts[keep]

    # Weighted fit: rings with more pixels are more stable.
    sigma_y = 1.0 / np.sqrt(np.maximum(w_fit, 1.0))

    initial_sigma = max(2.0, min(float(max_radius_px) / 4.0, float(max_radius_px)))
    initial_amplitude = peak_excess

    try:
        popt, _ = curve_fit(
            _gaussian_halo_model,
            r_fit,
            y_fit,
            p0=(initial_amplitude, initial_sigma),
            bounds=(
                (0.0, 0.5),
                (np.inf, float(max_radius_px) * 2.0),
            ),
            sigma=sigma_y,
            absolute_sigma=False,
            maxfev=5000,
        )
    except Exception:
        return None

    amplitude_hat = float(popt[0])
    sigma_hat = float(popt[1])

    if not np.isfinite(amplitude_hat) or not np.isfinite(sigma_hat):
        return None

    if amplitude_hat <= 0 or sigma_hat <= 0:
        return None

    mu_hat = (sigma_hat ** 2) / 2.0

    fitted = _gaussian_halo_model(r_fit, amplitude_hat, sigma_hat)

    residual = y_fit - fitted
    mae = float(np.mean(np.abs(residual)))
    mean_y = float(np.mean(np.abs(y_fit)))
    relative_mae = mae / max(mean_y, 1e-12)

    return {
        "cell_id": int(cell_id),
        "mu": float(mu_hat),
        "sigma_px": float(sigma_hat),
        "amplitude": float(amplitude_hat),
        "local_background": float(local_background),
        "peak_excess": float(peak_excess),
        "n_fit_points": int(r_fit.size),
        "relative_mae": float(relative_mae),
    }


def estimate_mu_from_distribution(
    distribution,
    patch_mask,
    max_radius_px=120,
    background_inner_radius_px=80,
    background_outer_radius_px=160,
    min_ring_pixels=10,
    min_points=5,
    min_excess_fraction=0.05,
    max_relative_mae=2.0,
    min_cells=3,
    eps=1e-12,
):
    """
    Estimate effective diffusion strength D*T from local halo rings around cells.

    This replaces the old global cell-vs-background contrast estimator.

    The old estimator was unsuitable for your data because the biological
    background is approximately homogeneous noise/baseline, not purely leaked
    cell signal. This estimator instead looks for a local excess halo around
    segmented cells after subtracting a local background annulus.

    Model
    -----
    observed_signal = baseline + local_diffusion_halo + noise

    For each cell:
        excess(r) = mean(signal in ring r) - local_background

    Fit:
        excess(r) = A * exp(-r^2 / (2 sigma^2))

    Your generator uses:
        sigma_px = sqrt(2 * D * T)

    Therefore:
        mu_est = D * T = sigma_px^2 / 2

    Parameters
    ----------
    distribution:
        2D high-resolution MSI channel.
    patch_mask:
        2D segmentation mask, same shape as distribution.
        Can be boolean or labeled integer mask.
    max_radius_px:
        Maximum outside-cell ring radius used for halo fitting.
    background_inner_radius_px:
        Inner radius for local background annulus.
    background_outer_radius_px:
        Outer radius for local background annulus.
    min_ring_pixels:
        Minimum number of pixels required in a ring.
    min_points:
        Minimum number of radial points required to fit one cell.
    min_excess_fraction:
        Keep ring points whose excess is at least this fraction of the peak excess.
    max_relative_mae:
        Reject very poor single-cell fits.
    min_cells:
        Minimum number of usable fitted cells desired. If fewer are found, a warning
        is included but the median of available cells is still returned.
    eps:
        Numerical safety value.

    Returns
    -------
    dict with at least:
        mu_est:
            Estimated D*T.
        sigma_px_est:
            Estimated Gaussian sigma in full-resolution pixels.
        method:
            "local_annular_halo_fit"
    """
    u = np.asarray(distribution, dtype=np.float64)
    labeled_mask = _as_labeled_mask(patch_mask)

    if u.ndim != 2:
        raise ValueError("distribution must be a 2D array.")

    if labeled_mask.shape != u.shape:
        raise ValueError("patch_mask must have the same shape as distribution.")

    if max_radius_px <= 0:
        raise ValueError("max_radius_px must be positive.")

    if background_outer_radius_px <= background_inner_radius_px:
        raise ValueError("background_outer_radius_px must be greater than background_inner_radius_px.")

    if not np.any(labeled_mask > 0):
        raise ValueError("patch_mask must contain at least one foreground cell label > 0.")

    if not np.any(labeled_mask == 0):
        raise ValueError("patch_mask must contain background pixels labeled 0.")

    cell_ids = np.unique(labeled_mask)
    cell_ids = cell_ids[cell_ids > 0]

    background_values = u[labeled_mask == 0]
    global_background = _robust_background(background_values, fallback=float(np.median(u)), method="median")
    global_background_mean = float(np.mean(background_values)) if background_values.size else float(np.mean(u))
    global_background_std = float(np.std(background_values)) if background_values.size else float(np.std(u))

    fits = []
    skipped_cells = 0
    rejected_cells = 0

    for cell_id in cell_ids:
        fit = _fit_single_cell_halo(
            u=u,
            labeled_mask=labeled_mask,
            cell_id=int(cell_id),
            global_background=global_background,
            max_radius_px=max_radius_px,
            background_inner_radius_px=background_inner_radius_px,
            background_outer_radius_px=background_outer_radius_px,
            min_ring_pixels=min_ring_pixels,
            min_points=min_points,
            min_excess_fraction=min_excess_fraction,
        )

        if fit is None:
            skipped_cells += 1
            continue

        if fit["relative_mae"] > max_relative_mae:
            rejected_cells += 1
            continue

        fits.append(fit)

    if len(fits) == 0:
        return {
            "mu_est": 0.0,
            "sigma_px_est": 0.0,
            "n_used_cells": 0,
            "n_skipped_cells": int(skipped_cells),
            "n_rejected_cells": int(rejected_cells),
            "n_total_cells": int(len(cell_ids)),
            "global_background": float(global_background),
            "global_background_mean": float(global_background_mean),
            "global_background_std": float(global_background_std),
            "method": "local_annular_halo_fit",
            "warning": "No valid local halo fits found. Try smaller/larger radii or lower min_excess_fraction.",
        }

    mus = np.asarray([fit["mu"] for fit in fits], dtype=np.float64)
    sigmas = np.asarray([fit["sigma_px"] for fit in fits], dtype=np.float64)
    amplitudes = np.asarray([fit["amplitude"] for fit in fits], dtype=np.float64)
    relative_maes = np.asarray([fit["relative_mae"] for fit in fits], dtype=np.float64)

    # Robust aggregate estimate.
    mu_est = float(np.median(mus))
    sigma_px_est = float(np.median(sigmas))

    result = {
        "mu_est": mu_est,
        "sigma_px_est": sigma_px_est,
        "n_used_cells": int(len(fits)),
        "n_skipped_cells": int(skipped_cells),
        "n_rejected_cells": int(rejected_cells),
        "n_total_cells": int(len(cell_ids)),
        "global_background": float(global_background),
        "global_background_mean": float(global_background_mean),
        "global_background_std": float(global_background_std),
        "median_amplitude": float(np.median(amplitudes)),
        "median_relative_mae": float(np.median(relative_maes)),
        "mu_mean": float(np.mean(mus)),
        "mu_median": float(np.median(mus)),
        "mu_p10": float(np.percentile(mus, 10)),
        "mu_p25": float(np.percentile(mus, 25)),
        "mu_p75": float(np.percentile(mus, 75)),
        "mu_p90": float(np.percentile(mus, 90)),
        "sigma_mean": float(np.mean(sigmas)),
        "sigma_median": float(np.median(sigmas)),
        "sigma_p10": float(np.percentile(sigmas, 10)),
        "sigma_p25": float(np.percentile(sigmas, 25)),
        "sigma_p75": float(np.percentile(sigmas, 75)),
        "sigma_p90": float(np.percentile(sigmas, 90)),
        "method": "local_annular_halo_fit",
    }

    if len(fits) < min_cells:
        result["warning"] = (
            f"Only {len(fits)} valid cells were fitted. "
            f"Estimate may be unstable. Try adjusting max_radius_px, "
            f"background annulus radii, or min_excess_fraction."
        )

    return result


def estimate_background_stats(distribution, patch_mask):
    """
    Estimate target background statistics from pixels outside segmented cells.

    This is useful if you later want to add a separate background/noise model
    after generating the cell diffusion signal.
    """
    u = np.asarray(distribution, dtype=np.float64)
    labeled_mask = _as_labeled_mask(patch_mask)

    if u.shape != labeled_mask.shape:
        raise ValueError("distribution and patch_mask must have the same shape.")

    background = u[labeled_mask == 0]

    if background.size == 0:
        background = u.ravel()

    return {
        "background_mean": float(np.mean(background)),
        "background_median": float(np.median(background)),
        "background_std": float(np.std(background)),
        "background_p05": float(np.percentile(background, 5)),
        "background_p25": float(np.percentile(background, 25)),
        "background_p75": float(np.percentile(background, 75)),
        "background_p95": float(np.percentile(background, 95)),
    }