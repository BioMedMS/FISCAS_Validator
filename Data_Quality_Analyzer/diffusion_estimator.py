import numpy as np
from scipy.ndimage import distance_transform_edt

def estimate_diffusion_parameter(msi, msi_generated, canvas=None, scaling_factor=None, eps=1e-12):
    """
    Compare target MSI and generated MSI by channel.

    Returns one leakage/mismatch score per channel.

    Kept for API compatibility.
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


def _safe_stat(values, fallback=0.0, method="median"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback)

    if method == "mean":
        return float(np.mean(values))

    return float(np.median(values))


def _prepare_estimation_distribution(distribution, estimation_power):
    u = np.asarray(distribution, dtype=np.float64)

    if u.ndim != 2:
        raise ValueError("distribution must be a 2D array.")

    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.clip(u, 0.0, None)

    power = float(estimation_power)

    if power <= 0:
        raise ValueError("estimation_power must be positive.")

    if power != 1.0:
        u = np.power(u, power)

    return u


def estimate_mu_from_distribution(
    distribution,
    mask_fraction,
    scaling_factor,
    median_cell_radius_ms,
    estimation_power=2.0,
    cell_fraction_threshold=0.5,
    background_fraction_threshold=0.05,
    near_background_radius_ms=1.5,
    far_background_radius_ms=3.0,
    intracellular_diffusion_fraction=0.25,
    max_mu=None,
    eps=1e-12,
):
    """
    Estimate diffusion strength D*T and foreground/background permeability alpha
    directly on the native MSI/MALDI grid.

    Background-ring distances are measured in native MSI pixels.

    Interpretation:
        mu_est:
            Returned in high-resolution mask-pixel units, because the generator
            diffuses on the high-resolution canvas with mic_spacing=1.

        barrier_alpha_est:
            0.0 -> zero permeability into background.
            1.0 -> full permeability into background.

    Estimation logic:
        - The MSI channel is optionally power-transformed for estimation only.
        - Cell signal is measured on MSI pixels substantially covered by cells.
        - Near background is measured in native MSI pixels near cell-covered pixels.
        - Far background is measured farther away, also in native MSI pixels.
        - If near background does not decay toward far background, alpha is 0.
        - mu is based on median cell size in MSI pixels and intracellular
          homogeneity.
    """
    raw = np.asarray(distribution, dtype=np.float64)
    frac = np.asarray(mask_fraction, dtype=np.float64)

    if raw.ndim != 2:
        raise ValueError("distribution must be a 2D array.")

    if frac.ndim != 2:
        raise ValueError("mask_fraction must be a 2D array.")

    if raw.shape != frac.shape:
        raise ValueError(
            "distribution and mask_fraction must have the same shape. "
            f"Got {raw.shape} and {frac.shape}."
        )

    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive.")

    if far_background_radius_ms <= near_background_radius_ms:
        raise ValueError("far_background_radius_ms must be greater than near_background_radius_ms.")

    u = _prepare_estimation_distribution(
        distribution=raw,
        estimation_power=estimation_power,
    )

    cell_fraction_threshold = float(cell_fraction_threshold)
    background_fraction_threshold = float(background_fraction_threshold)

    if not (0.0 < cell_fraction_threshold <= 1.0):
        raise ValueError("cell_fraction_threshold must be in (0, 1].")

    if not (0.0 <= background_fraction_threshold < cell_fraction_threshold):
        raise ValueError(
            "background_fraction_threshold must be >= 0 and smaller than cell_fraction_threshold."
        )

    cell_pixels = frac >= cell_fraction_threshold
    background_pixels = frac <= background_fraction_threshold
    mixed_pixels = (frac > background_fraction_threshold) & (frac < cell_fraction_threshold)

    if not np.any(cell_pixels):
        return {
            "mu_est": 0.0,
            "sigma_px_est": 0.0,
            "sigma_ms_est": 0.0,
            "barrier_alpha_est": 0.0,
            "method": "native_msi_grid_boundary_alpha",
            "warning": "No MSI pixels exceeded cell_fraction_threshold.",
            "estimation_power": float(estimation_power),
            "cell_fraction_threshold": float(cell_fraction_threshold),
            "background_fraction_threshold": float(background_fraction_threshold),
            "near_background_radius_ms": float(near_background_radius_ms),
            "far_background_radius_ms": float(far_background_radius_ms),
            "intracellular_diffusion_fraction": float(intracellular_diffusion_fraction),
            "median_cell_radius_ms": float(median_cell_radius_ms),
        }

    if not np.any(background_pixels):
        background_pixels = ~cell_pixels

    global_background = _safe_stat(
        u[background_pixels],
        fallback=float(np.median(u)),
        method="median",
    )

    cell_values = u[cell_pixels]
    cell_mean = float(np.mean(cell_values)) if cell_values.size else 0.0
    cell_std = float(np.std(cell_values)) if cell_values.size else 0.0
    cell_median = _safe_stat(cell_values, fallback=global_background, method="median")
    cell_cv = cell_std / max(abs(cell_mean), eps)

    homogeneity = 1.0 / (1.0 + cell_cv)

    sigma_ms_est = (
        float(median_cell_radius_ms)
        * float(intracellular_diffusion_fraction)
        * float(homogeneity)
    )

    sigma_ms_est = max(float(sigma_ms_est), 0.0)
    sigma_px_est = sigma_ms_est * float(scaling_factor)
    mu_est = float((sigma_px_est ** 2) / 2.0)

    if max_mu is not None:
        mu_est = float(min(mu_est, float(max_mu))) * np.exp(0.34 * max(0.0, mu_est - 5.0))
        sigma_px_est = float(np.sqrt(2.0 * mu_est)) if mu_est > 0 else 0.0
        sigma_ms_est = sigma_px_est / float(scaling_factor)

    dist_to_cell_ms = distance_transform_edt(~cell_pixels)

    near_background = (
        background_pixels
        & (dist_to_cell_ms > 0.0)
        & (dist_to_cell_ms <= float(near_background_radius_ms))
    )

    far_background = (
        background_pixels
        & (dist_to_cell_ms > float(near_background_radius_ms))
        & (dist_to_cell_ms <= float(far_background_radius_ms))
    )

    if not np.any(near_background) and np.any(mixed_pixels):
        near_background = mixed_pixels

    if not np.any(far_background):
        far_background = background_pixels & (dist_to_cell_ms > float(near_background_radius_ms))

    if not np.any(far_background):
        far_background = background_pixels

    near_background_median = _safe_stat(
        u[near_background],
        fallback=global_background,
        method="median",
    )

    far_background_median = _safe_stat(
        u[far_background],
        fallback=global_background,
        method="median",
    )

    edge_drop = float(cell_median - near_background_median)
    background_drop = float(near_background_median - far_background_median)

    if edge_drop <= eps:
        barrier_alpha_est = 0.0
    elif background_drop <= eps:
        barrier_alpha_est = 0.0
    else:
        barrier_alpha_est = float(np.clip(background_drop / max(edge_drop, eps), 0.0, 1.0))

    return {
        "mu_est": float(mu_est),
        "sigma_px_est": float(sigma_px_est),
        "sigma_ms_est": float(sigma_ms_est),
        "barrier_alpha_est": float(barrier_alpha_est),
        "method": "native_msi_grid_boundary_alpha",
        "estimation_power": float(estimation_power),
        "cell_fraction_threshold": float(cell_fraction_threshold),
        "background_fraction_threshold": float(background_fraction_threshold),
        "near_background_radius_ms": float(near_background_radius_ms),
        "far_background_radius_ms": float(far_background_radius_ms),
        "intracellular_diffusion_fraction": float(intracellular_diffusion_fraction),
        "median_cell_radius_ms": float(median_cell_radius_ms),
        "cell_mean": float(cell_mean),
        "cell_median": float(cell_median),
        "cell_std": float(cell_std),
        "cell_cv": float(cell_cv),
        "cell_homogeneity": float(homogeneity),
        "near_background_median": float(near_background_median),
        "far_background_median": float(far_background_median),
        "edge_drop": float(edge_drop),
        "background_drop": float(background_drop),
        "global_background": float(global_background),
        "n_cell_pixels": int(np.count_nonzero(cell_pixels)),
        "n_background_pixels": int(np.count_nonzero(background_pixels)),
        "n_mixed_pixels": int(np.count_nonzero(mixed_pixels)),
        "n_near_background_pixels": int(np.count_nonzero(near_background)),
        "n_far_background_pixels": int(np.count_nonzero(far_background)),
    }


def estimate_background_stats(distribution, mask_fraction, background_fraction_threshold=0.05):
    """
    Estimate target background statistics from native MSI pixels outside cells.
    """
    u = np.asarray(distribution, dtype=np.float64)
    frac = np.asarray(mask_fraction, dtype=np.float64)

    if u.shape != frac.shape:
        raise ValueError("distribution and mask_fraction must have the same shape.")

    background = u[frac <= float(background_fraction_threshold)]

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