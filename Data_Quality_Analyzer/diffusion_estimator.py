import numpy as np
from canvas_generation import diffuse_mask_2d

def estimate_diffusion_parameter(msi, msi_generated, canvas, scaling_factor):
    signal_leaked = []
    for i, bin in enumerate(msi):
        total_signal_gt = np.sum(bin)
        total_signal_sim = np.sum(msi_generated[:,:,i])
        leaked_signal_percentage = (total_signal_gt-total_signal_sim)/total_signal_gt
        if leaked_signal_percentage >= 0.5:
            signal_leaked.append(1)
        else:
            signal_leaked.append(leaked_signal_percentage*2)
    
    return signal_leaked

    for i, bin in enumerate(msi):
        bin_generated = msi_generated[i]
        mu_candidates = []
        while mu_candidates:
            pass

def estimate_mu_from_distribution(distribution, patch_mask, eps=1e-12):
    u = np.asarray(distribution, dtype=float)
    mask = np.asarray(patch_mask, dtype=bool)

    if u.ndim != 2:
        raise ValueError("distribution must be a 2D array.")
    if mask.shape != u.shape:
        raise ValueError("patch_mask must have the same shape as distribution.")
    if mask.all() or (~mask).all():
        raise ValueError("patch_mask must contain both patch and background cells.")

    n, m = u.shape

    patch_values = u[mask]
    background_values = u[~mask]

    patch_mean = patch_values.mean()
    background_mean = background_values.mean()
    global_mean = u.mean()

    contrast_remaining = abs(patch_mean - background_mean)

    patch_fraction = mask.mean()

    if patch_fraction <= eps:
        raise ValueError("Patch fraction is too small.")

    z_est = global_mean / patch_fraction
    contrast_initial_est = abs(z_est - 0.0)

    if contrast_initial_est <= eps:
        raise ValueError("Initial contrast estimate is zero or too small.")

    r = contrast_remaining / contrast_initial_est
    r = np.clip(r, eps, 1.0)

    lambda_1 = (np.pi ** 2) / (max(n, m) ** 2)

    mu_est = (1.0 - r) / (r * lambda_1)

    return {
        "mu_est": mu_est,
        "r": r,
        "lambda_1": lambda_1,
        "n": n,
        "m": m,
        "patch_mean": patch_mean,
        "background_mean": background_mean,
        "global_mean": global_mean,
        "contrast_remaining": contrast_remaining,
        "contrast_initial_est": contrast_initial_est,
    }