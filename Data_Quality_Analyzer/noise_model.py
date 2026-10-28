"""Gaussian noise postprocessing with optional residual-rank matching."""
import numpy as np


def apply_noise_model(canvas, target, seed=None, clip_nonnegative=True, rank_match=True):
    canvas = np.asarray(canvas, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if canvas.shape != target.shape:
        raise ValueError(
            f"canvas and target must have the same shape. "
            f"Got {canvas.shape} and {target.shape}."
        )

    rng = np.random.default_rng(seed)

    if canvas.ndim == 2:
        result = _apply_noise_model_single_channel(
            canvas,
            target,
            rng=rng,
            rank_match=rank_match,
        )

        if clip_nonnegative:
            result = np.clip(result, 0, None)

        return result.astype(np.float32)

    if canvas.ndim == 3:
        result = np.empty_like(canvas, dtype=np.float32)

        for c in range(canvas.shape[-1]):
            result[:, :, c] = _apply_noise_model_single_channel(
                canvas[:, :, c],
                target[:, :, c],
                rng=rng,
                rank_match=rank_match,
            )

        if clip_nonnegative:
            result = np.clip(result, 0, None)

        return result.astype(np.float32)

    raise ValueError("canvas and target must be either 2D or 3D arrays.")


def _apply_noise_model_single_channel(canvas_ch, target_ch, rng, rank_match=True):
    residual = target_ch - canvas_ch
    residual_flat = residual.ravel()

    mu = float(residual_flat.mean())
    sigma = float(residual_flat.std())

    if sigma <= 0:
        return canvas_ch + mu

    gaussian = rng.normal(
        loc=mu,
        scale=sigma,
        size=residual_flat.shape,
    ).astype(np.float32)

    if not rank_match:
        return canvas_ch + gaussian.reshape(residual.shape)

    residual_order = np.argsort(residual_flat)
    gaussian_sorted = np.sort(gaussian)

    assigned_noise = np.empty_like(gaussian, dtype=np.float32)
    assigned_noise[residual_order] = gaussian_sorted

    return canvas_ch + assigned_noise.reshape(residual.shape)
