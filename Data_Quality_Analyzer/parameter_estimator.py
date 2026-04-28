import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
from concurrent.futures import ThreadPoolExecutor

from canvas_generation import generate_msi_from_mask_slice
from diffusion_estimator import estimate_diffusion_parameter, estimate_mu_from_distribution
from error_estimation import symmetric_l1_ratio


def crop_canvas_and_msi(mask, msi, top, bottom, left, right):
    scaling_factor = int(mask.shape[0] / msi.shape[0])

    new_mask = mask[
        left * scaling_factor:mask.shape[0] - right * scaling_factor,
        top * scaling_factor:mask.shape[1] - bottom * scaling_factor,
    ]

    new_msi = msi[
        left:msi.shape[0] - right,
        top:msi.shape[1] - bottom,
        :,
    ]

    return new_mask, new_msi


def get_cell_bounding_boxes(mask):
    """
    Much faster than scanning every row/column for every cell.
    Returns dict: cell_id -> (top, bottom, left, right), bottom/right exclusive.
    """
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]

    boxes = {}

    for cell_id in cell_ids:
        ys, xs = np.where(mask == cell_id)

        if ys.size == 0:
            continue

        top = int(ys.min())
        bottom = int(ys.max()) + 1
        left = int(xs.min())
        right = int(xs.max()) + 1

        boxes[int(cell_id)] = (top, bottom, left, right)

    return cell_ids.astype(np.int32), boxes


def get_cell_neighbourhood_from_box(canvas, box, padding, scaling_factor):
    top0, bottom0, left0, right0 = box

    left = int(max(left0 - left0 % scaling_factor - padding * scaling_factor, 0))
    top = int(max(top0 - top0 % scaling_factor - padding * scaling_factor, 0))

    right = int(
        min(
            right0 - right0 % scaling_factor + padding * scaling_factor + scaling_factor,
            canvas.shape[1],
        )
    )

    bottom = int(
        min(
            bottom0 - bottom0 % scaling_factor + padding * scaling_factor + scaling_factor,
            canvas.shape[0],
        )
    )

    return canvas[top:bottom, left:right], (left, right, top, bottom)


def get_individual_cell_canvases(mask, msi_canvas, scaling_factor, padding):
    scaling_factor = int(scaling_factor)

    cell_indexes, boxes = get_cell_bounding_boxes(mask)

    masks = []
    targets = [[] for _ in range(msi_canvas.shape[-1])]

    for cell_id in cell_indexes:
        crop_mask, (left_hr, right_hr, top_hr, bottom_hr) = get_cell_neighbourhood_from_box(
            mask,
            boxes[int(cell_id)],
            padding,
            scaling_factor,
        )

        masks.append(crop_mask)

        left = left_hr // scaling_factor
        right = right_hr // scaling_factor
        top = top_hr // scaling_factor
        bottom = bottom_hr // scaling_factor

        msi_crop = msi_canvas[top:bottom, left:right, :]

        for channel_idx in range(msi_canvas.shape[-1]):
            targets[channel_idx].append(msi_crop[:, :, channel_idx])

    return cell_indexes, masks, targets


def block_fraction_from_hr_mask(mask_i, scaling_factor, target_shape):
    """
    Computes the fraction of each MSI pixel covered by the high-res cell mask.

    This replaces:

        target_hr = np.repeat(np.repeat(cell_target, sf, axis=0), sf, axis=1)
        target_hr[mask_i].sum() / sf**2

    with a much faster block-reduce on the boolean mask.
    """
    sf = int(scaling_factor)

    h_hr, w_hr = mask_i.shape
    pad_h = int(np.ceil(h_hr / sf) * sf)
    pad_w = int(np.ceil(w_hr / sf) * sf)

    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:h_hr, :w_hr] = mask_i.astype(np.float32)

    counts = skimage.measure.block_reduce(
        padded,
        block_size=(sf, sf),
        func=np.sum,
    ).astype(np.float32)

    fractions = counts / float(sf * sf)

    return fractions[:target_shape[0], :target_shape[1]]


def build_initial_sigmas_and_intensities(
    cell_indexes,
    cell_masks,
    cell_targets,
    scaling_factor,
    num_channels,
):
    sigmas = [[] for _ in range(num_channels)]
    intensities = [[] for _ in range(num_channels)]

    # Precompute masks and block fractions once per cell.
    cell_fractions = []

    for i, cell_id in enumerate(cell_indexes):
        mask_i = cell_masks[i] == cell_id

        # Use channel 0 shape as spatial reference.
        target_shape = cell_targets[0][i].shape

        frac = block_fraction_from_hr_mask(
            mask_i=mask_i,
            scaling_factor=scaling_factor,
            target_shape=target_shape,
        )

        cell_fractions.append(frac)

    for i in range(len(cell_indexes)):
        frac = cell_fractions[i]

        for c in range(num_channels):
            cell_target = cell_targets[c][i]

            h = min(frac.shape[0], cell_target.shape[0])
            w = min(frac.shape[1], cell_target.shape[1])

            intensity = float(np.sum(cell_target[:h, :w] * frac[:h, :w]))

            intensities[c].append(intensity)
            sigmas[c].append(cell_target.shape[0] * scaling_factor)

    return sigmas, intensities


def render_one_cell_to_msi(cell_mask, cell_id, sigma, intensity, scaling_factor):
    """
    Render a single labeled cell inside a high-res local mask crop and reduce it
    to MSI resolution. Used by the fast sigma optimizer.
    """
    sf = int(scaling_factor)

    coords = np.argwhere(cell_mask == cell_id)

    h_ms = int(np.ceil(cell_mask.shape[0] / sf))
    w_ms = int(np.ceil(cell_mask.shape[1] / sf))

    if coords.shape[0] == 0 or intensity <= 0:
        return np.zeros((h_ms, w_ms), dtype=np.float32)

    sigma = max(float(sigma), 1e-6)

    cy, cx = coords.mean(axis=0)

    yy = coords[:, 0].astype(np.float32)
    xx = coords[:, 1].astype(np.float32)

    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    values = np.exp(-((distance / sigma) ** 2)) + 0.1

    hr = np.zeros(cell_mask.shape, dtype=np.float32)
    hr[coords[:, 0], coords[:, 1]] = values.astype(np.float32)

    total = float(hr.sum())

    if total > 0:
        hr *= float(intensity) / total

    pad_h = h_ms * sf
    pad_w = w_ms * sf

    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:hr.shape[0], :hr.shape[1]] = hr

    ms = skimage.measure.block_reduce(
        padded,
        block_size=(sf, sf),
        func=np.sum,
    ).astype(np.float32)

    return ms


def crop_same_shape(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def optimize_single_sigma(
    i,
    c,
    cell_indexes,
    cell_masks,
    cell_targets,
    sigmas,
    intensities,
    scaling_factor,
    global_pos,
):
    """
    Optimized sigma search.

    Old version called generate_msi_from_mask_slice for every sigma trial.
    This version:
      1. Renders all local cells once.
      2. Keeps the neighbor contribution fixed.
      3. Re-renders only the target cell for each sigma trial.
    """
    local_cell_ids = np.unique(cell_masks[i])
    local_cell_ids = local_cell_ids[local_cell_ids > 0]

    valid_local_cell_ids = [
        int(cell_id)
        for cell_id in local_cell_ids
        if int(cell_id) in global_pos
    ]

    if len(valid_local_cell_ids) == 0:
        return float(sigmas[c][i])

    target_cell_id = int(cell_indexes[i])

    if target_cell_id not in valid_local_cell_ids:
        return float(sigmas[c][i])

    target = cell_targets[c][i].astype(np.float32, copy=False)
    target_local_idx = valid_local_cell_ids.index(target_cell_id)

    local_sigmas = [
        float(sigmas[c][global_pos[cell_id]])
        for cell_id in valid_local_cell_ids
    ]

    local_intensities = [
        float(intensities[c][global_pos[cell_id]])
        for cell_id in valid_local_cell_ids
    ]

    rendered_cells = []

    for local_idx, cell_id in enumerate(valid_local_cell_ids):
        rendered = render_one_cell_to_msi(
            cell_mask=cell_masks[i],
            cell_id=cell_id,
            sigma=local_sigmas[local_idx],
            intensity=local_intensities[local_idx],
            scaling_factor=scaling_factor,
        )

        rendered_cells.append(rendered)

    baseline = np.zeros_like(rendered_cells[0], dtype=np.float32)

    for rendered in rendered_cells:
        baseline[:rendered.shape[0], :rendered.shape[1]] += rendered

    target_old = rendered_cells[target_local_idx]

    baseline_cmp, target_cmp = crop_same_shape(baseline, target)
    best_score = symmetric_l1_ratio(target_cmp, baseline_cmp)
    best_sigma = float(local_sigmas[target_local_idx])

    step = max(1, int(round(scaling_factor / 2)))
    max_sigma = max(1, int(round(sigmas[c][i])))

    for sigma in range(1, max_sigma + 1, step):
        target_new = render_one_cell_to_msi(
            cell_mask=cell_masks[i],
            cell_id=target_cell_id,
            sigma=float(sigma),
            intensity=local_intensities[target_local_idx],
            scaling_factor=scaling_factor,
        )

        generated = baseline.copy()

        h_old = min(generated.shape[0], target_old.shape[0])
        w_old = min(generated.shape[1], target_old.shape[1])
        generated[:h_old, :w_old] -= target_old[:h_old, :w_old]

        h_new = min(generated.shape[0], target_new.shape[0])
        w_new = min(generated.shape[1], target_new.shape[1])
        generated[:h_new, :w_new] += target_new[:h_new, :w_new]

        generated_cmp, target_cmp = crop_same_shape(generated, target)

        score = symmetric_l1_ratio(target_cmp, generated_cmp)

        if score < best_score:
            best_score = score
            best_sigma = float(sigma)

    return best_sigma


def parallel_sigma_optimisation(
    cell_indexes,
    cell_masks,
    cell_targets,
    sigmas,
    intensities,
    scaling_factor,
    max_workers=4,
):
    num_channels = len(cell_targets)
    num_cells = len(cell_indexes)

    optimized_sigmas = [
        [float(sigmas[c][i]) for i in range(num_cells)]
        for c in range(num_channels)
    ]

    global_pos = {
        int(cell_id): j
        for j, cell_id in enumerate(cell_indexes)
    }

    def _task(args):
        c, i = args

        best_sigma = optimize_single_sigma(
            i=i,
            c=c,
            cell_indexes=cell_indexes,
            cell_masks=cell_masks,
            cell_targets=cell_targets,
            sigmas=sigmas,
            intensities=intensities,
            scaling_factor=scaling_factor,
            global_pos=global_pos,
        )

        return c, i, best_sigma

    tasks = [
        (c, i)
        for c in range(num_channels)
        for i in range(num_cells)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for c, i, best_sigma in executor.map(_task, tasks):
            optimized_sigmas[c][i] = best_sigma

    print("sigma optimization finished")

    for c in range(num_channels):
        sigma_arr = np.asarray(optimized_sigmas[c], dtype=np.float64)
        print(
            f"Channel {c}: "
            f"mean_sigma={float(np.mean(sigma_arr)):.6f}, "
            f"median_sigma={float(np.median(sigma_arr)):.6f}, "
            f"min_sigma={float(np.min(sigma_arr)):.6f}, "
            f"max_sigma={float(np.max(sigma_arr)):.6f}"
        )

    return optimized_sigmas


def scale_intensities_to_target_totals(intensities, msi):
    num_channels = msi.shape[-1]

    for c in range(num_channels):
        target_total = float(msi[:, :, c].sum())
        current_total = float(np.sum(intensities[c]))

        if current_total > 0:
            intensity_scale = target_total / current_total
            intensities[c] = [float(x) * intensity_scale for x in intensities[c]]
        else:
            intensity_scale = 1.0

        print(
            f"Channel {c}: "
            f"target_total={target_total:.6f}, "
            f"pre_scale_intensity_total={current_total:.6f}, "
            f"intensity_scale={intensity_scale:.6f}, "
            f"post_scale_intensity_total={float(np.sum(intensities[c])):.6f}"
        )

    return intensities


def estimate_channel_mus(msi, mask, scaling_factor, num_channels):
    mus = []

    for c in range(num_channels):
        # This is still memory-heavy, but usually only runs once per channel.
        msi_hr_c = np.repeat(
            np.repeat(msi[:, :, c], scaling_factor, axis=0),
            scaling_factor,
            axis=1,
        )

        mu_c = estimate_mu_from_distribution(msi_hr_c, mask)
        mus.append(mu_c)

    print("mu per channel:")

    for c, mu_c in enumerate(mus):
        print(f"Channel {c}: {mu_c}")

    return mus


def make_channel_generator(
    mask,
    sigmas,
    intensities,
    diffusion_total_times,
    diffusion_coefficients,
    diffusion_barrier_alpha,
    scaling_factor,
    diffusion_downsample=8,
    max_sigma_lowres=15.0,
):
    def _generate_channel(c):
        return generate_msi_from_mask_slice(
            mask_slice=mask,
            sigmas=sigmas[c],
            intensities=intensities[c],
            diffusion_total_time=diffusion_total_times[c],
            diffusion_coefficient=diffusion_coefficients[c],
            diffusion_barrier_alpha=diffusion_barrier_alpha,
            mic_spacing=1.0,
            leaky_outer_boundary=False,
            use_gpu=False,
            ms_spacing=scaling_factor,
            diffusion_downsample=diffusion_downsample,
            max_sigma_lowres=max_sigma_lowres,
        )

    return _generate_channel


def generate_all_channels(num_channels, channel_generator, max_workers=None):
    if max_workers == 1:
        generated_channels = [
            channel_generator(c)
            for c in range(num_channels)
        ]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            generated_channels = list(executor.map(channel_generator, range(num_channels)))

    return np.stack(generated_channels, axis=-1)


def print_channel_stats(msi, generated, label):
    print(f"\n{label}")

    for c in range(msi.shape[-1]):
        print(
            f"Channel {c}: "
            f"target_sum={float(msi[:, :, c].sum()):.6f}, "
            f"target_max={float(msi[:, :, c].max()):.6f}, "
            f"generated_sum={float(generated[:, :, c].sum()):.6f}, "
            f"generated_max={float(generated[:, :, c].max()):.6f}"
        )


def plot_channel_grid(msi, channels_to_show, output_path, title):
    n = len(channels_to_show)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    for ax, ch in zip(axes.flat, channels_to_show):
        im = ax.imshow(msi[:, :, ch], cmap="viridis", aspect="auto")
        ax.set_title(f"Channel {ch}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_channel_comparison(msi, generated, channels_to_show, output_path, title):
    nrows = len(channels_to_show)
    ncols = 2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 3.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, ch in enumerate(channels_to_show):
        target_ch = msi[:, :, ch]
        generated_ch = generated[:, :, ch]

        layer_l1 = symmetric_l1_ratio(target_ch, generated_ch)

        vmin = min(float(target_ch.min()), float(generated_ch.min()))
        vmax = max(float(target_ch.max()), float(generated_ch.max()))

        axes[row, 0].imshow(
            target_ch,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[row, 0].set_title("Target", fontsize=10)
        axes[row, 0].axis("off")

        im1 = axes[row, 1].imshow(
            generated_ch,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[row, 1].set_title("Simulation", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 0].set_ylabel(
            f"Channel {ch}\nL1: {layer_l1:.6f}",
            rotation=0,
            labelpad=55,
            va="center",
            ha="right",
            fontsize=10,
        )

        fig.colorbar(
            im1,
            ax=axes[row, :],
            location="right",
            shrink=0.85,
            pad=0.02,
        )

    fig.suptitle(title, fontsize=14)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_noise_model(canvas, target, seed=None, clip_nonnegative=True, rank_match=True):
    """
    Multi-channel noise model.

    rank_match=True reproduces your existing rank-matched residual behavior.
    rank_match=False is much faster because it skips two full-array sorts.
    """
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


if __name__ == "__main__":
    original_channels_to_show = range(88, 89)

    mask = np.load("input/seg.npy").astype(np.int32)
    msi = np.load("input/msi_data.npy")[:, :, original_channels_to_show]

    channels_to_show = range(msi.shape[-1])

    p = np.percentile(msi, 99, axis=(0, 1), keepdims=True)
    msi = np.clip(msi, 0, p)

    # Optional crop
    # mask, msi = crop_canvas_and_msi(
    #     mask,
    #     msi,
    #     top=110,
    #     bottom=110,
    #     left=100,
    #     right=100,
    # )

    plot_channel_grid(
        msi=msi,
        channels_to_show=channels_to_show,
        output_path="output/target_msi.png",
        title="Target MSI",
    )

    scaling_factor = int(mask.shape[0] // msi.shape[0])

    diffusion_barrier_alpha = 0.0

    cell_indexes, cell_masks, cell_targets = get_individual_cell_canvases(
        mask=mask,
        msi_canvas=msi,
        scaling_factor=scaling_factor,
        padding=2,
    )

    num_channels = len(cell_targets)

    sigmas, intensities = build_initial_sigmas_and_intensities(
        cell_indexes=cell_indexes,
        cell_masks=cell_masks,
        cell_targets=cell_targets,
        scaling_factor=scaling_factor,
        num_channels=num_channels,
    )

    sigmas = parallel_sigma_optimisation(
        cell_indexes=cell_indexes,
        cell_masks=cell_masks,
        cell_targets=cell_targets,
        sigmas=sigmas,
        intensities=intensities,
        scaling_factor=scaling_factor,
        max_workers=4,
    )

    diffusion_total_times = [0.0] * num_channels
    diffusion_coefficients = [0.0] * num_channels

    initial_generator = make_channel_generator(
        mask=mask,
        sigmas=sigmas,
        intensities=intensities,
        diffusion_total_times=diffusion_total_times,
        diffusion_coefficients=diffusion_coefficients,
        diffusion_barrier_alpha=diffusion_barrier_alpha,
        scaling_factor=scaling_factor,
        diffusion_downsample=8,
        max_sigma_lowres=15.0,
    )

    generated = generate_all_channels(
        num_channels=num_channels,
        channel_generator=initial_generator,
        max_workers=1,
    )

    print_channel_stats(msi, generated, "Initial no-diffusion stats after sigma optimization")

    plot_channel_comparison(
        msi=msi,
        generated=generated,
        channels_to_show=channels_to_show,
        output_path="output/target_v_initial_multi.png",
        title="Initial comparison after sigma optimization",
    )

    mus = estimate_channel_mus(
        msi=msi,
        mask=mask,
        scaling_factor=scaling_factor,
        num_channels=num_channels,
    )

    diffusion_total_times = [1.0] * num_channels

    diffusion_coefficients = [
        float(mu_c["mu_est"])
        for mu_c in mus
    ]

    for c, D in enumerate(diffusion_coefficients):
        T = diffusion_total_times[c]
        print(
            f"Channel {c}: "
            f"D={D:.6f}, "
            f"T={T:.6f}, "
            f"D*T={D * T:.6f}, "
            f"sigma_px={np.sqrt(2.0 * D * T):.6f}"
        )

    intensities = scale_intensities_to_target_totals(
        intensities=intensities,
        msi=msi,
    )

    mu_generator = make_channel_generator(
        mask=mask,
        sigmas=sigmas,
        intensities=intensities,
        diffusion_total_times=diffusion_total_times,
        diffusion_coefficients=diffusion_coefficients,
        diffusion_barrier_alpha=diffusion_barrier_alpha,
        scaling_factor=scaling_factor,
        diffusion_downsample=8,
        max_sigma_lowres=15.0,
    )

    generated = generate_all_channels(
        num_channels=num_channels,
        channel_generator=mu_generator,
        max_workers=1,
    )

    print_channel_stats(msi, generated, "Mu-based diffusion stats")

    plot_channel_comparison(
        msi=msi,
        generated=generated,
        channels_to_show=channels_to_show,
        output_path="output/target_v_mu_diffusion_multi.png",
        title="Mu-based diffusion comparison",
    )

    generated = apply_noise_model(
        generated,
        msi,
        rank_match=True,
    )

    plot_channel_comparison(
        msi=msi,
        generated=generated,
        channels_to_show=channels_to_show,
        output_path="output/target_v_final.png",
        title="Mu-based diffusion + noise comparison",
    )