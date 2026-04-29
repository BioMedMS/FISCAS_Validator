import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
from concurrent.futures import ThreadPoolExecutor

from canvas_generation import generate_msi_from_mask_slice
from diffusion_estimator import estimate_mu_from_distribution


def symmetric_l1_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = np.sum(np.abs(a)) + np.sum(np.abs(b))

    if denominator == 0:
        return 0.0

    return float(np.sum(np.abs(a - b)) / denominator)


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

    cell_fractions = []

    for i, cell_id in enumerate(cell_indexes):
        mask_i = cell_masks[i] == cell_id
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

    return optimized_sigmas


def scale_intensities_to_target_totals(intensities, msi):
    num_channels = msi.shape[-1]

    for c in range(num_channels):
        target_total = float(msi[:, :, c].sum())
        current_total = float(np.sum(intensities[c]))

        if current_total > 0:
            intensity_scale = target_total / current_total
            intensities[c] = [float(x) * intensity_scale for x in intensities[c]]

    return intensities


def estimate_channel_mus(
    msi,
    mask,
    scaling_factor,
    num_channels,
    diffusion_estimation_power=2.0,
    diffusion_inner_boundary_px=2,
    diffusion_outer_boundary_px=2,
    diffusion_far_background_inner_px=8,
    diffusion_far_background_outer_px=20,
    diffusion_intracellular_fraction=0.25,
    diffusion_min_cell_pixels=5,
    diffusion_max_mu=None,
):
    mus = []

    for c in range(num_channels):
        msi_hr_c = np.repeat(
            np.repeat(msi[:, :, c], scaling_factor, axis=0),
            scaling_factor,
            axis=1,
        )

        mu_c = estimate_mu_from_distribution(
            distribution=msi_hr_c,
            patch_mask=mask,
            estimation_power=diffusion_estimation_power,
            inner_boundary_px=diffusion_inner_boundary_px,
            outer_boundary_px=diffusion_outer_boundary_px,
            far_background_inner_px=diffusion_far_background_inner_px,
            far_background_outer_px=diffusion_far_background_outer_px,
            intracellular_diffusion_fraction=diffusion_intracellular_fraction,
            min_cell_pixels=diffusion_min_cell_pixels,
            max_mu=diffusion_max_mu,
        )

        mus.append(mu_c)

    return mus


def _value_for_channel(values, c):
    if np.isscalar(values):
        return float(values)

    values = list(values)
    return float(values[c])


def make_channel_generator(
    mask,
    sigmas,
    intensities,
    diffusion_total_times,
    diffusion_coefficients,
    diffusion_barrier_alphas,
    scaling_factor,
    diffusion_downsample=8,
    max_sigma_lowres=15.0,
):
    def _generate_channel(c):
        return generate_msi_from_mask_slice(
            mask_slice=mask,
            sigmas=sigmas[c],
            intensities=intensities[c],
            diffusion_total_time=_value_for_channel(diffusion_total_times, c),
            diffusion_coefficient=_value_for_channel(diffusion_coefficients, c),
            diffusion_barrier_alpha=_value_for_channel(diffusion_barrier_alphas, c),
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


def _channel_suffix(channel_index, channel_labels=None):
    if channel_labels is None:
        return int(channel_index)

    label = channel_labels[channel_index]

    if isinstance(label, np.generic):
        label = label.item()

    if isinstance(label, float):
        return f"{label:g}"

    return str(label)


def _channel_output_path(output_dir, base_filename, channel_number):
    stem, ext = os.path.splitext(os.path.basename(base_filename))

    return os.path.join(output_dir, f"{stem}_channel{channel_number}{ext}")


def plot_mask(mask, output_path, title="Segmentation mask"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    im = ax.imshow(mask, cmap="nipy_spectral", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_single_channel_comparison(msi, generated, channel_idx, output_path, title, channel_number=None):
    channel_number = channel_idx if channel_number is None else channel_number

    target_ch = msi[:, :, channel_idx]
    generated_ch = generated[:, :, channel_idx]
    layer_l1 = symmetric_l1_ratio(target_ch, generated_ch)

    vmin = min(float(target_ch.min()), float(generated_ch.min()))
    vmax = max(float(target_ch.max()), float(generated_ch.max()))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.4),
        gridspec_kw={"wspace": 0.0},
        constrained_layout=False,
    )

    axes[0].imshow(
        target_ch,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    axes[0].set_title("Target", fontsize=10, pad=2)
    axes[0].axis("off")

    im1 = axes[1].imshow(
        generated_ch,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1].set_title("Simulation", fontsize=10, pad=2)
    axes[1].axis("off")

    fig.suptitle(
        f"{title} | Channel {channel_number} | L1: {layer_l1:.6f}",
        fontsize=11,
        y=0.98,
    )

    fig.subplots_adjust(left=0.005, right=0.90, bottom=0.005, top=0.88, wspace=0.0)
    cbar = fig.colorbar(
        im1,
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.92,
        pad=0.01,
    )
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return layer_l1


def plot_channel_comparison(msi, generated, channels_to_show, output_path, title, channel_labels=None):
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    l1_errors = {}

    for ch in channels_to_show:
        channel_number = _channel_suffix(ch, channel_labels)
        channel_output_path = _channel_output_path(output_dir, output_path, channel_number)

        l1_errors[channel_number] = plot_single_channel_comparison(
            msi=msi,
            generated=generated,
            channel_idx=ch,
            output_path=channel_output_path,
            title=title,
            channel_number=channel_number,
        )

    return l1_errors


def save_l1_errors(l1_errors_by_step, output_dir, filename="l1_errors_by_channel.csv"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    all_channels = sorted({
        channel
        for step_errors in l1_errors_by_step.values()
        for channel in step_errors.keys()
    })

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("step,channel,l1_error\n")

        for step_name, step_errors in l1_errors_by_step.items():
            for channel in all_channels:
                if channel in step_errors:
                    f.write(f"{step_name},{channel},{step_errors[channel]:.12g}\n")

    return output_path


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


def _csv_value(value):
    if value is None:
        return ""

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return f"{value:.12g}"

    return str(value)


def save_parameter_summary(
    output_dir,
    filename,
    channel_labels,
    channels,
    diffusion_coefficients,
    diffusion_dt,
    diffusion_barrier_alphas,
    t,
    mus,
    noise_seed,
    apply_final_noise,
    noise_rank_match,
    intensities,
    sigmas,
):
    print("Saving parameter summary CSV")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    diagnostic_columns = [
        "sigma_px_est",
        "sigma_ms_est",
        "barrier_alpha_est",
        "estimation_power",
        "cell_fraction_threshold",
        "background_fraction_threshold",
        "near_background_radius_ms",
        "far_background_radius_ms",
        "intracellular_diffusion_fraction",
        "median_cell_radius_ms",
        "cell_mean",
        "cell_median",
        "cell_std",
        "cell_cv",
        "cell_homogeneity",
        "near_background_median",
        "far_background_median",
        "edge_drop",
        "background_drop",
        "global_background",
        "n_cell_pixels",
        "n_background_pixels",
        "n_mixed_pixels",
        "n_near_background_pixels",
        "n_far_background_pixels",
        "method",
        "warning",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "mass,"
            "channel,"
            "diffusion_coefficient,"
            "diffusion_dt,"
            "diffusion_barrier_alpha,"
            "t,"
            "noise_seed,"
            "apply_final_noise,"
            "noise_rank_match,"
            + ",".join(diagnostic_columns)
            + ",intensities,sigmas\n"
        )

        for local_idx, channel_label in enumerate(channel_labels):
            intensity_values = ";".join(
                f"{x:.12g}"
                for x in intensities[local_idx]
            )

            sigma_values = ";".join(
                f"{x:.12g}"
                for x in sigmas[local_idx]
            )

            noise_seed_value = "" if noise_seed is None else str(noise_seed)

            diagnostic_values = [
                _csv_value(mus[local_idx].get(col, ""))
                for col in diagnostic_columns
            ]

            f.write(
                f"{_csv_value(channel_label)},"
                f"{channels[local_idx]},"
                f"{diffusion_coefficients[local_idx]:.12g},"
                f"{diffusion_dt[local_idx]:.12g},"
                f"{diffusion_barrier_alphas[local_idx]:.12g},"
                f"{t:.12g},"
                f"{noise_seed_value},"
                f"{bool(apply_final_noise)},"
                f"{bool(noise_rank_match)},"
                + ",".join(diagnostic_values)
                + f",\"{intensity_values}\","
                f"\"{sigma_values}\"\n"
            )

    print(f"Saved parameter summary to {output_path}")

    return output_path

def estimate_median_cell_radius_ms(mask, scaling_factor):
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]

    radii_ms = []

    for cell_id in cell_ids:
        area_hr = int(np.count_nonzero(mask == cell_id))

        if area_hr <= 0:
            continue

        radius_hr = float(np.sqrt(area_hr / np.pi))
        radius_ms = radius_hr / float(scaling_factor)
        radii_ms.append(radius_ms)

    if len(radii_ms) == 0:
        return 0.0

    return float(np.median(np.asarray(radii_ms, dtype=np.float64)))


def build_msi_mask_fraction(mask, scaling_factor, target_shape):
    return block_fraction_from_hr_mask(
        mask_i=(mask > 0),
        scaling_factor=scaling_factor,
        target_shape=target_shape,
    )


def estimate_channel_mus(
    msi,
    mask,
    scaling_factor,
    num_channels,
    diffusion_estimation_power=2.0,
    diffusion_cell_fraction_threshold=0.5,
    diffusion_background_fraction_threshold=0.05,
    diffusion_near_background_radius_ms=1.5,
    diffusion_far_background_radius_ms=3.0,
    diffusion_intracellular_fraction=0.25,
    diffusion_max_mu=None,
):
    mask_fraction_ms = build_msi_mask_fraction(
        mask=mask,
        scaling_factor=scaling_factor,
        target_shape=msi.shape[:2],
    )

    median_cell_radius_ms = estimate_median_cell_radius_ms(
        mask=mask,
        scaling_factor=scaling_factor,
    )

    mus = []

    for c in range(num_channels):
        mu_c = estimate_mu_from_distribution(
            distribution=msi[:, :, c],
            mask_fraction=mask_fraction_ms,
            scaling_factor=scaling_factor,
            median_cell_radius_ms=median_cell_radius_ms,
            estimation_power=diffusion_estimation_power,
            cell_fraction_threshold=diffusion_cell_fraction_threshold,
            background_fraction_threshold=diffusion_background_fraction_threshold,
            near_background_radius_ms=diffusion_near_background_radius_ms,
            far_background_radius_ms=diffusion_far_background_radius_ms,
            intracellular_diffusion_fraction=diffusion_intracellular_fraction,
            max_mu=diffusion_max_mu,
        )

        mus.append(mu_c)

    return mus

def estimate_parameters(
    mask,
    msi,
    channels=None,
    masses=None,
    channels_to_show=None,
    crop_top=0,
    crop_bottom=0,
    crop_left=0,
    crop_right=0,
    flatten_top_percent=True,
    plotting_enabled=True,
    output_dir="output",
    sigma_max_workers=4,
    generation_max_workers=1,
    padding=2,
    diffusion_barrier_alpha=0.0,
    diffusion_total_time=1.0,
    diffusion_downsample=8,
    max_sigma_lowres=15.0,
    diffusion_estimation_power=2.0,
    diffusion_cell_fraction_threshold=0.5,
    diffusion_background_fraction_threshold=0.05,
    diffusion_near_background_radius_ms=1.5,
    diffusion_far_background_radius_ms=3.0,
    diffusion_intracellular_fraction=0.25,
    diffusion_max_mu=None,
    noise_seed=None,
    apply_final_noise=True,
    noise_rank_match=True,
    save_parameter_summary_csv=True):
    """
    Estimate MSI generation parameters.

    Returns:
        intensities_2d
        sigmas_2d
        diffusion_coefficients_1d
        diffusion_dt_1d
        diffusion_barrier_alphas_1d
        t
    """
    print("Preparing input arrays")

    mask = np.asarray(mask, dtype=np.int32)
    msi = np.asarray(msi, dtype=np.float32)

    if msi.ndim != 3:
        raise ValueError(f"msi must have shape (H, W, C). Got {msi.shape}.")

    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W). Got {mask.shape}.")

    crop_top = int(crop_top)
    crop_bottom = int(crop_bottom)
    crop_left = int(crop_left)
    crop_right = int(crop_right)

    if min(crop_top, crop_bottom, crop_left, crop_right) < 0:
        raise ValueError("Crop values must be non-negative integers.")

    if any([crop_top, crop_bottom, crop_left, crop_right]):
        print("Cropping mask and MSI")
        mask, msi = crop_canvas_and_msi(
            mask=mask,
            msi=msi,
            top=crop_top,
            bottom=crop_bottom,
            left=crop_left,
            right=crop_right,
        )

    if channels is None:
        channels = list(range(msi.shape[-1]))
    else:
        channels = list(channels)

    if len(channels) == 0:
        raise ValueError("channels must contain at least one channel index.")

    print("Selecting MSI channels")
    msi = msi[:, :, channels]

    if masses is None:
        channel_labels = list(channels)
    else:
        masses = list(np.asarray(masses).ravel())

        if len(masses) != len(channels):
            raise ValueError(
                "masses must contain one entry per selected channel. "
                f"Got {len(masses)} masses for {len(channels)} selected channels."
            )

        channel_labels = masses

    if flatten_top_percent:
        print("Flattening top percent of selected MSI channels")
        p = np.percentile(msi, 99, axis=(0, 1), keepdims=True)
        msi = np.clip(msi, 0, p)

    num_channels = int(msi.shape[-1])

    if channels_to_show is None:
        channels_to_show = range(num_channels)

    channels_to_show = list(channels_to_show)

    for ch in channels_to_show:
        if ch < 0 or ch >= num_channels:
            raise ValueError(
                "channels_to_show must contain local channel indexes after slicing. "
                f"Got {ch}, but sliced MSI has {num_channels} channels."
            )

    if plotting_enabled:
        print("Plotting segmentation mask")
        os.makedirs(output_dir, exist_ok=True)

        plot_mask(
            mask=mask,
            output_path=os.path.join(output_dir, "mask.png"),
            title="Segmentation mask",
        )

    l1_errors_by_step = {}

    print("Computing scaling factor")
    scaling_factor = int(mask.shape[0] // msi.shape[0])

    print("Extracting individual cell canvases")
    cell_indexes, cell_masks, cell_targets = get_individual_cell_canvases(
        mask=mask,
        msi_canvas=msi,
        scaling_factor=scaling_factor,
        padding=padding,
    )

    print("Building initial sigmas and intensities")
    sigmas, intensities = build_initial_sigmas_and_intensities(
        cell_indexes=cell_indexes,
        cell_masks=cell_masks,
        cell_targets=cell_targets,
        scaling_factor=scaling_factor,
        num_channels=num_channels,
    )

    zero_diffusion_total_times = [0.0] * num_channels
    zero_diffusion_coefficients = [0.0] * num_channels
    zero_diffusion_barrier_alphas = [diffusion_barrier_alpha] * num_channels

    print("Generating initial no-diffusion simulation before sigma optimization")
    preopt_generator = make_channel_generator(
        mask=mask,
        sigmas=sigmas,
        intensities=intensities,
        diffusion_total_times=zero_diffusion_total_times,
        diffusion_coefficients=zero_diffusion_coefficients,
        diffusion_barrier_alphas=zero_diffusion_barrier_alphas,
        scaling_factor=scaling_factor,
        diffusion_downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
    )

    generated_preopt = generate_all_channels(
        num_channels=num_channels,
        channel_generator=preopt_generator,
        max_workers=generation_max_workers,
    )

    if plotting_enabled:
        print("Plotting initial no-optimization comparison")
        l1_errors_by_step["initial_no_optimization"] = plot_channel_comparison(
            msi=msi,
            generated=generated_preopt,
            channels_to_show=channels_to_show,
            output_path=os.path.join(output_dir, "target_v_initial.png"),
            title="Initial comparison before sigma optimization",
            channel_labels=channel_labels,
        )

    print("Optimizing sigmas")
    sigmas = parallel_sigma_optimisation(
        cell_indexes=cell_indexes,
        cell_masks=cell_masks,
        cell_targets=cell_targets,
        sigmas=sigmas,
        intensities=intensities,
        scaling_factor=scaling_factor,
        max_workers=sigma_max_workers,
    )

    print("Generating no-diffusion simulation after sigma optimization")
    optimized_generator = make_channel_generator(
        mask=mask,
        sigmas=sigmas,
        intensities=intensities,
        diffusion_total_times=zero_diffusion_total_times,
        diffusion_coefficients=zero_diffusion_coefficients,
        diffusion_barrier_alphas=zero_diffusion_barrier_alphas,
        scaling_factor=scaling_factor,
        diffusion_downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
    )

    generated_optimized = generate_all_channels(
        num_channels=num_channels,
        channel_generator=optimized_generator,
        max_workers=generation_max_workers,
    )

    if plotting_enabled:
        print("Plotting optimized-sigma comparison")
        l1_errors_by_step["initial_no_diffusion"] = plot_channel_comparison(
            msi=msi,
            generated=generated_optimized,
            channels_to_show=channels_to_show,
            output_path=os.path.join(output_dir, "target_v_optimal_sigma.png"),
            title="Initial comparison after sigma optimization",
            channel_labels=channel_labels,
        )

    print("Estimating diffusion coefficients and barrier alphas")
    mus = estimate_channel_mus(
        msi=msi,
        mask=mask,
        scaling_factor=scaling_factor,
        num_channels=num_channels,
        diffusion_estimation_power=diffusion_estimation_power,
        diffusion_cell_fraction_threshold=diffusion_cell_fraction_threshold,
        diffusion_background_fraction_threshold=diffusion_background_fraction_threshold,
        diffusion_near_background_radius_ms=diffusion_near_background_radius_ms,
        diffusion_far_background_radius_ms=diffusion_far_background_radius_ms,
        diffusion_intracellular_fraction=diffusion_intracellular_fraction,
        diffusion_max_mu=diffusion_max_mu,
    )

    print("Preparing diffusion parameters")
    t = float(diffusion_total_time)
    diffusion_dt_1d = np.full(num_channels, t, dtype=np.float64)

    diffusion_coefficients_1d = np.asarray(
        [float(mu_c.get("mu_est", 0.0)) for mu_c in mus],
        dtype=np.float64,
    )

    diffusion_barrier_alphas_1d = np.asarray(
        [float(mu_c.get("barrier_alpha_est", diffusion_barrier_alpha)) for mu_c in mus],
        dtype=np.float64,
    )

    diffusion_barrier_alphas_1d = np.clip(diffusion_barrier_alphas_1d, 0.0, 1.0)

    print("Generating barrier-aware mu-based diffusion simulation")
    mu_generator = make_channel_generator(
        mask=mask,
        sigmas=sigmas,
        intensities=intensities,
        diffusion_total_times=diffusion_dt_1d.tolist(),
        diffusion_coefficients=diffusion_coefficients_1d.tolist(),
        diffusion_barrier_alphas=diffusion_barrier_alphas_1d.tolist(),
        scaling_factor=scaling_factor,
        diffusion_downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
    )

    generated_mu = generate_all_channels(
        num_channels=num_channels,
        channel_generator=mu_generator,
        max_workers=generation_max_workers,
    )

    if plotting_enabled:
        print("Plotting barrier-aware mu-based diffusion comparison")
        l1_errors_by_step["mu_diffusion"] = plot_channel_comparison(
            msi=msi,
            generated=generated_mu,
            channels_to_show=channels_to_show,
            output_path=os.path.join(output_dir, "target_v_diffusion.png"),
            title="Barrier-aware mu-based diffusion comparison",
            channel_labels=channel_labels,
        )

    if apply_final_noise:
        print("Applying final noise model")
        generated_final = apply_noise_model(
            generated_mu,
            msi,
            seed=noise_seed,
            rank_match=noise_rank_match,
        )

        if plotting_enabled:
            print("Plotting final noise comparison")
            l1_errors_by_step["mu_diffusion_noise"] = plot_channel_comparison(
                msi=msi,
                generated=generated_final,
                channels_to_show=channels_to_show,
                output_path=os.path.join(output_dir, "target_v_final.png"),
                title="Barrier-aware diffusion + noise comparison",
                channel_labels=channel_labels,
            )

    if plotting_enabled and l1_errors_by_step:
        print("Saving L1 errors")
        save_l1_errors(
            l1_errors_by_step=l1_errors_by_step,
            output_dir=output_dir,
            filename="l1_errors_by_channel.csv",
        )

    print("Preparing return values")
    intensities_2d = np.asarray(intensities, dtype=np.float64)
    sigmas_2d = np.asarray(sigmas, dtype=np.float64)

    if save_parameter_summary_csv:
        save_parameter_summary(
            output_dir=output_dir,
            filename="parameters_by_mass.csv",
            channel_labels=channel_labels,
            channels=channels,
            diffusion_coefficients=diffusion_coefficients_1d,
            diffusion_dt=diffusion_dt_1d,
            diffusion_barrier_alphas=diffusion_barrier_alphas_1d,
            t=t,
            mus=mus,
            noise_seed=noise_seed,
            apply_final_noise=apply_final_noise,
            noise_rank_match=noise_rank_match,
            intensities=intensities_2d,
            sigmas=sigmas_2d,
        )

    print("Parameter estimation finished")

    return (
        intensities_2d,
        sigmas_2d,
        diffusion_coefficients_1d,
        diffusion_dt_1d,
        diffusion_barrier_alphas_1d,
        t,
    )


if __name__ == "__main__":
    target_masses = np.asarray(
        [
            369.35102274,
            744.55183039,
            760.58278915,
            885.78784207,
        ],
        dtype=np.float64,
    )

    mask = np.load("input/seg.npy").astype(np.int32)
    msi = np.load("input/msi_data.npy").astype(np.float32)

    all_masses = np.load("input/mass_list.npy").astype(np.float64).ravel()

    channels = [
        int(np.argmin(np.abs(all_masses - target_mass)))
        for target_mass in target_masses
    ]

    masses = all_masses[channels]

    print("Selected channels:")
    for requested_mass, channel, matched_mass in zip(target_masses, channels, masses):
        print(
            f"requested mass={requested_mass:.8f}, "
            f"channel={channel}, "
            f"matched mass={matched_mass:.8f}, "
            f"delta={abs(matched_mass - requested_mass):.8g}"
        )

    (
        intensities,
        sigmas,
        diffusion_coefficients,
        diffusion_dt,
        diffusion_barrier_alphas,
        t,
    ) = estimate_parameters(
        mask=mask,
        msi=msi,
        channels=channels,
        masses=masses,
        crop_top=110,
        crop_bottom=110,
        crop_left=100,
        crop_right=100,
        flatten_top_percent=True,
        plotting_enabled=True,
        output_dir="output",
        sigma_max_workers=4,
        generation_max_workers=1,
        diffusion_total_time=1.0,
        diffusion_downsample=8,
        max_sigma_lowres=15.0,
        diffusion_estimation_power=2.0,
        diffusion_cell_fraction_threshold=0.5,
        diffusion_background_fraction_threshold=0.05,
        diffusion_near_background_radius_ms=1.5,
        diffusion_far_background_radius_ms=3.0,
        diffusion_intracellular_fraction=0.25,
        diffusion_max_mu=None,
        noise_seed=None,
        apply_final_noise=True,
        noise_rank_match=True,
        save_parameter_summary_csv=True,
    )