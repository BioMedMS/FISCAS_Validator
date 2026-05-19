import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
from concurrent.futures import ThreadPoolExecutor

from canvas_generation import generate_msi_from_mask_slice, _choose_cell_center_from_mask_and_target
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
        top * scaling_factor:mask.shape[0] - bottom * scaling_factor,
        left * scaling_factor:mask.shape[1] - right * scaling_factor,
    ]

    new_msi = msi[
        top:msi.shape[0] - bottom,
        left:msi.shape[1] - right,
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

def prepare_cell_render_geometry(cell_mask, cell_id, scaling_factor, target=None):
    sf = int(scaling_factor)
    local_mask = cell_mask == cell_id
    coords = np.argwhere(local_mask)

    if target is None:
        h_ms = int(np.ceil(cell_mask.shape[0] / sf))
        w_ms = int(np.ceil(cell_mask.shape[1] / sf))
    else:
        h_ms, w_ms = target.shape[:2]

    if coords.shape[0] == 0:
        return cell_mask.shape, coords, None, h_ms, w_ms, sf

    center = _choose_cell_center_from_mask_and_target(
        local_mask=local_mask,
        target_patch=target,
        scaling_factor=scaling_factor,
    )

    if center is None:
        cy, cx = coords.mean(axis=0)
    else:
        cy, cx = center

    yy = coords[:, 0].astype(np.float32)
    xx = coords[:, 1].astype(np.float32)
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    return cell_mask.shape, coords, distance, h_ms, w_ms, sf

def render_prepared_cell_to_msi(geometry, sigma, intensity):
    cell_shape, coords, distance, h_ms, w_ms, sf = geometry

    if coords.shape[0] == 0 or intensity <= 0:
        return np.zeros((h_ms, w_ms), dtype=np.float32)

    sigma = max(float(sigma), 1e-6)
    values = np.exp(-((distance / sigma) ** 2)) + 0.1

    hr = np.zeros(cell_shape, dtype=np.float32)
    hr[coords[:, 0], coords[:, 1]] = values.astype(np.float32)

    total = float(hr.sum())

    if total > 0:
        hr *= float(intensity) / total

    pad_h = h_ms * sf
    pad_w = w_ms * sf

    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:hr.shape[0], :hr.shape[1]] = hr

    return skimage.measure.block_reduce(
        padded,
        block_size=(sf, sf),
        func=np.sum,
    ).astype(np.float32)[:h_ms, :w_ms]

def render_one_cell_to_msi(cell_mask, cell_id, sigma, intensity, scaling_factor, target=None):
    geometry = prepare_cell_render_geometry(
        cell_mask=cell_mask,
        cell_id=cell_id,
        scaling_factor=scaling_factor,
        target=target,
    )

    return render_prepared_cell_to_msi(
        geometry=geometry,
        sigma=sigma,
        intensity=intensity,
    )

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

    local_pos = {
        cell_id: idx
        for idx, cell_id in enumerate(valid_local_cell_ids)
    }

    target = cell_targets[c][i].astype(np.float32, copy=False)
    target_local_idx = local_pos[target_cell_id]

    local_sigmas = [
        float(sigmas[c][global_pos[cell_id]])
        for cell_id in valid_local_cell_ids
    ]

    local_intensities = [
        float(intensities[c][global_pos[cell_id]])
        for cell_id in valid_local_cell_ids
    ]

    geometries = [
        prepare_cell_render_geometry(
            cell_mask=cell_masks[i],
            cell_id=cell_id,
            scaling_factor=scaling_factor,
            target=target,
        )
        for cell_id in valid_local_cell_ids
    ]

    rendered_cells = [
        render_prepared_cell_to_msi(
            geometry=geometries[local_idx],
            sigma=local_sigmas[local_idx],
            intensity=local_intensities[local_idx],
        )
        for local_idx in range(len(valid_local_cell_ids))
    ]

    baseline = np.zeros_like(target, dtype=np.float32)

    for rendered in rendered_cells:
        baseline += rendered

    target_old = rendered_cells[target_local_idx]
    best_score = symmetric_l1_ratio(target, baseline)
    best_sigma = float(local_sigmas[target_local_idx])

    baseline_without_target = baseline - target_old

    step = max(1, int(round(scaling_factor / 2)))
    max_sigma = max(1, int(round(sigmas[c][i])))
    target_geometry = geometries[target_local_idx]
    target_intensity = local_intensities[target_local_idx]

    for sigma in range(1, max_sigma + 1, step):
        target_new = render_prepared_cell_to_msi(
            geometry=target_geometry,
            sigma=float(sigma),
            intensity=target_intensity,
        )

        generated = baseline_without_target + target_new
        score = symmetric_l1_ratio(target, generated)

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

def build_cell_fraction_maps(mask, cell_indexes, scaling_factor, target_shape):
    return {
        int(cell_id): block_fraction_from_hr_mask(
            mask_i=(mask == int(cell_id)),
            scaling_factor=scaling_factor,
            target_shape=target_shape,
        ).astype(np.float32)
        for cell_id in cell_indexes
    }

def adjust_intensity_parameters_after_diffusion(
    generated_mu,
    generated_optimized,
    intensities,
    cell_indexes,
    cell_fraction_maps,
    eps=1e-12,
):
    post = np.asarray(generated_mu, dtype=np.float32)
    pre = np.asarray(generated_optimized, dtype=np.float32)

    if post.shape != pre.shape:
        raise ValueError(
            f"generated_mu and generated_optimized must have the same shape. "
            f"Got {post.shape} and {pre.shape}."
        )

    if post.ndim != 3:
        raise ValueError(
            f"generated_mu must have shape (H, W, C). Got {post.shape}."
        )

    adjusted_intensities = [
        [float(x) for x in channel_intensities]
        for channel_intensities in intensities
    ]

    num_channels = post.shape[-1]

    for i, cell_id in enumerate(cell_indexes):
        cell_fraction = cell_fraction_maps.get(int(cell_id))

        if cell_fraction is None:
            continue

        cell_pixels = cell_fraction > 0

        if not np.any(cell_pixels):
            continue

        weights = cell_fraction[cell_pixels]

        for c in range(num_channels):
            pre_sum = float(np.sum(pre[:, :, c][cell_pixels] * weights))
            post_sum = float(np.sum(post[:, :, c][cell_pixels] * weights))

            if pre_sum <= eps:
                adjusted_intensities[c][i] = 0.0
                continue

            if post_sum <= eps:
                continue

            adjusted_intensities[c][i] *= pre_sum / post_sum

    return adjusted_intensities

def _value_for_channel(values, c):
    if np.isscalar(values):
        return float(values)

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
    target_msi=None,
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
            target_canvas=(target_msi[:, :, c] if target_msi is not None else None),
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

def relabel_mask_for_display(mask):
    mask = np.asarray(mask)
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]

    relabeled = np.zeros_like(mask, dtype=np.int32)

    for new_id, old_id in enumerate(cell_ids, start=1):
        relabeled[mask == old_id] = new_id

    return relabeled

def compute_channel_l1_errors(msi, generated, channels_to_show, channel_labels=None):
    l1_errors = {}

    for ch in channels_to_show:
        channel_number = _channel_suffix(ch, channel_labels)
        target_ch = msi[:, :, ch]
        generated_ch = generated[:, :, ch]
        l1_errors[channel_number] = symmetric_l1_ratio(target_ch, generated_ch)

    return l1_errors

def plot_channel_pipeline_summary(
    mask,
    step_images,
    channel_idx,
    output_path,
    title="MSI generation pipeline",
    channel_number=None,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if channel_number is None:
        channel_number = channel_idx

    mask_display = relabel_mask_for_display(mask)
    mask_display = np.ma.masked_where(mask_display == 0, mask_display)

    mask_cmap = plt.get_cmap("nipy_spectral").copy()
    mask_cmap.set_bad(color="black")

    target = np.asarray(step_images["Target"], dtype=np.float32)[:, :, channel_idx]
    initial = np.asarray(step_images["Initial"], dtype=np.float32)[:, :, channel_idx]
    sigma_opt = np.asarray(step_images["Sigma-opt"], dtype=np.float32)[:, :, channel_idx]
    mu_diff = np.asarray(step_images["Mu-diffusion"], dtype=np.float32)[:, :, channel_idx]
    final = np.asarray(step_images["Final"], dtype=np.float32)[:, :, channel_idx]
    sim_noise = np.asarray(step_images["Sim-noise"], dtype=np.float32)[:, :, channel_idx]

    panel_data = [
        ("Mask", mask_display, None, True),
        ("Initial", initial, symmetric_l1_ratio(target, initial), False),
        ("Sigma-opt", sigma_opt, symmetric_l1_ratio(target, sigma_opt), False),
        ("Mu-diffusion", mu_diff, symmetric_l1_ratio(target, mu_diff), False),
        ("Final", final, symmetric_l1_ratio(target, final), False),
        ("Sim-noise", sim_noise, symmetric_l1_ratio(target, sim_noise), False),
        ("Target", target, None, False),
    ]

    intensity_images = [initial, sigma_opt, mu_diff, final, sim_noise, target]
    vmin = min(float(img.min()) for img in intensity_images)
    vmax = max(float(img.max()) for img in intensity_images)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    left_margin = 0.20
    right_margin = 0.06
    bottom_margin = 0.18
    top_margin = 0.10

    title_band = 0.42
    subplot_title_band = 0.34

    panel_gap = 0.05
    cbar_gap = 0.06
    cbar_width = 0.12

    cbar_tick_label_space = 0.50

    panel_height = 3.00

    panel_widths = []
    for _, img, _, _ in panel_data:
        h, w = img.shape[:2]
        aspect = float(w) / float(h)
        panel_widths.append(panel_height * aspect)

    fig_width = (
        left_margin
        + sum(panel_widths)
        + panel_gap * (len(panel_widths) - 1)
        + cbar_gap
        + cbar_width
        + cbar_tick_label_space
        + right_margin
    )

    fig_height = (
        bottom_margin
        + panel_height
        + subplot_title_band
        + title_band
        + top_margin
    )

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=200, constrained_layout=False)

    total_w = fig_width
    total_h = fig_height

    panel_bottom = bottom_margin / total_h
    panel_h = panel_height / total_h

    x = left_margin / total_w
    axes = []

    for panel_w in panel_widths:
        ax = fig.add_axes([
            x,
            panel_bottom,
            panel_w / total_w,
            panel_h,
        ])
        axes.append(ax)
        x += (panel_w + panel_gap) / total_w

    cax = fig.add_axes([
        x + cbar_gap / total_w,
        panel_bottom,
        cbar_width / total_w,
        panel_h,
    ])

    im = None

    for ax, (name, img, l1, is_mask) in zip(axes, panel_data):
        if is_mask:
            ax.imshow(
                img,
                cmap=mask_cmap,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_facecolor("black")
            ax.set_title("Mask", fontsize=10, pad=6)
        else:
            im = ax.imshow(
                img,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )

            if name == "Target":
                ax.set_title("Target", fontsize=10, pad=6)
            else:
                ax.set_title(f"{name}\nL1: {l1:.4f}", fontsize=9, pad=6)

        ax.axis("off")

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{title} | Channel {channel_number}",
        fontsize=15,
        y=1.0 - top_margin / fig_height,
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    
def plot_final_channel_summaries(
    mask,
    step_images,
    channels_to_show,
    output_dir,
    channel_labels=None,
    base_filename="pipeline_summary",
    title="MSI generation pipeline",
):
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []

    for ch in channels_to_show:
        channel_number = _channel_suffix(ch, channel_labels)
        output_path = os.path.join(output_dir, f"{base_filename}_channel{channel_number}.png")

        plot_channel_pipeline_summary(
            mask=mask,
            step_images=step_images,
            channel_idx=ch,
            output_path=output_path,
            title=title,
            channel_number=channel_number,
        )

        output_paths.append(output_path)

    return output_paths

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

    print("Building cell fraction maps")
    cell_fraction_maps = build_cell_fraction_maps(
        mask=mask,
        cell_indexes=cell_indexes,
        scaling_factor=scaling_factor,
        target_shape=msi.shape[:2],
    )

    generated_preopt = None

    if plotting_enabled:
        print("Generating initial no-diffusion simulation before sigma optimization")
        preopt_generator = make_channel_generator(
            mask=mask,
            sigmas=sigmas,
            intensities=intensities,
            diffusion_total_times=0.0,
            diffusion_coefficients=0.0,
            diffusion_barrier_alphas=diffusion_barrier_alpha,
            scaling_factor=scaling_factor,
            diffusion_downsample=diffusion_downsample,
            max_sigma_lowres=max_sigma_lowres,
            target_msi=msi,
        )

        generated_preopt = generate_all_channels(
            num_channels=num_channels,
            channel_generator=preopt_generator,
            max_workers=generation_max_workers,
        )

        print("Computing initial no-optimization errors")
        l1_errors_by_step["initial_no_optimization"] = compute_channel_l1_errors(
            msi=msi,
            generated=generated_preopt,
            channels_to_show=channels_to_show,
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
        diffusion_total_times=0.0,
        diffusion_coefficients=0.0,
        diffusion_barrier_alphas=diffusion_barrier_alpha,
        scaling_factor=scaling_factor,
        diffusion_downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
        target_msi=msi,
    )

    generated_optimized = generate_all_channels(
        num_channels=num_channels,
        channel_generator=optimized_generator,
        max_workers=generation_max_workers,
    )

    if plotting_enabled:
        print("Computing optimized-sigma errors")
        l1_errors_by_step["initial_no_diffusion"] = compute_channel_l1_errors(
            msi=msi,
            generated=generated_optimized,
            channels_to_show=channels_to_show,
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
    diffusion_total_times_1d = np.full(num_channels, t, dtype=np.float64)

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
        diffusion_total_times=diffusion_total_times_1d,
        diffusion_coefficients=diffusion_coefficients_1d.tolist(),
        diffusion_barrier_alphas=diffusion_barrier_alphas_1d.tolist(),
        scaling_factor=scaling_factor,
        diffusion_downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
        target_msi=msi,
    )

    generated_mu_uncorrected = generate_all_channels(
        num_channels=num_channels,
        channel_generator=mu_generator,
        max_workers=generation_max_workers,
    )

    print("Adjusting intensity parameters after diffusion")
    intensities = adjust_intensity_parameters_after_diffusion(
        generated_mu=generated_mu_uncorrected,
        generated_optimized=generated_optimized,
        intensities=intensities,
        cell_indexes=cell_indexes,
        cell_fraction_maps=cell_fraction_maps,
    )

    print("Generating barrier-aware mu-based diffusion simulation with adjusted intensities")
    mu_generator = make_channel_generator(
        mask=mask,
        sigmas=sigmas,
        intensities=intensities,
        diffusion_total_times=diffusion_total_times_1d,
        diffusion_coefficients=diffusion_coefficients_1d.tolist(),
        diffusion_barrier_alphas=diffusion_barrier_alphas_1d.tolist(),
        scaling_factor=scaling_factor,
        diffusion_downsample=diffusion_downsample,
        max_sigma_lowres=max_sigma_lowres,
        target_msi=msi,
    )

    generated_mu = generate_all_channels(
        num_channels=num_channels,
        channel_generator=mu_generator,
        max_workers=generation_max_workers,
    )

    if apply_final_noise:
        print("Applying final noise model")
        generated_final = apply_noise_model(
            generated_mu,
            msi,
            seed=noise_seed,
            rank_match=noise_rank_match,
        )
    else:
        generated_final = generated_mu

    generated_sim_noise = generated_mu

    if plotting_enabled:
        print("Computing barrier-aware mu-based diffusion errors")
        l1_errors_by_step["mu_diffusion"] = compute_channel_l1_errors(
            msi=msi,
            generated=generated_mu,
            channels_to_show=channels_to_show,
            channel_labels=channel_labels,
        )

        if apply_final_noise:
            print("Applying simulation-style random Gaussian noise")
            generated_sim_noise = apply_noise_model(
                generated_mu,
                msi,
                seed=noise_seed,
                rank_match=False,
            )

            print("Computing final noise errors")
            l1_errors_by_step["mu_diffusion_noise"] = compute_channel_l1_errors(
                msi=msi,
                generated=generated_final,
                channels_to_show=channels_to_show,
                channel_labels=channel_labels,
            )

            print("Computing simulation-style random noise errors")
            l1_errors_by_step["mu_diffusion_sim_noise"] = compute_channel_l1_errors(
                msi=msi,
                generated=generated_sim_noise,
                channels_to_show=channels_to_show,
                channel_labels=channel_labels,
            )

    if plotting_enabled:
        print("Rendering final per-channel pipeline summaries")

        step_images = {
            "Target": msi,
            "Initial": generated_preopt,
            "Sigma-opt": generated_optimized,
            "Mu-diffusion": generated_mu,
            "Final": generated_final,
            "Sim-noise": generated_sim_noise,
        }

        plot_final_channel_summaries(
            mask=mask,
            step_images=step_images,
            channels_to_show=channels_to_show,
            output_dir=output_dir,
            channel_labels=channel_labels,
            base_filename="pipeline_summary",
            title="MSI generation pipeline",
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
            diffusion_dt=diffusion_total_times_1d,
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
        diffusion_total_times_1d,
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

    mask = np.load("input/seg.npy")
    msi = np.load("input/msi_data.npy")

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