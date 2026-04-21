import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import matplotlib.pyplot as plt

from canvas_generation import generate_msi_from_mask_slice
from error_estimation import sum_abs_diff, mean_absolute_percentage_error

def crop_canvas_and_msi(mask, msi, top, bottom, left, right):
    scaling_factor = int(mask.shape[0] / msi.shape[0])
    print(scaling_factor)
    new_mask = mask[left*scaling_factor:mask.shape[0]-right*scaling_factor, top*scaling_factor:mask.shape[1]-bottom*scaling_factor]
    new_msi = msi[left:msi.shape[0]-right, top:msi.shape[1]-bottom]
    print(new_msi)
    return new_mask, new_msi

def get_cell_neighbourhood(canvas, cell_index, padding, scaling_factor):
    rows_found = False
    first_collumn = 0
    last_collumn = 0
    collumns_found = False
    first_row = 0
    last_row = 0

    for i in range(canvas.shape[0]):
        if cell_index in canvas[i, :]:
            if not rows_found:
                rows_found = True
                first_row = i
        else:
            if rows_found:
                last_row = i - 1
                break

    if rows_found and last_row == 0:
        last_row = canvas.shape[0] - 1

    for i in range(canvas.shape[1]):
        if cell_index in canvas[:, i]:
            if not collumns_found:
                collumns_found = True
                first_collumn = i
        else:
            if collumns_found:
                last_collumn = i - 1
                break

    if collumns_found and last_collumn == 0:
        last_collumn = canvas.shape[1] - 1

    left = int(max(first_collumn - first_collumn % scaling_factor - padding * scaling_factor, 0))
    top = int(max(first_row - first_row % scaling_factor - padding * scaling_factor, 0))

    right = int(min(
        last_collumn - last_collumn % scaling_factor + padding * scaling_factor + scaling_factor,
        canvas.shape[1]
    ))
    bottom = int(min(
        last_row - last_row % scaling_factor + padding * scaling_factor + scaling_factor,
        canvas.shape[0]
    ))

    return canvas[top:bottom, left:right], (left, right, top, bottom)

def get_individual_cell_canvases(mask, msi_canvas, scaling_factor, padding):
    scaling_factor = int(scaling_factor)

    cell_indexes = np.unique(mask)
    cell_indexes = cell_indexes[cell_indexes != 0]
    masks = []
    targets = []

    for index in cell_indexes:
        crop_mask, (left_hr, right_hr, top_hr, bottom_hr) = get_cell_neighbourhood(
            mask, index, padding, scaling_factor
        )
        masks.append(crop_mask)

        left = left_hr // scaling_factor
        right = right_hr // scaling_factor
        top = top_hr // scaling_factor
        bottom = bottom_hr // scaling_factor

        targets.append(msi_canvas[top:bottom, left:right])

    return cell_indexes, masks, targets

def parallel_sigma_optimisation(cell_indexes, cell_masks, cell_targets, sigmas, intensities, scaling_factor):
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(
                optimize_single_sigma,
                i,
                cell_indexes,
                cell_masks,
                cell_targets,
                sigmas,
                intensities,
                scaling_factor
            )
            for i in range(len(cell_indexes))
        ]

        optimized_sigmas = list(sigmas)
        for i, f in enumerate(futures):
            optimized_sigmas[i] = f.result()

    return optimized_sigmas


def optimize_single_sigma(i, cell_indexes, cell_masks, cell_targets, sigmas, intensities, scaling_factor):
    local_cell_ids = np.unique(cell_masks[i])
    local_cell_ids = local_cell_ids[local_cell_ids > 0]

    global_pos = {cell_id: j for j, cell_id in enumerate(cell_indexes)}

    local_sigmas = [sigmas[global_pos[cell_id]] for cell_id in local_cell_ids]
    local_intensities = [intensities[global_pos[cell_id]] for cell_id in local_cell_ids]

    target_cell_id = cell_indexes[i]
    target_local_idx = np.where(local_cell_ids == target_cell_id)[0][0]

    step = max(1, int(round(scaling_factor / 2)))
    max_sigma = max(1, int(sigmas[i]))
    sigma_trials = range(1, max_sigma + 1, step)

    best_sigma = local_sigmas[target_local_idx]

    best_score = mean_absolute_percentage_error(
        cell_targets[i],
        generate_msi_from_mask_slice(
            cell_masks[i],
            local_sigmas,
            local_intensities,
            0,
            0,
            0,
            ms_spacing=scaling_factor
        )
    )

    for sigma in sigma_trials:
        trial_sigmas = list(local_sigmas)
        trial_sigmas[target_local_idx] = sigma

        score = mean_absolute_percentage_error(
            cell_targets[i],
            generate_msi_from_mask_slice(
                cell_masks[i],
                trial_sigmas,
                local_intensities,
                0,
                0,
                0,
                ms_spacing=scaling_factor
            )
        )

        if score < best_score:
            best_score = score
            best_sigma = sigma

    return best_sigma

def apply_noise_model(canvas, target):
    r = target - canvas
    shape = r.shape
    r_flat = r.ravel()
    mu = r_flat.mean()
    sigma = r_flat.std()
    g = np.random.normal(mu, sigma, size=r_flat.shape)
    idx = np.argsort(r_flat)
    g_sorted = np.sort(g)
    g_assigned = np.empty_like(g)
    g_assigned[idx] = g_sorted
    g_assigned = g_assigned.reshape(shape)
    result = canvas + g_assigned
    return result

if __name__ == "__main__":
    mask = np.load("input/seg.npy").astype(np.int32)
    canvas = np.load("input/msi_data.npy")[:, :, 500]
    p = np.percentile(canvas, 99)
    canvas = np.clip(canvas, 0, p)
    mask, canvas = crop_canvas_and_msi(mask, canvas, 110, 100, 100, 100)
    
    #plot mask
    fig, ax = plt.subplots()
    cax = ax.imshow(mask, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=ax)
    ax.set_title("Cell masks")
    plt.savefig("output/mask.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    #plot mask
    fig, ax = plt.subplots()
    cax = ax.imshow(canvas, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=ax)
    ax.set_title("Cell masks")
    plt.savefig("output/target_canvas.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    scaling_factor = mask.shape[0] // canvas.shape[0]
    diffusion_total_time = 0
    diffusion_coefficient = 0
    diffusion_barrier_alpha = 0

    cell_indexes, cell_masks, cell_targets = get_individual_cell_canvases(mask, canvas, scaling_factor, 2)
    number_of_cells = len(cell_masks)

    sigmas = []
    intensities = []
    for i, cell_target in enumerate(cell_targets):
        target_hr = np.repeat(np.repeat(cell_target, scaling_factor, axis=0), scaling_factor, axis=1)
        mask_i = (cell_masks[i] == cell_indexes[i])
        intensities.append(float(target_hr[mask_i].sum()) / (scaling_factor * scaling_factor))
        sigmas.append(cell_targets[i].shape[0]*scaling_factor)
    generated = generate_msi_from_mask_slice(
        mask,
        sigmas,
        intensities,
        diffusion_total_time,
        diffusion_coefficient,
        diffusion_barrier_alpha,
        ms_spacing=scaling_factor
    )

    mape_initial = mean_absolute_percentage_error(canvas, generated)

    # Plot comparison of target vs initial config
    vmin = min(canvas.min(), generated.min())
    vmax = max(canvas.max(), generated.max())
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im1 = axes[0].imshow(canvas, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    im2 = axes[1].imshow(generated, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title("Target")
    axes[1].set_title("Simulation")
    fig.suptitle(f"Initial comparison | MAPE-like score: {mape_initial:.6f}")
    cbar = fig.colorbar(im1, ax=axes, location='right', shrink=0.9)
    plt.savefig("output/target_v_initial.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    sigmas = parallel_sigma_optimisation(cell_indexes, cell_masks, cell_targets, sigmas, intensities, scaling_factor)

    generated = generate_msi_from_mask_slice(
        mask,
        sigmas,
        intensities,
        diffusion_total_time,
        diffusion_coefficient,
        diffusion_barrier_alpha,
        ms_spacing=scaling_factor
    )

    mape_optimized = mean_absolute_percentage_error(canvas, generated)

    # Plot comparison of target vs optimized config
    vmin = min(canvas.min(), generated.min())
    vmax = max(canvas.max(), generated.max())
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im1 = axes[0].imshow(canvas, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    im2 = axes[1].imshow(generated, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title("Target")
    axes[1].set_title("Simulation")
    fig.suptitle(f"Optimized comparison | MAPE-like score: {mape_optimized:.6f}")
    cbar = fig.colorbar(im1, ax=axes, location='right', shrink=0.9)
    plt.savefig("output/target_v_optimized.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    generated = apply_noise_model(generated, canvas)

    mape_optimized = mean_absolute_percentage_error(canvas, generated)

    # Plot comparison of target vs optimized config
    vmin = min(canvas.min(), generated.min())
    vmax = max(canvas.max(), generated.max())
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im1 = axes[0].imshow(canvas, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    im2 = axes[1].imshow(generated, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title("Target")
    axes[1].set_title("Simulation")
    fig.suptitle(f"Optimized comparison | MAPE-like score: {mape_optimized:.6f}")
    cbar = fig.colorbar(im1, ax=axes, location='right', shrink=0.9)
    plt.savefig("output/target_v_optimized_plus_noise.png", dpi=300, bbox_inches='tight')
    plt.close(fig)