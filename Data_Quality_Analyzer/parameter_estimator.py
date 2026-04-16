import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import matplotlib.pyplot as plt

from canvas_generation import generate_msi_from_mask_slice
from error_estimation import sum_abs_diff

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
    right = int(min(last_collumn - last_collumn % scaling_factor + padding * scaling_factor, canvas.shape[1] - 1))
    top = int(max(first_row - first_row % scaling_factor - padding * scaling_factor, 0))
    bottom = int(min(last_row - last_row % scaling_factor + padding * scaling_factor, canvas.shape[0] - 1))
    return (canvas[top:bottom, left:right], (left, right, top, bottom))

def get_individual_cell_canvases(mask, msi_canvas, scaling_factor, padding):
    cell_indexes = np.unique(mask)
    cell_indexes = cell_indexes[cell_indexes != 0]
    masks = []
    targets = []

    for index in cell_indexes:
        crop = get_cell_neighbourhood(mask, index, padding, scaling_factor)
        masks.append(crop[0])
        left, right, top, bottom = tuple([round(x / scaling_factor) for x in crop[1]])
        targets.append(msi_canvas[top:bottom, left:right])

    return masks, targets

def optimize_cell_neighbourhood(sigmas,
                                intensities,
                                cell_index,
                                diffusion_total_time,
                                diffusion_coefficient,
                                diffusion_barrier_alpha,
                                canvas,
                                target,
                                scaling_factor):
    result = minimize(
        objective_cell_params,
        x0=[sigmas[cell_index - 1], intensities[cell_index - 1]],
        args=(
            cell_index,
            canvas,
            target,
            sigmas,
            intensities,
            diffusion_total_time,
            diffusion_coefficient,
            diffusion_barrier_alpha,
            scaling_factor,
        ),
        method="Nelder-Mead",
        bounds=[(sigmas[cell_index - 1]-100, sigmas[cell_index - 1]+100), (intensities[cell_index - 1]-100, intensities[cell_index - 1]+100)],
    )
    return result.x

def objective_cell_params(params,
                          cell_index,
                          mask_slice,
                          target_slice,
                          global_sigmas,
                          global_intensities,
                          diffusion_total_time,
                          diffusion_coefficient,
                          diffusion_barrier_alpha,
                          scaling_factor):
    sigma, intensity = params

    sigmas = global_sigmas.copy()
    intensities = global_intensities.copy()

    sigmas[cell_index - 1] = sigma
    intensities[cell_index - 1] = intensity

    generated = generate_msi_from_mask_slice(
        mask_slice,
        sigmas,
        intensities,
        diffusion_total_time,
        diffusion_coefficient,
        diffusion_barrier_alpha,
        ms_spacing=scaling_factor,
    )
    error = sum_abs_diff(target_slice, generated)
    return error

def optimize_one_cell_from_snapshot(i,
                                    sigmas_snapshot,
                                    intensities_snapshot,
                                    diffusion_total_time,
                                    diffusion_coefficient,
                                    diffusion_barrier_alpha,
                                    cell_mask,
                                    cell_target,
                                    scaling_factor):
    sigma, intensity = optimize_cell_neighbourhood(
        sigmas_snapshot,
        intensities_snapshot,
        i + 1,
        diffusion_total_time,
        diffusion_coefficient,
        diffusion_barrier_alpha,
        cell_mask,
        cell_target,
        scaling_factor,
    )
    return i, sigma, intensity

def parallel_optimize_cells(sigmas,
                            intensities,
                            number_of_cells,
                            cell_masks,
                            cell_targets,
                            diffusion_total_time,
                            diffusion_coefficient,
                            diffusion_barrier_alpha,
                            scaling_factor,
                            max_workers=4):
    state_lock = threading.Lock()

    def submit_job(executor, i):
        with state_lock:
            sigmas_snapshot = sigmas.copy()
            intensities_snapshot = intensities.copy()

        future = executor.submit(
            optimize_one_cell_from_snapshot,
            i,
            sigmas_snapshot,
            intensities_snapshot,
            diffusion_total_time,
            diffusion_coefficient,
            diffusion_barrier_alpha,
            cell_masks[i],
            cell_targets[i],
            scaling_factor,
        )
        return future

    next_index = 0
    running = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_index < number_of_cells and len(running) < max_workers:
            future = submit_job(executor, next_index)
            running[future] = next_index
            print(f"submitted {next_index}")
            next_index += 1

        while running:
            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                submitted_i = running.pop(future)
                i, sigma, intensity = future.result()

                with state_lock:
                    sigmas[i] = sigma
                    intensities[i] = intensity

                print(f"finished {i}")

                if next_index < number_of_cells:
                    new_future = submit_job(executor, next_index)
                    running[new_future] = next_index
                    print(f"submitted {next_index}")
                    next_index += 1

    return sigmas, intensities

def apply_noise_model(canvas, target):
    r = (canvas - target)
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
    result = target + g_assigned
    return result

if __name__ == "__main__":
    mask = np.load("input/seg.npy").astype(np.int32)
    canvas = np.load("input/msi_data.npy")[:, :, 500]
    p = np.percentile(canvas, 99)
    canvas = np.clip(canvas, 0, p)
    mask, canvas = crop_canvas_and_msi(mask, canvas, 70, 70, 70, 70)

    plt.imshow(canvas, interpolation="none")
    plt.show()

    scaling_factor = mask.shape[0] / canvas.shape[0]
    diffusion_total_time = 0
    diffusion_coefficient = 0
    diffusion_barrier_alpha = 0

    cell_masks, cell_targets = get_individual_cell_canvases(mask, canvas, scaling_factor, 2)
    number_of_cells = len(cell_masks)

    sigmas = []
    intensities = []
    for cell_target in cell_targets:
        intensities.append(int(np.max(cell_target)))
        sigmas.append(round(cell_target.shape[0])*scaling_factor)
    generated = generate_msi_from_mask_slice(
        mask,
        sigmas,
        intensities,
        diffusion_total_time,
        diffusion_coefficient,
        diffusion_barrier_alpha,
        ms_spacing=scaling_factor,
    )
    plt.imshow(generated, interpolation="none")
    plt.show()

    sigmas, intensities = parallel_optimize_cells(
        sigmas=sigmas,
        intensities=intensities,
        number_of_cells=number_of_cells,
        cell_masks=cell_masks,
        cell_targets=cell_targets,
        diffusion_total_time=diffusion_total_time,
        diffusion_coefficient=diffusion_coefficient,
        diffusion_barrier_alpha=diffusion_barrier_alpha,
        scaling_factor=scaling_factor,
        max_workers=12,  
    )

    generated = generate_msi_from_mask_slice(
        mask,
        sigmas,
        intensities,
        diffusion_total_time,
        diffusion_coefficient,
        diffusion_barrier_alpha,
        ms_spacing=scaling_factor,
    )
    
    plt.imshow(generated, interpolation="none")
    plt.show()

    generated = apply_noise_model(generated, canvas)
    plt.imshow(generated, interpolation="none")
    plt.show()