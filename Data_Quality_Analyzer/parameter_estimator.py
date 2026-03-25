import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from canvas_generation import generate_msi_from_mask_slice
from error_estimation import sum_abs_diff

def get_cell_neighbourhood(canvas, cell_index, padding, scaling_factor):
    rows_found = False
    first_collumn = 0
    last_collumn = 0
    collumns_found = False
    first_row = 0
    last_row = 0
    for i in range(canvas.shape[0]):
        if cell_index in canvas[i,:]:
            if rows_found:
                pass
            else:
                rows_found = True
                first_row = i
        else:
            if rows_found:
                last_row = i-1
                break
    for i in range(canvas.shape[1]):
        if cell_index in canvas[:,i]:
            if collumns_found:
                pass
            else:
                collumns_found = True
                first_collumn = i
        else:
            if collumns_found:
                last_collumn = i-1
                break
    left = int(max(first_collumn-first_collumn%scaling_factor-padding*scaling_factor, 0))
    right = int(min(last_collumn-last_collumn%scaling_factor+padding*scaling_factor, canvas.shape[1]-1))
    top = int(max(first_row-first_row%scaling_factor-padding*scaling_factor, 0))
    bottom = int(min(last_row-last_row%scaling_factor+padding*scaling_factor, canvas.shape[0]-1))
    return (canvas[top:bottom,left:right], (left,right,top,bottom))

def get_individual_cell_canvases(mask, msi_canvas, scaling_factor, padding):
    cell_indexes = np.unique(mask)
    cell_indexes = cell_indexes[cell_indexes != 0]
    masks = []
    targets = []
    for index in cell_indexes:
        crop = get_cell_neighbourhood(mask, index, padding, scaling_factor)
        masks.append(crop[0])
        left, right, top, bottom = tuple([round(x/scaling_factor) for x in crop[1]])
        targets.append(msi_canvas[top:bottom,left:right])
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
            scaling_factor
        ),
        method="Nelder-Mead",
        bounds=[(1, 10000), (1, 10000)],
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
    generated = generate_msi_from_mask_slice(mask_slice, sigmas, intensities, diffusion_total_time, diffusion_coefficient, diffusion_barrier_alpha, ms_spacing=scaling_factor)
    error = sum_abs_diff(target_slice, generated)
    return error

if __name__ == "__main__":
    mask = np.load("input/seg.npy").astype(np.int32)
    canvas = np.load("input/msi_data.npy")[:,:,500]
    plt.imshow(canvas, interpolation='none')
    plt.show()
    scaling_factor = mask.shape[0]/canvas.shape[0]
    diffusion_total_time = 0
    diffusion_coefficient = 0
    diffusion_barrier_alpha = 0
    cell_masks, cell_targets = get_individual_cell_canvases(mask, canvas, scaling_factor, 2)
    number_of_cells = len(cell_masks)
    sigmas = [1000]*number_of_cells
    intensities = [1]*number_of_cells
    generated = generate_msi_from_mask_slice(mask, sigmas, intensities, diffusion_total_time, diffusion_coefficient, diffusion_barrier_alpha, ms_spacing=scaling_factor)
    plt.imshow(generated, interpolation='none')
    plt.show()
    for i in range(number_of_cells):
        print(i)
        sigmas[i], intensities[i] = optimize_cell_neighbourhood(sigmas, 
                                                                intensities,
                                                                i+1,
                                                                diffusion_total_time,
                                                                diffusion_coefficient,
                                                                diffusion_barrier_alpha,
                                                                cell_masks[i],
                                                                cell_targets[i],
                                                                scaling_factor)
    generated = generate_msi_from_mask_slice(mask, sigmas, intensities, diffusion_total_time, diffusion_coefficient, diffusion_barrier_alpha, ms_spacing=scaling_factor)
    plt.imshow(generated, interpolation='none')
    plt.show()