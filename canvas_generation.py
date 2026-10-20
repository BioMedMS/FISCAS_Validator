import numpy as np
import random
import time
import matplotlib.pyplot as plt
import skimage

from diffusion import diffuse_mask_batched


def slide_cell_until_free(mask, mask_canvas, p_top_left):
    h_cell, w_cell = mask.shape
    H_canvas, W_canvas = mask_canvas.shape

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]
    dx, dy = random.choice(directions)

    x0, y0 = p_top_left

    while True:
        new_x0 = x0 + dx
        new_y0 = y0 + dy
        new_x1 = new_x0 + w_cell
        new_y1 = new_y0 + h_cell

        if (new_x0 <= 0 or new_y0 <= 0 or
                new_x1 >= W_canvas or new_y1 >= H_canvas):
            return False, p_top_left

        canvas_area = mask_canvas[new_y0:new_y1, new_x0:new_x1]

        if not np.any(np.logical_and(canvas_area, mask)):
            return True, (new_x0, new_y0)

        x0, y0 = new_x0, new_y0


def generate_and_plot_canvases(validator):
    """
    Populate microscopy / MSI canvases, run diffusion, and (optionally) plot & save them.
    All results are written back onto the `validator` instance.
    """



    start = time.perf_counter()

    # spacing & conversion factors
    scaling_factor = int(validator.ms_spacing / validator.mic_spacing)
    validator.mic_spacing = validator.ms_spacing / scaling_factor
    mic_to_ms = validator.mic_spacing / validator.ms_spacing
    ms_to_mic = validator.ms_spacing / validator.mic_spacing


    # initializing empty canvasses
    validator.mic_canvas = np.zeros(
        (int(validator.canvas_size / validator.mic_spacing),
         int(validator.canvas_size / validator.mic_spacing),
         3)
    )
    validator.mask_canvas = np.zeros(
        (int(validator.canvas_size / validator.mic_spacing),
         int(validator.canvas_size / validator.mic_spacing))
    )
    validator.cell_canvas = np.zeros(
        (int(0.5 * validator.canvas_size / validator.mic_spacing),
         int(0.5 * validator.canvas_size / validator.mic_spacing),
         len(validator.mass_list))
    )
    validator.ms_canvas = np.zeros(
        (int(np.ceil(len(validator.mic_canvas) * 0.5 * mic_to_ms)),
         int(np.ceil(len(validator.mic_canvas[0]) * 0.5 * mic_to_ms)),
         len(validator.mass_list))
    )


    # MSI TL/BR in microscopy coordinates
    msi_tl = (
        int(0.25 * len(validator.mic_canvas[0])),
        int(0.25 * len(validator.mic_canvas))
    )
    msi_br = (
        int(0.25 * len(validator.mic_canvas[0]) + len(validator.ms_canvas) * ms_to_mic),
        int(0.25 * len(validator.mic_canvas) + len(validator.ms_canvas[0]) * ms_to_mic)
    )

    # expose these to the Validator instance for later use (registration & spectra)
    validator.msi_tl = msi_tl
    validator.msi_br = msi_br


    density = 0.0
    i = 0  # counter for placement iterations
    current_id = 1  # counter for current mask index

    # loop to populate the canvas with cells
    while density < validator.cell_density:
        if i > validator.max_cell_placement_iterations:
            # print(f"aborted populating canvas. Too many cell placement cycles ({i})")
            break

        i += 1

        # choose random canvas coordinate to place the cell
        x = random.randint(0, len(validator.mic_canvas[0]))
        y = random.randint(0, len(validator.mic_canvas))
        p = (x, y)

        cell_type = random.choices(validator.cell_types, weights=[ct.population for ct in validator.cell_types], k=1)[0]
        
        cell, mask, current_mask_index = validator.load_random_cell(cell_type.get_directory(),
            mask_index=current_id
        )


        # prevent cell images to violate canvas border
        p_top_left = (
            int(p[0] - cell.shape[1] // 2),
            int(p[1] - cell.shape[0] // 2)
        )
        p_bottom_right = (
            int(p[0] + cell.shape[1] // 2),
            int(p[1] + cell.shape[0] // 2)
        )

        if (p_top_left[0] > 0
            and p_top_left[1] > 0
            and p_bottom_right[0] < len(validator.mic_canvas[0])
            and p_bottom_right[1] < len(validator.mic_canvas)):

            canvas_area = validator.mask_canvas[
                p_top_left[1]:p_top_left[1] + len(mask),
                p_top_left[0]:p_top_left[0] + len(mask[0])
            ]

            overlap = np.any(np.logical_and(canvas_area, mask))

            if overlap:
                success, new_p_top_left = slide_cell_until_free(
                    mask,
                    validator.mask_canvas,
                    p_top_left
                )

                if not success:
                    current_id = current_mask_index + 1
                    continue

                p_top_left = (
                    int(new_p_top_left[0]),
                    int(new_p_top_left[1])
                )
                p_bottom_right = (
                    p_top_left[0] + cell.shape[1],
                    p_top_left[1] + cell.shape[0]
                )
                canvas_area = validator.mask_canvas[
                    p_top_left[1]:p_top_left[1] + len(mask),
                    p_top_left[0]:p_top_left[0] + len(mask[0])
                ]
                x = p_top_left[0] + cell.shape[1] // 2
                y = p_top_left[1] + cell.shape[0] // 2
                p = (x, y)

            if not np.any(np.logical_and(canvas_area, mask)):
                # draw cell on mic_canvas; += prevents the background to override existing cells
                validator.mic_canvas[
                    p_top_left[1]:p_top_left[1] + len(cell),
                    p_top_left[0]:p_top_left[0] + len(cell[0]),
                    1
                ] += cell[:, :, 1]

                validator.mic_canvas[
                    p_top_left[1]:p_top_left[1] + len(cell),
                    p_top_left[0]:p_top_left[0] + len(cell[0]),
                    2
                ] += cell[:, :, 2]

                validator.mask_canvas[
                    p_top_left[1]:p_top_left[1] + len(mask),
                    p_top_left[0]:p_top_left[0] + len(mask[0])
                ] += mask

                # store microscopy coordinates for all masks of this cell (including multiplets)
                for cell_id in range(current_id, current_mask_index + 1):
                    validator.mic_coordinates[cell_id] = (x, y)

                density = validator.calculate_cell_density()

                # add MSI data, only if the cell is within the MSI region
                if (p_top_left[0] > 0.25 * len(validator.mic_canvas[0])
                    and p_top_left[0] < 0.75 * len(validator.mic_canvas[0])
                    and p_top_left[1] > 0.25 * len(validator.mic_canvas)
                    and p_top_left[1] < 0.75 * len(validator.mic_canvas)):

                    # conversion to local ms coordinate system (cell_canvas)
                    p_top_left_cell = np.array(
                        (p_top_left[0] - msi_tl[0],
                         p_top_left[1] - msi_tl[1]),
                        dtype=np.uint16
                    )
                    p_bottom_right_cell = np.array(
                        (p_bottom_right[0] - msi_tl[0],
                         p_bottom_right[1] - msi_tl[1]),
                        dtype=np.uint16
                    )

                    # center of the cell in MSI pixel coordinates (global MSI canvas)
                    p_ms = np.array(
                        (round((p[0] - msi_tl[0]) * mic_to_ms),
                         round((p[1] - msi_tl[1]) * mic_to_ms)),
                        dtype=np.uint16
                    )

                    for cell_id in range(current_id, current_mask_index + 1):
                        cell_mask = np.zeros((len(mask), len(mask[0])))
                        mask_region = validator.mask_canvas[
                            p_top_left[1]:p_top_left[1] + len(mask),
                            p_top_left[0]:p_top_left[0] + len(mask[0])
                        ]
                        cell_mask[mask_region == cell_id] = cell_id
                        msi_cell, cell_intensities = validator.generate_maldi_cell(
                            cell_type,
                            cell_mask,
                            sigma=40
                        )

                        if (p_top_left_cell[0] >= 0
                            and p_top_left_cell[1] >= 0
                            and p_bottom_right_cell[0] < len(validator.cell_canvas[0])
                            and p_bottom_right_cell[1] < len(validator.cell_canvas)):

                            for j in range(len(validator.mass_list)):
                                validator.cell_canvas[
                                    p_top_left_cell[1]:p_top_left_cell[1] + len(msi_cell),
                                    p_top_left_cell[0]:p_top_left_cell[0] + len(msi_cell[0]),
                                    j
                                ] += msi_cell[:, :, j]

                            # store local ms coordinates (center of cell in MSI grid)
                            validator.msi_coordinates[cell_id] = (
                                int(p_ms[0]),
                                int(p_ms[1])
                            )
                            # store total msi cell intensity
                            validator.cell_intensities[cell_id] = cell_intensities
                            validator.cell_labels[cell_id] = cell_type.name

        current_id = current_mask_index + 1

    
    # diffusion & downsampling into final MSI canvas
    h, w = validator.mask_canvas.shape

    validator.diffusion_canvas = diffuse_mask_batched(
        validator.mask_canvas[h // 4: h // 4 + len(validator.cell_canvas), w // 4: w // 4 + len(validator.cell_canvas[0])],
        validator.cell_canvas,
        total_time=validator.diffusion_total_time,
        dt=validator.diffusion_dt,
        diffusion_coeff=validator.diffusion_coeff,
        barrier_alpha=validator.diffusion_barrier_alpha,
        leaky_outer_boundary=validator.diffusion_leaky_outer_boundary
    )


    

    # plotting canvases if requested
    if validator.plot or validator.save_img:
        # --- Microscopy Canvas ---
        fig_mic, ax_mic = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_mic.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("Microscopy Canvas")
        ax_mic.imshow(validator.mic_canvas[:, :])
        ax_mic.axis("off")

        # --- Ground Truth Mask Canvas ---
        fig_mask, ax_mask = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_mask.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("Ground Truth Mask Canvas")
        masked_canvas = np.ma.masked_where(validator.mask_canvas == 0, validator.mask_canvas)
        my_cmap = plt.cm.prism.copy()
        my_cmap.set_bad(color="black")
        ax_mask.imshow(masked_canvas, cmap=my_cmap)
        ax_mask.set_title("Ground Truth Mask Canvas")
        ax_mask.axis("off")

        

        # --- MSI Canvas Origin Resolution ---

        fig_cell_panel, ax = plt.subplots(2, 2, figsize=(6, 6))
        mgr = getattr(fig_cell_panel.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas After Diffusion")

        ax[0, 0].imshow(
            validator.cell_canvas[:, :, 0],
            vmax=np.percentile(validator.cell_canvas[:, :, 0], 99.0),
            cmap="inferno",
        )
        ax[0, 0].set_title(f"m/z {validator.mass_list[0]}")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(
            validator.cell_canvas[:, :, 1],
            vmax=np.percentile(validator.cell_canvas[:, :, 1], 99.0),
            cmap="inferno",
        )
        ax[0, 1].set_title(f"m/z {validator.mass_list[1]}")
        ax[0, 1].axis("off")

        ax[1, 0].imshow(
            validator.cell_canvas[:, :, 2],
            vmax=np.percentile(validator.cell_canvas[:, :, 2], 99.0),
            cmap="inferno",
        )
        ax[1, 0].set_title(f"m/z {validator.mass_list[2]}")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(
            validator.cell_canvas[:, :, 3],
            vmax=np.percentile(validator.cell_canvas[:, :, 3], 99.0),
            cmap="inferno",
        )
        ax[1, 1].set_title(f"m/z {validator.mass_list[3]}")
        ax[1, 1].axis("off")

        
        fig_cell, ax_cell = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_cell.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas Origin Resolution")
        ax_cell.imshow(
            validator.cell_canvas[:, :, 0],
            vmax=np.percentile(validator.cell_canvas[:, :, 0], 99.0),
            cmap="inferno",
        )
        ax_cell.axis("off")

        # --- MSI Canvas After Diffusion (single channel) ---
        fig_diff, ax_diff = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_diff.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas After Diffusion")
        ax_diff.imshow(
            validator.diffusion_canvas[:, :, 0],
            vmax=np.percentile(validator.diffusion_canvas[:, :, 0], 99.0),
            cmap="inferno",
        )
        ax_diff.axis("off")

        # --- MSI Canvas After Diffusion (2x2 panel) ---
        fig_panel, ax = plt.subplots(2, 2, figsize=(6, 6))
        mgr = getattr(fig_panel.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas After Diffusion")

        ax[0, 0].imshow(
            validator.diffusion_canvas[:, :, 0],
            vmax=np.percentile(validator.diffusion_canvas[:, :, 0], 99.0),
            cmap="inferno",
        )
        ax[0, 0].set_title(f"m/z {validator.mass_list[0]}")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(
            validator.diffusion_canvas[:, :, 1],
            vmax=np.percentile(validator.diffusion_canvas[:, :, 1], 99.0),
            cmap="inferno",
        )
        ax[0, 1].set_title(f"m/z {validator.mass_list[1]}")
        ax[0, 1].axis("off")

        ax[1, 0].imshow(
            validator.diffusion_canvas[:, :, 2],
            vmax=np.percentile(validator.diffusion_canvas[:, :, 2], 99.0),
            cmap="inferno",
        )
        ax[1, 0].set_title(f"m/z {validator.mass_list[2]}")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(
            validator.diffusion_canvas[:, :, 3],
            vmax=np.percentile(validator.diffusion_canvas[:, :, 3], 99.0),
            cmap="inferno",
        )
        ax[1, 1].set_title(f"m/z {validator.mass_list[3]}")
        ax[1, 1].axis("off")

        
        #if validator.plot:
        #    plt.show()
        # --- Save if requested ---
        if validator.save_img:
            fig_mic.savefig(
                validator.output + "/mic_canvas.svg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            fig_mask.savefig(
                validator.output + "/mask_canvas.svg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            skimage.io.imsave(validator.output + "/mask_canvas.tiff", validator.mask_canvas.astype(np.uint16))
            fig_cell.savefig(
                validator.output + "/cell_canvas.svg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            fig_panel.savefig(
                            validator.output + "/panel_canvas.svg",
                            bbox_inches="tight",
                            pad_inches=0,
                            dpi=600,
                        )
            fig_cell_panel.savefig(
                            validator.output + "/cell_panel_canvas.svg",
                            bbox_inches="tight",
                            pad_inches=0,
                            dpi=600,
                        )
            
            
        #plt.close(fig_mic)
        #plt.close(fig_mask)
        #plt.close(fig_cell)
        #plt.close(fig_diff)
        #plt.close(fig_panel)
        #plt.close(fig_coords)
        #plt.close(fig_grid)

                

    # canvas-level metrics
    validator.canvas_metrics["n_cells"] = len(validator.mic_coordinates)
    validator.canvas_metrics["n_msi_cells"] = len(validator.cell_intensities)

    stop = time.perf_counter()
    validator.canvas_metrics["time"] = stop - start