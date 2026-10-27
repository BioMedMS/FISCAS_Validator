"""Cell-signal generation, diffusion, scene placement and rendering."""
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import distance_transform_edt, find_objects


def cell_profile(mask):
    """Distance-transform cell profile with a 0.1 intensity offset."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Cell mask must be 2D")
    if not binary.any():
        return np.zeros(binary.shape, dtype=float)
    distance = distance_transform_edt(binary)
    distance[~binary] = np.nan
    return np.nan_to_num(distance / np.nanmax(distance) + 0.1)


def sample_maldi_cell(mask, intensity_factors):
    factors = np.asarray(intensity_factors, dtype=float)
    if factors.ndim != 1 or not np.isfinite(factors).all():
        raise ValueError("Provide one finite intensity factor per channel")
    field = cell_profile(mask)[..., None] * factors
    return field, field.sum(axis=(0, 1))


def _cells(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError("mask must be a nonnegative 2D integer label array")
    ids = np.unique(mask)
    return mask, ids[ids > 0]


def _unit_cells(mask):
    """Yield cell IDs, bounding boxes, unit profiles and profile totals."""
    mask, ids = _cells(mask)
    if len(ids) and int(ids[-1]) > max(10_000, 10 * len(ids)):
        labels = np.searchsorted(np.r_[0, ids], mask)
        boxes = find_objects(labels)
        cell_boxes = [boxes[i] for i in range(len(ids))]
    else:
        boxes = find_objects(mask)
        cell_boxes = [boxes[int(cell_id) - 1] for cell_id in ids]
    for cell_id, box in zip(ids, cell_boxes):
        top, bottom = box[0].start, box[0].stop
        left, right = box[1].start, box[1].stop
        # Include a one-pixel background halo, clipped to the canvas.
        y0, y1 = max(0, top - 1), min(mask.shape[0], bottom + 1)
        x0, x1 = max(0, left - 1), min(mask.shape[1], right + 1)
        binary = mask[y0:y1, x0:x1] == cell_id
        field = cell_profile(binary)
        unit = field[top - y0:bottom - y0, left - x0:right - x0]
        yield int(cell_id), (top, bottom, left, right), unit, float(field.sum())


def simulator_intensity_factors(mask, intensities):
    """Convert channel-by-cell totals to distance-profile multipliers."""
    _, ids = _cells(mask)
    totals = np.asarray(intensities, dtype=float)
    if totals.ndim != 2 or totals.shape[1] != len(ids):
        raise ValueError("intensities must have shape (channels, number_of_cells)")
    weights = np.asarray([weight for _, _, _, weight in _unit_cells(mask)])
    return totals / weights[None, :]


def render_simulator_source(mask, intensities, intensity_mode="total"):
    """Render one channel with per-cell totals or distance-profile multipliers.

    intensity_mode='total' normalizes each profile to the supplied cell total.
    intensity_mode='factor' multiplies each profile by the supplied factor.
    """
    mask, ids = _cells(mask)
    values = np.asarray(intensities, dtype=float)
    if values.shape != (len(ids),) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            "provide one finite nonnegative intensity per positive mask ID")
    if intensity_mode not in ("total", "factor"):
        raise ValueError("intensity_mode must be 'total' or 'factor'")
    source = np.zeros(mask.shape, dtype=float)
    for value, (_, (top, bottom, left, right), unit, total) in zip(values, _unit_cells(mask)):
        factor = value / total if intensity_mode == "total" else value
        source[top:bottom, left:right] += unit * factor
    return source


class CPUForward:
    """CPU backend for label-weighted implicit Euler diffusion."""

    def __init__(self, mask, rtol):
        from Data_Quality_Analyzer.simulator_iterative import diffuse_mask_iterative
        self.mask, self.rtol = mask, rtol
        self._diffuse = diffuse_mask_iterative

    def source(self, field):
        return np.asarray(field, dtype=float)

    def diffuse(self, source, coefficient, alpha, time, dt, leaky=True,
                callback=None, return_device=False):
        return self._diffuse(self.mask, source, time, dt, coefficient,
                             alpha, leaky, self.rtol, maxiter=self.mask.size, callback=callback)

    def msi(self, source, d, alpha, total_time, dt, scale, leaky=True, callback=None):
        return sum_pool(self.diffuse(source, d, alpha, total_time, dt, leaky,
                                     callback), scale)


def sum_pool(field, scale):
    if int(scale) != scale or scale < 1:
        raise ValueError("scaling_factor must be a positive integer")
    scale = int(scale)
    h, w = field.shape
    if h % scale or w % scale:
        field = np.pad(field, ((0, (-h) % scale), (0, (-w) % scale)))
    return field.reshape(field.shape[0] // scale, scale,
                         field.shape[1] // scale, scale).sum(axis=(1, 3))


class CanvasGenerator:
    """Generate cell signals and MALDI images on the CPU or GPU.

    Intensities are channel-by-cell, ordered by ascending positive label ID.
    """

    def __init__(self, cell_mask, backend="auto", rtol=1e-8):
        self.mask, self.cell_ids = _cells(cell_mask)
        if not 0 < rtol < 1 or not np.isfinite(rtol):
            raise ValueError("rtol must lie between zero and one")
        if backend not in ("auto", "gpu", "cpu"):
            raise ValueError("backend must be auto, gpu or cpu")
        self.rtol = float(rtol)
        if backend in ("auto", "gpu"):
            try:
                from Data_Quality_Analyzer.simulator_gpu import GPUSimulator
                # Compact arbitrary IDs before transferring to the int32 GPU mask.
                labels = np.searchsorted(
                    np.r_[0, self.cell_ids], self.mask).astype(np.int32)
                self.forward = GPUSimulator(labels, self.rtol)
                self.backend = "gpu"
            except (ImportError, RuntimeError):
                if backend == "gpu":
                    raise
                self.forward = CPUForward(self.mask, self.rtol)
                self.backend = "cpu"
        else:
            self.forward = CPUForward(self.mask, self.rtol)
            self.backend = "cpu"

    def source(self, field):
        return self.forward.source(field)

    def msi(self, source, d, alpha, total_time, dt, scale, leaky=True, callback=None):
        return self.forward.msi(source, d, alpha, total_time, dt, scale, leaky, callback)

    @staticmethod
    def _parameters(values, channels, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = np.repeat(values, channels)
        if values.shape != (channels,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a scalar or one finite value per channel")
        return values

    def diffuse(self, source, diffusion_coefficients, barrier_alphas,
                total_time=1., dt=.1, leaky_outer_boundary=True, callback=None):
        source = np.asarray(source, dtype=float)
        single = source.ndim == 2
        if single:
            source = source[..., None]
        if source.ndim != 3 or source.shape[:2] != self.mask.shape or not np.isfinite(source).all():
            raise ValueError("Source must match the mask and have finite intensities")
        ds = self._parameters(diffusion_coefficients, source.shape[-1], "D")
        aa = self._parameters(barrier_alphas, source.shape[-1], "alpha")
        if np.any(ds < 0) or np.any((aa < 0) | (aa > 1)) or not np.isfinite(total_time + dt) or total_time < 0 or dt <= 0:
            raise ValueError("Invalid diffusion parameters")
        result = np.stack([self.forward.diffuse(self.source(source[..., c]), d, a,
                           total_time, dt, leaky_outer_boundary, callback)
                           for c, (d, a) in enumerate(zip(ds, aa))], axis=-1)
        return result[..., 0] if single else result

    def generate(self, cell_intensities, diffusion_coefficients, barrier_alphas,
                 total_time=1., dt=.1, scaling_factor=1,
                 intensity_mode="total", leaky_outer_boundary=True, callback=None):
        values = np.asarray(cell_intensities, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.cell_ids) or not len(values):
            raise ValueError("cell_intensities must be channel-by-cell")
        ds = self._parameters(diffusion_coefficients, len(values), "D")
        aa = self._parameters(barrier_alphas, len(values), "alpha")
        if int(scaling_factor) != scaling_factor or scaling_factor < 1:
            raise ValueError("scaling_factor must be a positive integer")
        if np.any(ds < 0) or np.any((aa < 0) | (aa > 1)) or not np.isfinite(total_time + dt) or total_time < 0 or dt <= 0:
            raise ValueError("Invalid diffusion parameters")
        return np.stack([self.msi(self.source(render_simulator_source(
            self.mask, values[c], intensity_mode)), d, a, total_time, dt,
            int(scaling_factor), leaky_outer_boundary, callback)
            for c, (d, a) in enumerate(zip(ds, aa))], axis=-1)


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

        cell_type = random.choices(validator.cell_types, weights=[
                                   ct.population for ct in validator.cell_types], k=1)[0]

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
                        msi_cell, cell_intensities = sample_maldi_cell(
                            cell_mask, [cell_type.get_intensity(
                                mass) for mass in validator.mass_list]
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

    engine = CanvasGenerator(
        validator.mask_canvas[h // 4:h // 4 + len(validator.cell_canvas),
                              w // 4:w // 4 + len(validator.cell_canvas[0])].astype(np.int32),
        backend=getattr(validator, "computation_backend", "cpu"),
    )
    validator.diffusion_canvas = engine.diffuse(
        validator.cell_canvas,
        total_time=validator.diffusion_total_time,
        dt=validator.diffusion_dt,
        diffusion_coefficients=validator.diffusion_coeff,
        barrier_alphas=validator.diffusion_barrier_alpha,
        leaky_outer_boundary=validator.diffusion_leaky_outer_boundary,
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
        masked_canvas = np.ma.masked_where(
            validator.mask_canvas == 0, validator.mask_canvas)
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

        # if validator.plot:
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
            skimage.io.imsave(validator.output + "/mask_canvas.tiff",
                              validator.mask_canvas.astype(np.uint16))
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

        # plt.close(fig_mic)
        # plt.close(fig_mask)
        # plt.close(fig_cell)
        # plt.close(fig_diff)
        # plt.close(fig_panel)
        # plt.close(fig_coords)
        # plt.close(fig_grid)

    # canvas-level metrics
    validator.canvas_metrics["n_cells"] = len(validator.mic_coordinates)
    validator.canvas_metrics["n_msi_cells"] = len(validator.cell_intensities)

    stop = time.perf_counter()
    validator.canvas_metrics["time"] = stop - start
