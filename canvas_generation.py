"""Cell-signal generation, diffusion, scene placement perturbation and rendering."""
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from PIL import Image
import copy
import glob
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import distance_transform_edt, find_objects


def get_image_center(image):
    size = image.GetSize()
    spacing = image.GetSpacing()
    center = [(index) * spacing[dim]
              for dim, index in enumerate(np.array(size) / 2)]
    return center


def perturbation_transform(center, rotation, scale):
    """Build the shared rotation and scaling transform around a physical centre."""
    import SimpleITK as sitk

    scaling_transform = sitk.ScaleTransform(2)
    scaling_transform.SetScale((1 / scale, 1 / scale))
    scaling_transform.SetCenter(center)
    rotation_transform = sitk.Euler2DTransform()
    rotation_transform.SetCenter(center)
    rotation_transform.SetAngle(np.deg2rad(rotation))
    transform = sitk.CompositeTransform(2)
    transform.AddTransform(rotation_transform)
    transform.AddTransform(scaling_transform)
    return transform


def pertubate_img(image, rotation, scale, inverse=False):
    """Resample an image with the scene's geometric perturbation."""
    import SimpleITK as sitk

    array_input = not isinstance(image, sitk.Image)
    img = sitk.GetImageFromArray(image) if array_input else image
    transform = perturbation_transform(get_image_center(img), rotation, scale)
    sampling_transform = transform.GetInverse() if inverse else transform
    result = sitk.Resample(img, img, sampling_transform, sitk.sitkNearestNeighbor,
                           0.0, img.GetPixelID())
    return transform, sitk.GetArrayFromImage(result) if array_input else result


def pertubate_coordinates(coordinates, center, rotation, scale, inverse=False):
    """Transform landmark coordinates using the scene's geometric perturbation."""
    transform = perturbation_transform(center, rotation, scale)
    if inverse:
        transform = transform.GetInverse()
    return np.array([transform.TransformPoint(point) for point in coordinates])


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
                             alpha, leaky, self.rtol, callback=callback)

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


def slide_cell_until_free(mask, mask_canvas, p_top_left, rng=None):
    rng = random if rng is None else rng
    h_cell, w_cell = mask.shape
    H_canvas, W_canvas = mask_canvas.shape

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]
    dx, dy = rng.choice(directions)

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


class CanvasSettings:
    """Settings for synthetic microscopy and MALDI generation.

    mic_spacing is adjusted to ms_spacing / floor(ms_spacing / mic_spacing).
    A noninteger pixel-spacing ratio prints the requested and effective spacing.
    """

    def __init__(self, cell_types, *,
                 canvas_size, mic_spacing, ms_spacing, cell_density,
                 diffusion_total_time, diffusion_dt, diffusion_coeff,
                 diffusion_barrier_alpha, output="./output",
                 max_cell_placement_iterations=1000, rotation=0, scale=1.0,
                 mass_list=None, seed=0, plot=True, save_img=True,
                 texture=None, texture_path=None, texture_epsilon_add=0,
                 texture_epsilon_multiply=0, diffusion_leaky_outer_boundary=True,
                 computation_backend="cpu"):
        if not cell_types:
            raise ValueError("Provide at least one cell type")
        channel_count = len(cell_types[0].parameters) - 1
        channel_ids = list(range(1, channel_count + 1))
        if channel_count < 1 or any(
                set(cell.parameters) != {"path", *channel_ids} for cell in cell_types):
            raise ValueError(
                "Cell parameters must contain path and channels numbered 1, 2, 3, ...")
        if mass_list is None:
            mass_list = channel_ids
        else:
            if not isinstance(mass_list, (str, Path)):
                raise TypeError("mass_list must be a .npy file path")
            mass_list = np.load(mass_list, allow_pickle=False)
            if (mass_list.ndim != 1 or len(mass_list) != channel_count
                    or not np.isfinite(mass_list).all() or len(np.unique(mass_list)) != channel_count):
                raise ValueError(
                    "Mass file must contain one unique finite m/z per channel")
            cell_types = copy.deepcopy(cell_types)
            for cell in cell_types:
                original = cell.parameters
                cell.parameters = {"path": original["path"], **{
                    mass: original[channel] for channel, mass in zip(channel_ids, mass_list)}}
        for name, value in (("canvas_size", canvas_size), ("mic_spacing", mic_spacing),
                            ("ms_spacing", ms_spacing)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if ms_spacing < mic_spacing:
            raise ValueError("ms_spacing must be at least mic_spacing")
        if not np.isfinite(cell_density) or not 0 < cell_density <= 1:
            raise ValueError("cell_density must lie in (0, 1]")
        if (not np.isfinite([diffusion_total_time, diffusion_dt]).all()
                or diffusion_total_time < 0 or diffusion_dt <= 0
                or (diffusion_total_time > 0 and diffusion_dt >= diffusion_total_time)):
            raise ValueError(
                "Require time >= 0 and dt > 0, with dt < time when diffusion is enabled")
        for name, value in (("diffusion_coeff", diffusion_coeff),
                            ("diffusion_barrier_alpha", diffusion_barrier_alpha)):
            values = np.asarray(value)
            if (values.ndim > 1 or (values.ndim == 1 and len(values) != channel_count)
                    or not np.isfinite(values).all() or np.any(values < 0)
                    or (name == "diffusion_barrier_alpha" and np.any(values > 1))):
                raise ValueError(
                    f"Invalid {name}: supply a scalar or one value per channel")
        if texture is not None and texture_path is None:
            raise ValueError("texture_path is required when texture is enabled")
        if computation_backend not in ("auto", "gpu", "cpu"):
            raise ValueError("computation_backend must be auto, gpu or cpu")
        if not np.isfinite(rotation) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Require finite rotation and positive scale")
        self.cell_types = cell_types
        self.canvas_size = canvas_size
        self.mic_spacing = mic_spacing
        self.ms_spacing = ms_spacing
        self.cell_density = cell_density
        self.diffusion_total_time = diffusion_total_time
        self.diffusion_dt = diffusion_dt
        self.diffusion_coeff = diffusion_coeff
        self.diffusion_barrier_alpha = diffusion_barrier_alpha
        self.output = output
        self.max_cell_placement_iterations = max_cell_placement_iterations
        self.rotation = rotation
        self.scale = scale
        self.mass_list = mass_list
        self.seed = seed
        self.plot = plot
        self.save_img = save_img
        self.texture = texture
        self.texture_path = texture_path
        self.texture_epsilon_add = texture_epsilon_add
        self.texture_epsilon_multiply = texture_epsilon_multiply
        self.diffusion_leaky_outer_boundary = diffusion_leaky_outer_boundary
        self.computation_backend = computation_backend


@dataclass
class CanvasResult:
    """Generated images, ground truth and settings supplied to validation."""

    settings: dict
    mic_canvas: np.ndarray
    mask_canvas: np.ndarray
    cell_canvas: np.ndarray
    ms_canvas: np.ndarray
    diffusion_canvas: np.ndarray
    maldi_canvas: np.ndarray
    registration_image: np.ndarray
    mic_coordinates: dict
    msi_coordinates: dict
    cell_intensities: dict
    cell_labels: dict
    msi_tl: object
    msi_br: object
    pertubation_transform_maldi: object
    canvas_metrics: dict
    mic_spacing: float
    ms_spacing: float
    mass_list: list | np.ndarray
    random_state: tuple


class CellType:

    def __init__(self, name, parameters, population):
        self.name = name
        self.parameters = parameters
        self.population = population
        return None

    def get_distribution(self, mz):
        return self.parameters[mz][0]

    def get_location(self, mz):
        return self.parameters[mz][1]

    def get_scale(self, mz):
        return self.parameters[mz][2]

    def get_intensity(self, mz, rng=None):
        rng = np.random if rng is None else rng
        if self.parameters[mz][0] == "normal":
            intensity = np.abs(rng.normal(
                loc=self.parameters[mz][1], scale=self.parameters[mz][2]))
        elif self.parameters[mz][0] == "uniform":
            low = self.parameters[mz][1] - self.parameters[mz][2]
            high = self.parameters[mz][1] + self.parameters[mz][2]
            intensity = np.abs(rng.uniform(low=low, high=high))
        return intensity

    def get_directory(self):
        return self.parameters["path"]


def load_random_cell(cell_directory, mask_index=255, seed=None, rng=None):
    rng = random if rng is None else rng
    all_images = glob.glob(os.path.join(
        cell_directory, '**', '*.*'), recursive=True)
    images = sorted([
        file for file in all_images if 'cell' in os.path.basename(file) and 'seg' not in os.path.basename(file)])
    masks = sorted([
        file for file in all_images if 'seg' in os.path.basename(file)])
    if seed is None:
        angle = rng.randint(0, 360)
        scaling_factor = rng.uniform(0.8, 1.2)
        cell_chosen = rng.randint(0, len(images) - 1)
    else:
        angle = 0
        scaling_factor = 1
        cell_chosen = seed
    # open corresponding cell mask
    cell_mask = np.array(Image.open(masks[cell_chosen]), dtype=np.uint16)
    mask_ids = np.unique(cell_mask)
    for mask_id in mask_ids:
        if mask_id != 0:
            cell_mask[cell_mask == mask_id] = mask_index
            mask_index += 1
    mask_index -= 1
    current_mask_index = mask_index
    # with Image.open(images[cell_chosen]) as img:
    #    cell_image = np.zeros((img.size[1], img.size[0], img.n_frames))
    #    for i in range(img.n_frames):
    #       img.seek(i)
    #       cell_image[:, :, i] = np.array(img)
    #       cell_image[:, :, i] = skimage.exposure.equalize_adapthist(np.array(img))
    cell_image = skimage.io.imread(images[cell_chosen])
    cell_rotated = skimage.transform.rotate(cell_image, angle, resize=True)
    mask_rotated = skimage.transform.rotate(
        np.array(cell_mask), angle, order=0, resize=True)
    cell_scaled = skimage.transform.rescale(
        cell_rotated, scaling_factor, anti_aliasing=True, channel_axis=-1)
    mask_scaled = skimage.transform.rescale(
        mask_rotated, scaling_factor, order=0, anti_aliasing=False)
    return cell_scaled, mask_scaled.astype(np.uint16), current_mask_index


def calculate_cell_density(mask_canvas):
    total_area = mask_canvas.shape[0] * mask_canvas.shape[1]
    cell_area = np.count_nonzero(mask_canvas)
    density = cell_area / total_area
    return density


def add_texture(image, texture_path, texture_epsilon_add):
    if np.abs(texture_epsilon_add) > 1:
        # catch epsilon values outside of [-1, 1]
        texture_epsilon_add /= np.abs(texture_epsilon_add)
    image = np.array(image)
    texture = np.array(Image.open(texture_path))
    if len(texture.shape) == 3:
        texture = np.mean(texture, axis=2)  # Convert to grayscale if RGB
    texture = skimage.transform.resize(texture, (image.shape[1], image.shape[0]))
    # Normalize texture and weight it by epsilon
    texture = texture / np.max(texture) * texture_epsilon_add
    intensity = np.percentile(image, 99)
    texture = skimage.exposure.rescale_intensity(
        texture, in_range=(0, 1), out_range=(0, intensity))
    combined_image = np.zeros_like(image)
    combined_image = texture + image
    return combined_image


def multiply_texture(image, texture_path, texture_epsilon_multiply):
    if np.abs(texture_epsilon_multiply) > 1:
        # catch epsilon values outside of [-1, 1]
        texture_epsilon_multiply /= np.abs(texture_epsilon_multiply)
    image = np.array(image)
    texture = np.array(Image.open(texture_path))
    if len(texture.shape) == 3:
        texture = np.mean(texture, axis=2)  # Convert to grayscale if RGB
    texture = skimage.transform.resize(texture, (image.shape[1], image.shape[0]))
    texture = (texture / np.max(texture)).astype(float)
    texture += texture_epsilon_multiply
    # Normalize texture and weight it by epsilon
    texture = texture / np.max(texture)
    combined_image = np.zeros_like(image)
    combined_image = texture * image
    return combined_image


def generate_and_plot_canvases(settings):
    """Generate a synthetic scene from CanvasSettings and return its ground truth."""
    if not isinstance(settings, CanvasSettings):
        raise TypeError("settings must be CanvasSettings")
    scene = SimpleNamespace(**copy.deepcopy(vars(settings)))
    snapshot = copy.deepcopy(vars(settings))
    for name in ("mic_coordinates", "msi_coordinates", "cell_intensities", "cell_labels", "canvas_metrics"):
        setattr(scene, name, {})
    scene.random_rng = random.Random(scene.seed)
    scene.intensity_rng = np.random.RandomState(scene.seed)
    if scene.plot or scene.save_img:
        plt.rcParams.update({'font.size': 30})
        n = len(scene.mass_list)
        c = int(np.ceil(np.sqrt(n)))
        fig, ax = plt.subplots(c, c, figsize=(6, 6), squeeze=False)
        k = 0
        colors = ["red", "black", "blue", "green",
                  "orange", "purple", "brown", "pink"]
        for mz in scene.mass_list:
            for c, cell_type in enumerate(scene.cell_types):
                try:
                    m = cell_type.parameters[mz][1]
                    s = cell_type.parameters[mz][2]
                    min_x = m - 4 * s
                    min_x = min_x if min_x >= 0 else 0
                    max_x = m + 4 * s
                    x = np.linspace(min_x, max_x, 100)
                    if cell_type.parameters[mz][0] == "normal":
                        y = (1 / (s * np.sqrt(2 * np.pi))) * \
                            np.exp(-0.5 * ((x - m) / s) ** 2)
                    if cell_type.parameters[mz][0] == "uniform":
                        y = np.where((x >= m - s) & (x <= m + s), 1 / (2 * s), 0)
                    ax.flat[k].plot(x, y, label=f"{cell_type.name} - " + r"$\it{m/z}$ " + f"{
                        mz} (μ={m}, σ={s})", color=colors[c])
                    ax.flat[k].legend(loc='lower center',
                                      bbox_to_anchor=(0.5, 1), ncol=1)
                except Exception as e:
                    continue
            k += 1
        plt.tight_layout()
        if scene.save_img:
            plt.savefig(scene.output + "/cell_type_distributions.svg",
                        bbox_inches='tight', pad_inches=0, dpi=600)

    _generate_scene(scene)
    scene.registration_image = scene.maldi_canvas[..., 0].copy()
    if scene.texture == "registration":
        scene.registration_image = multiply_texture(scene.registration_image, scene.texture_path,
                                                    scene.texture_epsilon_multiply)
        scene.registration_image = add_texture(scene.registration_image, scene.texture_path,
                                               scene.texture_epsilon_add)
    return CanvasResult(settings=snapshot, random_state=scene.random_rng.getstate(), **{name: getattr(scene, name) for name in (
        "mic_canvas",
        "mask_canvas",
        "cell_canvas",
        "ms_canvas",
        "diffusion_canvas",
        "maldi_canvas",
        "registration_image",
        "mic_coordinates",
        "msi_coordinates",
        "cell_intensities",
        "cell_labels",
        "msi_tl",
        "msi_br",
        "pertubation_transform_maldi",
        "canvas_metrics",
        "mic_spacing",
        "ms_spacing",
        "mass_list",
    )})


def _generate_scene(scene):
    """Place cells, diffuse signals and render the generated scene."""
    start = time.perf_counter()

    # spacing & conversion factors
    requested_spacing = scene.mic_spacing
    scaling_factor = int(scene.ms_spacing / requested_spacing)
    scene.mic_spacing = scene.ms_spacing / scaling_factor
    if not np.isclose(requested_spacing, scene.mic_spacing, rtol=1e-12, atol=0):
        print(f"Adapting microscopy pixel spacing from {requested_spacing:.12g} to "
              f"{scene.mic_spacing:.12g} µm to fit {scaling_factor} microscopy pixels "
              f"per MALDI pixel ({scene.ms_spacing:.12g} µm).", flush=True)
    mic_to_ms = scene.mic_spacing / scene.ms_spacing
    ms_to_mic = scene.ms_spacing / scene.mic_spacing

    # initializing empty canvasses
    scene.mic_canvas = np.zeros(
        (int(scene.canvas_size / scene.mic_spacing),
         int(scene.canvas_size / scene.mic_spacing),
         3)
    )
    scene.mask_canvas = np.zeros(
        (int(scene.canvas_size / scene.mic_spacing),
         int(scene.canvas_size / scene.mic_spacing))
    )
    scene.cell_canvas = np.zeros(
        (int(0.5 * scene.canvas_size / scene.mic_spacing),
         int(0.5 * scene.canvas_size / scene.mic_spacing),
         len(scene.mass_list))
    )
    scene.ms_canvas = np.zeros(
        (int(np.ceil(len(scene.mic_canvas) * 0.5 * mic_to_ms)),
         int(np.ceil(len(scene.mic_canvas[0]) * 0.5 * mic_to_ms)),
         len(scene.mass_list))
    )

    # MSI TL/BR in microscopy coordinates
    msi_tl = (
        int(0.25 * len(scene.mic_canvas[0])),
        int(0.25 * len(scene.mic_canvas))
    )
    msi_br = (
        int(0.25 * len(scene.mic_canvas[0]) + len(scene.ms_canvas) * ms_to_mic),
        int(0.25 * len(scene.mic_canvas) + len(scene.ms_canvas[0]) * ms_to_mic)
    )

    # expose these to the Validator instance for later use (registration & spectra)
    scene.msi_tl = msi_tl
    scene.msi_br = msi_br

    density = 0.0
    i = 0  # counter for placement iterations
    current_id = 1  # counter for current mask index

    # loop to populate the canvas with cells
    while density < scene.cell_density:
        if i > scene.max_cell_placement_iterations:
            # print(f"aborted populating canvas. Too many cell placement cycles ({i})")
            break

        i += 1

        # choose random canvas coordinate to place the cell
        x = scene.random_rng.randint(0, len(scene.mic_canvas[0]))
        y = scene.random_rng.randint(0, len(scene.mic_canvas))
        p = (x, y)

        cell_type = scene.random_rng.choices(scene.cell_types, weights=[
            ct.population for ct in scene.cell_types], k=1)[0]

        cell, mask, current_mask_index = load_random_cell(cell_type.get_directory(),
                                                          mask_index=current_id, rng=scene.random_rng
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
            and p_bottom_right[0] < len(scene.mic_canvas[0])
                and p_bottom_right[1] < len(scene.mic_canvas)):

            canvas_area = scene.mask_canvas[
                p_top_left[1]:p_top_left[1] + len(mask),
                p_top_left[0]:p_top_left[0] + len(mask[0])
            ]

            overlap = np.any(np.logical_and(canvas_area, mask))

            if overlap:
                success, new_p_top_left = slide_cell_until_free(
                    mask,
                    scene.mask_canvas,
                    p_top_left, rng=scene.random_rng
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
                canvas_area = scene.mask_canvas[
                    p_top_left[1]:p_top_left[1] + len(mask),
                    p_top_left[0]:p_top_left[0] + len(mask[0])
                ]
                x = p_top_left[0] + cell.shape[1] // 2
                y = p_top_left[1] + cell.shape[0] // 2
                p = (x, y)

            if not np.any(np.logical_and(canvas_area, mask)):
                # draw cell on mic_canvas; += prevents the background to override existing cells
                scene.mic_canvas[
                    p_top_left[1]:p_top_left[1] + len(cell),
                    p_top_left[0]:p_top_left[0] + len(cell[0]),
                    1
                ] += cell[:, :, 1]

                scene.mic_canvas[
                    p_top_left[1]:p_top_left[1] + len(cell),
                    p_top_left[0]:p_top_left[0] + len(cell[0]),
                    2
                ] += cell[:, :, 2]

                scene.mask_canvas[
                    p_top_left[1]:p_top_left[1] + len(mask),
                    p_top_left[0]:p_top_left[0] + len(mask[0])
                ] += mask

                # store microscopy coordinates for all masks of this cell (including multiplets)
                for cell_id in range(current_id, current_mask_index + 1):
                    scene.mic_coordinates[cell_id] = (x, y)

                density = calculate_cell_density(scene.mask_canvas)

                # add MSI data, only if the cell is within the MSI region
                if (p_top_left[0] > 0.25 * len(scene.mic_canvas[0])
                    and p_top_left[0] < 0.75 * len(scene.mic_canvas[0])
                    and p_top_left[1] > 0.25 * len(scene.mic_canvas)
                        and p_top_left[1] < 0.75 * len(scene.mic_canvas)):

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
                        mask_region = scene.mask_canvas[
                            p_top_left[1]:p_top_left[1] + len(mask),
                            p_top_left[0]:p_top_left[0] + len(mask[0])
                        ]
                        cell_mask[mask_region == cell_id] = cell_id
                        msi_cell, cell_intensities = sample_maldi_cell(
                            cell_mask, [cell_type.get_intensity(
                                mass, rng=scene.intensity_rng) for mass in scene.mass_list]
                        )

                        if (p_top_left_cell[0] >= 0
                            and p_top_left_cell[1] >= 0
                            and p_bottom_right_cell[0] < len(scene.cell_canvas[0])
                                and p_bottom_right_cell[1] < len(scene.cell_canvas)):

                            for j in range(len(scene.mass_list)):
                                scene.cell_canvas[
                                    p_top_left_cell[1]:p_top_left_cell[1] + len(msi_cell),
                                    p_top_left_cell[0]:p_top_left_cell[0] + len(msi_cell[0]),
                                    j
                                ] += msi_cell[:, :, j]

                            # store local ms coordinates (center of cell in MSI grid)
                            scene.msi_coordinates[cell_id] = (
                                int(p_ms[0]),
                                int(p_ms[1])
                            )
                            # store total msi cell intensity
                            scene.cell_intensities[cell_id] = cell_intensities
                            scene.cell_labels[cell_id] = cell_type.name

        current_id = current_mask_index + 1

    # diffusion & downsampling into final MSI canvas
    h, w = scene.mask_canvas.shape

    engine = CanvasGenerator(
        scene.mask_canvas[h // 4:h // 4 + len(scene.cell_canvas),
                          w // 4:w // 4 + len(scene.cell_canvas[0])].astype(np.int32),
        backend=getattr(scene, "computation_backend", "cpu"),
    )
    scene.diffusion_canvas = engine.diffuse(
        scene.cell_canvas,
        total_time=scene.diffusion_total_time,
        dt=scene.diffusion_dt,
        diffusion_coefficients=scene.diffusion_coeff,
        barrier_alphas=scene.diffusion_barrier_alpha,
        leaky_outer_boundary=scene.diffusion_leaky_outer_boundary,
    )

    # plotting canvases if requested
    if scene.plot or scene.save_img:
        # --- Microscopy Canvas ---
        fig_mic, ax_mic = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_mic.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("Microscopy Canvas")
        ax_mic.imshow(scene.mic_canvas[:, :])
        ax_mic.axis("off")

        # --- Ground Truth Mask Canvas ---
        fig_mask, ax_mask = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_mask.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("Ground Truth Mask Canvas")
        masked_canvas = np.ma.masked_where(
            scene.mask_canvas == 0, scene.mask_canvas)
        my_cmap = plt.cm.prism.copy()
        my_cmap.set_bad(color="black")
        ax_mask.imshow(masked_canvas, cmap=my_cmap)
        ax_mask.set_title("Ground Truth Mask Canvas")
        ax_mask.axis("off")

        # --- MSI Canvas Origin Resolution ---

        fig_cell_panel, ax = plt.subplots(
            int(np.ceil(len(scene.mass_list) / 2)), min(2, len(scene.mass_list)),
            figsize=(6, 6), squeeze=False)
        mgr = getattr(fig_cell_panel.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas After Diffusion")

        for channel, axis in enumerate(ax.flat):
            if channel >= len(scene.mass_list):
                axis.axis("off")
                continue
            image = scene.cell_canvas[..., channel]
            axis.imshow(image, vmax=np.percentile(image, 99.0), cmap="inferno")
            axis.set_title(f"Channel {scene.mass_list[channel]}")
            axis.axis("off")

        fig_cell, ax_cell = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_cell.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas Origin Resolution")
        ax_cell.imshow(
            scene.cell_canvas[:, :, 0],
            vmax=np.percentile(scene.cell_canvas[:, :, 0], 99.0),
            cmap="inferno",
        )
        ax_cell.axis("off")

        # --- MSI Canvas After Diffusion (single channel) ---
        fig_diff, ax_diff = plt.subplots(figsize=(6, 6))
        mgr = getattr(fig_diff.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas After Diffusion")
        ax_diff.imshow(
            scene.diffusion_canvas[:, :, 0],
            vmax=np.percentile(scene.diffusion_canvas[:, :, 0], 99.0),
            cmap="inferno",
        )
        ax_diff.axis("off")

        # --- MSI Canvas After Diffusion (2x2 panel) ---
        fig_panel, ax = plt.subplots(
            int(np.ceil(len(scene.mass_list) / 2)), min(2, len(scene.mass_list)),
            figsize=(6, 6), squeeze=False)
        mgr = getattr(fig_panel.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title("MSI Canvas After Diffusion")

        for channel, axis in enumerate(ax.flat):
            if channel >= len(scene.mass_list):
                axis.axis("off")
                continue
            image = scene.diffusion_canvas[..., channel]
            axis.imshow(image, vmax=np.percentile(image, 99.0), cmap="inferno")
            axis.set_title(f"Channel {scene.mass_list[channel]}")
            axis.axis("off")

        # if scene.plot:
        #    plt.show()
        # --- Save if requested ---
        if scene.save_img:
            fig_mic.savefig(
                scene.output + "/mic_canvas.svg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            fig_mask.savefig(
                scene.output + "/mask_canvas.svg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            skimage.io.imsave(scene.output + "/mask_canvas.tiff",
                              scene.mask_canvas.astype(np.uint16))
            fig_cell.savefig(
                scene.output + "/cell_canvas.svg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            fig_panel.savefig(
                scene.output + "/panel_canvas.svg",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            fig_cell_panel.savefig(
                scene.output + "/cell_panel_canvas.svg",
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
    scene.canvas_metrics["n_cells"] = len(scene.mic_coordinates)
    scene.canvas_metrics["n_msi_cells"] = len(scene.cell_intensities)

    generate_perturbed_maldi(scene)

    stop = time.perf_counter()
    scene.canvas_metrics["time"] = stop - start


def generate_perturbed_maldi(scene):
    """Transform diffused signals, pool MALDI pixels and apply channel textures."""
    import SimpleITK as sitk

    ms_to_mic = scene.ms_spacing / scene.mic_spacing
    img = skimage.measure.block_reduce(
        scene.diffusion_canvas[:, :, 0],
        block_size=(int(ms_to_mic), int(ms_to_mic)),
        func=np.sum
    )
    scene.maldi_canvas = np.zeros(
        (img.shape[0], img.shape[1], len(scene.mass_list)))

    for i in range(len(scene.mass_list)):
        img = sitk.GetImageFromArray(scene.diffusion_canvas[:, :, i])
        img.SetSpacing((scene.mic_spacing, scene.mic_spacing))
        scene.pertubation_transform_maldi, pertubated_img = pertubate_img(
            img, scene.rotation, scene.scale)
        pertubated_arr = sitk.GetArrayFromImage(pertubated_img)

        if scene.plot or scene.save_img:
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(
                pertubated_arr,
                vmax=np.percentile(pertubated_arr, 99.),
                cmap='inferno'
            )
            plt.axis('off')
            if scene.save_img:
                fig.savefig(
                    scene.output + f"/pertubated_img_{i}.svg",
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=600
                )

        pertubated_maldi_img = skimage.measure.block_reduce(
            pertubated_arr,
            block_size=(int(ms_to_mic), int(ms_to_mic)),
            func=np.sum
        )

        scene.maldi_canvas[:, :, i] = pertubated_maldi_img
        if scene.texture == 'all':
            scene.maldi_canvas[:, :, i] = multiply_texture(
                scene.maldi_canvas[:, :, i],
                scene.texture_path, scene.texture_epsilon_multiply
            )
            scene.maldi_canvas[:, :, i] = add_texture(
                scene.maldi_canvas[:, :, i],
                scene.texture_path, scene.texture_epsilon_add
            )
