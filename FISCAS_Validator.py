# print("importing modules...")
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import skimage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score
import SimpleITK as sitk
from tqdm import trange
import copy
import time
from scipy.optimize import linear_sum_assignment
import pickle
from sklearn.decomposition import PCA
from canvas_generation import (CanvasSettings, CanvasResult, CellType, generate_and_plot_canvases,
                               get_image_center, perturbation_transform, pertubate_coordinates)
import uuid
from pathlib import Path

# matplotlib.use('Agg')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# automatically adjust subplot parameters to fit the figure
plt.rc('figure', autolayout=True)


class Validator:
    def __init__(self, canvas, *, output="./output", use_perfect_registration=False,
                 use_perfect_segmentation=True, teaching_imprecision=None,
                 plot=True, save_img=True, remove_ambiguous=True, cell_diameter=0,
                 render_animation=False):
        if not isinstance(canvas, CanvasResult):
            raise TypeError("canvas must be a CanvasResult from canvas generation")
        if not use_perfect_registration and teaching_imprecision is None:
            raise ValueError(
                "teaching_imprecision is required for landmark registration")
        if teaching_imprecision is not None and (
                not np.isfinite(teaching_imprecision) or teaching_imprecision < 0):
            raise ValueError("teaching_imprecision must be finite and nonnegative")
        self.canvas = canvas
        self.cell_types = tuple(cell.name for cell in canvas.settings["cell_types"])
        self.mic_canvas = canvas.mic_canvas
        self.mask_canvas = canvas.mask_canvas
        self.cell_canvas = canvas.cell_canvas
        self.ms_canvas = canvas.ms_canvas
        self.diffusion_canvas = canvas.diffusion_canvas
        self.maldi_canvas = canvas.maldi_canvas
        self.registration_image = canvas.registration_image
        self.mic_coordinates = canvas.mic_coordinates
        self.msi_coordinates = canvas.msi_coordinates
        self.cell_intensities = canvas.cell_intensities
        self.cell_labels = canvas.cell_labels
        self.msi_tl = canvas.msi_tl
        self.msi_br = canvas.msi_br
        self.pertubation_transform_maldi = canvas.pertubation_transform_maldi
        self.canvas_metrics = canvas.canvas_metrics
        self.mic_spacing = canvas.mic_spacing
        self.ms_spacing = canvas.ms_spacing
        self.mass_list = canvas.mass_list
        self.seed = canvas.settings["seed"]
        self.rotation = canvas.settings["rotation"]
        self.scale = canvas.settings["scale"]
        self.output = output
        self.use_perfect_registration = use_perfect_registration
        self.use_perfect_segmentation = use_perfect_segmentation
        self.teaching_imprecision = teaching_imprecision
        self.plot = plot
        self.save_img = save_img
        self.remove_ambiguous = remove_ambiguous
        self.cell_diameter = cell_diameter
        self.render_animation = render_animation
        self.registration_metrics = {}
        self.segmentation_metrics = {}
        self.spectra_metrics = {}
        self.cluster_metrics = {}
        plt.rcParams.update({"figure.max_open_warning": 0})

    # function to return dict with settings for report generation with quarto
    def return_settings(self):
        settings = dict(self.canvas.settings)
        settings["cell_types"] = [cell.name for cell in settings["cell_types"]]
        settings["mic_spacing"] = self.mic_spacing
        settings["ID"] = uuid.uuid4()
        settings["output"] = self.output
        settings["use_perfect_registration"] = self.use_perfect_registration
        settings["use_perfect_segmentation"] = self.use_perfect_segmentation
        settings["teaching_imprecision"] = self.teaching_imprecision
        settings["plot"] = self.plot
        settings["save_img"] = self.save_img
        settings["remove_ambiguous"] = self.remove_ambiguous
        settings["cell_diameter"] = self.cell_diameter
        settings["render_animation"] = self.render_animation
        return settings

    def decompose_affine_2d(self, matrix):
        u, s, vh = np.linalg.svd(matrix)
        rotation = np.dot(u, vh)
        scale = s
        shear_angle = np.arctan2(rotation[0, 1], rotation[0, 0])
        return rotation, scale, shear_angle

    def calculate_rotation_angle(self, rotation_matrix):
        a, b = rotation_matrix[0]
        c, d = rotation_matrix[1]
        theta_rad = np.arctan2(c, a)
        theta_deg = np.degrees(theta_rad)
        return theta_rad, theta_deg

    # interpret registration results (2d-affine)
    def interpret_registration_matrix(self, transformations):
        if isinstance(transformations, sitk.CompositeTransform):
            for i in range(transformations.GetNumberOfTransforms()):
                sub_transform = transformations.GetNthTransform(i)
                if isinstance(sub_transform, sitk.Euler2DTransform):
                    matrix = np.reshape(sub_transform.GetMatrix(), (2, 2))
                    self.registration_metrics['matrix'] = matrix
                    self.registration_metrics['translation'] = sub_transform.GetTranslation(
                    )
                    theta_rad, theta_deg = self.calculate_rotation_angle(matrix)
                    self.registration_metrics['theta_deg'] = theta_deg
                    self.registration_metrics['theta_rad'] = theta_rad
                elif isinstance(sub_transform, sitk.AffineTransform):
                    matrix = sub_transform.GetMatrix()
                    self.registration_metrics['matrix'] = matrix
                    self.registration_metrics['translation'] = sub_transform.GetTranslation(
                    )
                    rot, scale, shear = self.decompose_affine_2d(
                        np.reshape(np.asarray(matrix), (2, 2)))
                    theta_rad, theta_deg = self.calculate_rotation_angle(rot)
                    self.registration_metrics['rot'] = rot
                    self.registration_metrics['scale'] = scale
                    self.registration_metrics['shear'] = shear
                    self.registration_metrics['theta_deg'] = theta_deg
                    self.registration_metrics['theta_rad'] = theta_rad
                elif isinstance(sub_transform, sitk.TranslationTransform):
                    self.registration_metrics['translation'] = sub_transform.GetOffset()
                elif isinstance(sub_transform, sitk.Similarity2DTransform):
                    matrix = np.reshape(sub_transform.GetMatrix(), (2, 2))
                    self.registration_metrics['matrix'] = matrix
                    self.registration_metrics['translation'] = sub_transform.GetTranslation(
                    )
                    self.registration_metrics['scale'] = sub_transform.GetScale()
                    theta_rad, theta_deg = self.calculate_rotation_angle(matrix)
                    self.registration_metrics['theta_deg'] = theta_deg
                    self.registration_metrics['theta_rad'] = theta_rad
        else:
            if isinstance(transformations, sitk.AffineTransform):
                self.registration_metrics['matrix'] = transformations.GetMatrix()
                self.registration_metrics['translation'] = transformations.GetTranslation(
                )
            elif isinstance(transformations, sitk.TranslationTransform):
                self.registration_metrics['translation'] = transformations.GetOffset()
            elif isinstance(transformations, sitk.Similarity2DTransform):
                matrix = np.reshape(transformations.GetMatrix(), (2, 2))
                self.registration_metrics['matrix'] = matrix
                theta_rad, theta_deg = self.calculate_rotation_angle(matrix)
                self.registration_metrics['translation'] = transformations.GetTranslation(
                )
                self.registration_metrics['scale'] = transformations.GetScale()
                self.registration_metrics['theta_deg'] = theta_deg
                self.registration_metrics['theta_rad'] = theta_rad
        return None

    def evaluate_registration(self):
        error = []
        for key, value in self.msi_coordinates.items():
            x, y = value
            x_global, y_global = x * self.ms_spacing, y * self.ms_spacing
            x_pertubated, y_pertubated = self.pertubation_transform_maldi.GetInverse(
            ).TransformPoint((x_global, y_global))
            landmark_moving = self.final_transform.GetInverse(
            ).TransformPoint((x_pertubated, y_pertubated))
            adjusted_translation = (-self.ms_spacing / 2, -self.ms_spacing / 2)
            # adjusted_translation = (0,0)
            translation_transform = sitk.TranslationTransform(2)
            translation_transform.SetOffset(adjusted_translation)
            landmark_moving = np.array(
                translation_transform.TransformPoint(landmark_moving))
            landmark_fixed = np.array(self.mic_coordinates[key]) * self.mic_spacing
            error.append(landmark_fixed - landmark_moving)
        error = np.array(error)
        if self.plot or self.save_img:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            radii = [5, 10, 15, 20]
            for radius in radii:
                circle = plt.Circle((0, 0), radius, color='grey',
                                    fill=False, linestyle='-', linewidth=1)
                ax.add_artist(circle)
                ax.text(0, -radius, str(radius), fontsize=10, ha='center', va='center', color='grey',
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            ax.set_aspect('equal', 'box')
            limit = max(radii) + 1
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            # ax.set_xlabel('Error X in µm')
            # ax.set_ylabel('Error Y in µm')
            # ax.set_title('Registration Error Plot')
            ax.scatter(0, 0, color='grey', s=200, marker='+')
            ax.scatter(error[:, 0], error[:, 1], color='blue',
                       s=30, marker='.', zorder=10, alpha=0.5)
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/Registration_error_plot.svg",
                            bbox_inches='tight', pad_inches=0, dpi=600)
        self.registration_metrics["mre"] = np.mean(np.abs(error), axis=0)
        self.registration_metrics["mre_std"] = np.std(np.abs(error), axis=0)
        return None

    # SITK helper function
    def normalize_to_uint8(self, img):
        arr = sitk.GetArrayFromImage(img)
        normalized_arr = arr / np.max(arr) * 255
        uint8_arr = np.clip(normalized_arr, 0, 255).astype(np.uint8)
        uint8_img = sitk.GetImageFromArray(uint8_arr)
        uint8_img.CopyInformation(img)
        return uint8_img

    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0

    # match predicted mask indices with ground truth indices using Kuhn-Munkres algorithm
    def match_masks(self, predictions, ground_truths, iou_threshold=0.5):
        if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must lie between zero and one")
        num_pred = len(predictions)
        num_gt = len(ground_truths)
        iou_matrix = np.zeros((num_pred, num_gt))
        for i, pred_mask in enumerate(predictions):
            for j, gt_mask in enumerate(ground_truths):
                iou_matrix[i, j] = self.calculate_iou(pred_mask, gt_mask)
        # Negative for maximization
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = [(i, j) for i, j in zip(row_ind, col_ind)
                   if iou_matrix[i, j] > iou_threshold]
        return matches, iou_matrix

    # convert label image to binary masks
    def label_image_to_masks(self, label_image):
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels != 0]
        masks = [(label_image == label_id) for label_id in unique_labels]
        return masks

    def evaluate_segmentation(self, predictions, ground_truths, iou_threshold=0.5):
        # Edge cases: nothing to compare
        if len(predictions) == 0 and len(ground_truths) == 0:
            return {}, 0.0, 0.0, 0.0
        if len(predictions) == 0:
            # all ground truths are missed -> recall = 0
            return {}, 0.0, 0.0, 0.0
        if len(ground_truths) == 0:
            # all predictions are false positives -> precision = 0
            return {}, 0.0, 0.0, 0.0

        matches, iou_matrix = self.match_masks(
            predictions, ground_truths, iou_threshold)

        iou_all_masks = {}
        for pred_index, gt_index in matches:
            iou_all_masks[gt_index] = iou_matrix[pred_index, gt_index]

        tp = len(matches)  # true positives
        matched_pred_indices = {pred_index for pred_index, _ in matches}
        fp = len(predictions) - len(matched_pred_indices)  # false positives
        matched_gt_indices = {gt_index for _, gt_index in matches}
        fn = len(ground_truths) - len(matched_gt_indices)  # false negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return iou_all_masks, precision, recall, f1_score

    def main(self):
        rng = random.Random()
        rng.setstate(self.canvas.random_state)
        ms_to_mic = self.ms_spacing / self.mic_spacing

        maldi_sitk = sitk.GetImageFromArray(self.maldi_canvas[:, :, 0])
        maldi_sitk.SetSpacing((self.ms_spacing, self.ms_spacing))
        maldi_img = self.registration_image

        if self.plot or self.save_img:

            # --- Cell Coordinates on MSI Canvas ---
            fig_coords, ax_coords = plt.subplots(figsize=(6, 6))
            mgr = getattr(fig_coords.canvas, "manager", None)
            if mgr is not None and hasattr(mgr, "set_window_title"):
                mgr.set_window_title("Cell Coordinates on MSI Canvas")

            ax_coords.imshow(
                self.maldi_canvas[:, :, 0],
                vmax=np.percentile(self.maldi_canvas[:, :, 0], 99.0),
                cmap="inferno",
            )
            ax_coords.axis("off")

            x_coords = [xy[0] for xy in self.msi_coordinates.values()]
            y_coords = [xy[1] for xy in self.msi_coordinates.values()]
            ax_coords.scatter(x_coords, y_coords, color="deepskyblue", s=10, marker="x")

            # --- Grid of MSI channels ---
            grid_size = int(np.ceil(np.sqrt(len(self.mass_list))))
            if grid_size > 5:
                grid_size = 5  # limit grid size for better visibility

            fig_grid, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
            mgr = getattr(fig_grid.canvas, "manager", None)
            if mgr is not None and hasattr(mgr, "set_window_title"):
                mgr.set_window_title("MSI Canvas (m/z grid)")

            axes_flat = np.array(axes).reshape(-1)
            max_panels = min(len(self.mass_list), grid_size * grid_size)

            for i, mz in enumerate(self.mass_list[:max_panels]):
                ax_i = axes_flat[i]
                ax_i.set_title(f"m/z {mz}")
                ax_i.imshow(
                    self.maldi_canvas[:, :, i],
                    vmax=np.percentile(self.maldi_canvas[:, :, i], 99.0),
                    cmap="inferno",
                )
                ax_i.axis("off")

            for j in range(max_panels, len(axes_flat)):
                axes_flat[j].axis("off")

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(
                maldi_img,
                vmax=np.percentile(maldi_img, 99.),
                cmap='inferno'
            )
            plt.axis('off')

            if self.save_img:
                fig_coords.savefig(
                    self.output + "/msi_coordinates.svg",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=600,
                )
                fig_grid.savefig(
                    self.output + "/msi_canvas.svg",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=600,
                )

                fig.savefig(
                    self.output + "/pertubated_msi_canvas.svg",
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=600
                )

        # canvas_metrics["time"] was already set in generate_and_plot_canvases
        # continue with registration / segmentation / spectra as before...

        # img = np.zeros((len(self.mic_canvas), len(self.mic_canvas[0]), 3))
        # maldi_scaled = skimage.transform.rescale(self.ms_canvas[:,:,0], ms_to_mic, order=0, anti_aliasing=False)
        # start_y = msi_tl[1]
        # start_x = msi_tl[0]
        # img[start_y:start_y+len(maldi_scaled), start_x:start_x+len(maldi_scaled[0]), 0] = maldi_scaled / np.max(maldi_scaled)
        # img[start_y:start_y+len(maldi_scaled), start_x:start_x+len(maldi_scaled[0]), 1] = self.mic_canvas[start_y:start_y+len(maldi_scaled), start_x:start_x+len(maldi_scaled[0]),1]
        # img[:,:,2] = self.mic_canvas[:,:,2]
        # plt.figure()
        # plt.imshow(img)
        start = time.perf_counter()
        maldi_sitk = sitk.GetImageFromArray(maldi_img)
        mic_sitk = sitk.GetImageFromArray(self.mic_canvas[:, :, 1])
        mask_sitk = sitk.GetImageFromArray(self.mask_canvas)
        maldi_sitk = sitk.Cast(maldi_sitk, sitk.sitkFloat32)
        mic_sitk = sitk.Cast(mic_sitk, sitk.sitkFloat32)
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkFloat32)
        maldi_sitk.SetSpacing([self.ms_spacing, self.ms_spacing])
        mic_sitk.SetSpacing([self.mic_spacing, self.mic_spacing])
        mask_sitk.SetSpacing([self.mic_spacing, self.mic_spacing])
        scaling_factor = int(maldi_sitk.GetSpacing()[0] / mic_sitk.GetSpacing()[0])
        self.moving_spacing = maldi_sitk.GetSpacing()[0] / scaling_factor
        moving_size = (maldi_sitk.GetSize()[
                       0] * scaling_factor, maldi_sitk.GetSize()[1] * scaling_factor)
        moving_sitk = sitk.Resample(maldi_sitk, moving_size, interpolator=sitk.sitkNearestNeighbor, outputOrigin=(
            0.0, 0.0), outputSpacing=(self.moving_spacing, self.moving_spacing))

        mic_landmarks = [
            [self.msi_tl[0] * self.mic_spacing, self.msi_tl[1] * self.mic_spacing],
            [self.msi_tl[0] * self.mic_spacing, self.msi_br[1] * self.mic_spacing],
            [self.msi_br[0] * self.mic_spacing, self.msi_tl[1] * self.mic_spacing],
            [self.msi_br[0] * self.mic_spacing, self.msi_br[1] * self.mic_spacing],
        ]
        moving_landmarks = [
            [0.0, 0.0],
            [0.0, len(self.maldi_canvas) * self.ms_spacing],
            [len(self.maldi_canvas[0]) * self.ms_spacing, 0.0],
            [len(self.maldi_canvas[0]) * self.ms_spacing,
             len(self.maldi_canvas) * self.ms_spacing],
        ]

        center = np.array(self.maldi_canvas.shape[:2], dtype=float) / 2

        center = center * self.ms_spacing
        moving_pertubated_landmarks = pertubate_coordinates(
            moving_landmarks, center, self.rotation, self.scale, inverse=True)

        if self.teaching_imprecision is not None and self.teaching_imprecision > 0:
            for i in range(len(mic_landmarks)):
                for j in range(2):
                    shift = rng.uniform(-self.teaching_imprecision / 2,
                                        self.teaching_imprecision / 2)
                    mic_landmarks[i][j] += shift
                    shift = rng.uniform(0, self.teaching_imprecision)
                    moving_pertubated_landmarks[i][j] += shift

        mic_image_points_flat = [c for p in mic_landmarks for c in p]
        moving_image_points_flat = [c for p in moving_pertubated_landmarks for c in p]

        landmark_affine_transform = sitk.LandmarkBasedTransformInitializer(
            sitk.AffineTransform(2),
            mic_image_points_flat,
            moving_image_points_flat
        )
        self.interpret_registration_matrix(landmark_affine_transform)

        affine_transform = sitk.AffineTransform(mic_sitk.GetDimension())
        composite_transform = sitk.CompositeTransform(mic_sitk.GetDimension())
        composite_transform.AddTransform(landmark_affine_transform)
        composite_transform.AddTransform(affine_transform)
        if self.plot or self.save_img:
            landmark_registration_sitk = sitk.Resample(moving_sitk,
                                                       mic_sitk,
                                                       landmark_affine_transform,
                                                       sitk.sitkNearestNeighbor,
                                                       0.0,
                                                       moving_sitk.GetPixelID()
                                                       )
            composite_img = sitk.Compose([self.normalize_to_uint8(landmark_registration_sitk), self.normalize_to_uint8(
                landmark_registration_sitk), self.normalize_to_uint8(mic_sitk)])
            plt.figure(figsize=(6, 6))
            plt.imshow(sitk.GetArrayViewFromImage(composite_img))
            plt.axis('off')
            # plt.title("Initial Transform for Registration")
            if self.save_img:
                plt.savefig(self.output + "/initial_transform.svg",
                            bbox_inches='tight', pad_inches=0, dpi=600)

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
        #    numberOfIterations=100,
        #    maximumNumberOfCorrections=5,
        #    maximumNumberOfFunctionEvaluations=1000,
        #    costFunctionConvergenceFactor=1e+6
        # )
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=2000,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=50
        )
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(composite_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(
            0.1, 1 + int(self.seed) % (2**32 - 1))
        registration_method.SetNumberOfThreads(1)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        if self.use_perfect_registration:
            translation_transform = sitk.TranslationTransform(2)
            center_fixed = get_image_center(mic_sitk)
            center_moving = get_image_center(moving_sitk)
            self.moving_array = sitk.GetArrayFromImage(moving_sitk)
            translation = np.array(center_fixed) - np.array(center_moving) + \
                np.array((self.ms_spacing / 2, self.ms_spacing / 2))
            translation_transform.SetOffset(-translation)
            self.final_transform = sitk.CompositeTransform(2)
            self.pertubation_transform_moving = perturbation_transform(
                get_image_center(moving_sitk), self.rotation, self.scale)
            self.final_transform.AddTransform(
                self.pertubation_transform_moving.GetInverse())
            self.final_transform.AddTransform(translation_transform)
            self.registration_metrics['optimizer_iterations'] = 0
        else:
            self.final_transform = registration_method.Execute(mic_sitk, moving_sitk)
            # self.registration_metrics['optimizer_stopping_condition'] = registration_method.GetOptimizerStopConditionDescription()
            self.registration_metrics['optimizer_iterations'] = registration_method.GetOptimizerIteration(
            )

        if self.plot or self.save_img:
            test_sitk = sitk.Resample(mic_sitk,
                                      moving_sitk,
                                      self.final_transform.GetInverse(),
                                      sitk.sitkLinear,
                                      0.0,
                                      moving_sitk.GetPixelID()
                                      )
            composite_img = sitk.Compose([self.normalize_to_uint8(
                moving_sitk), self.normalize_to_uint8(moving_sitk), self.normalize_to_uint8(test_sitk)])
            self.registered_img = sitk.GetArrayFromImage(composite_img)
            plt.figure(figsize=(6, 6))
            # plt.title("Registration Overlay")
            plt.imshow(sitk.GetArrayViewFromImage(composite_img))
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/registration_overlay.svg",
                            bbox_inches='tight', pad_inches=0, dpi=600)

        self.interpret_registration_matrix(self.final_transform.GetNthTransform(0))

        self.evaluate_registration()

        # correct for pixel starting point in top left corner instead of pixel center
        adjusted_translation = (self.ms_spacing / 2, self.ms_spacing / 2)
        translation_transform = sitk.TranslationTransform(2)
        translation_transform.SetOffset(adjusted_translation)
        self.final_transform.AddTransform(translation_transform)

        mic_transformed = sitk.Resample(mic_sitk,
                                        moving_sitk,
                                        self.final_transform.GetInverse(),
                                        sitk.sitkLinear,
                                        0.0,
                                        moving_sitk.GetPixelID()
                                        )
        mask_transformed = sitk.Resample(mask_sitk,
                                         moving_sitk,
                                         self.final_transform.GetInverse(),
                                         sitk.sitkNearestNeighbor,
                                         0.0,
                                         moving_sitk.GetPixelID()
                                         )
        mic_cut = sitk.GetArrayFromImage(mic_transformed)
        mask_cut = sitk.GetArrayFromImage(mask_transformed)
        stop = time.perf_counter()
        self.registration_metrics['time'] = stop - start

        # print("Running Segmentation...")
        start = time.perf_counter()
        if self.use_perfect_segmentation:
            masks = mask_cut
        else:
            # TODO: add a table for cellpose settings in report files
            from cellpose import models
            model = models.Cellpose(gpu=False, model_type='cyto2')
            masks, flows, styles, diams = model.eval(
                mic_cut, diameter=self.cell_diameter)

        labels = set(np.unique(masks))
        masks_cleared = skimage.segmentation.clear_border(masks)
        reduced_labels = set(np.unique(masks_cleared))
        deleted_labels = labels.difference(reduced_labels)
        # print(f"Deleted labels: {deleted_labels}")
        mask_cut_cleared = skimage.segmentation.clear_border(mask_cut)
        predictions = self.label_image_to_masks(masks_cleared)
        ground_truths = self.label_image_to_masks(mask_cut_cleared)
        iou_all_masks, precision, recall, f1_score = self.evaluate_segmentation(
            predictions, ground_truths, iou_threshold=0.5)
        self.segmentation_metrics['mean IoU'] = (
            float(np.mean(list(iou_all_masks.values()))) if iou_all_masks else 0.0)
        self.segmentation_metrics['precision'] = precision
        self.segmentation_metrics['recall'] = recall
        self.segmentation_metrics['f1_score'] = f1_score

        if self.plot or self.save_img:
            overlay_img = np.zeros((len(masks), len(masks[0]), 3))
            overlay_img[:, :, 0] = mask_cut_cleared
            overlay_img[:, :, 1] = masks_cleared
            plt.figure(figsize=(6, 6))
            # plt.title("Segmentation")
            plt.imshow(overlay_img)
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/segmentation_results.svg",
                            bbox_inches='tight', pad_inches=0, dpi=600)
        stop = time.perf_counter()
        self.segmentation_metrics['time'] = stop - start

        # print("Calculating Single Cell Spectra...")

        start = time.perf_counter()
        ms_img = self.maldi_canvas[:, :, 0]
        ms_scale_factor = self.ms_spacing / self.moving_spacing

        mask_max = np.max(masks_cleared)
        mask_min = np.min(masks_cleared)
        num_cells = int(mask_max - mask_min - 1)
        cells = np.zeros((num_cells + 2, len(self.mass_list)))
        cell_areas = np.zeros((num_cells + 2, 1))
        k = 0  # counter for animation frames
        # iterate over each ms pixel and assign its intensity to the corresponding cell(s)
        for i in trange(len(ms_img)):
            for j in range(len(ms_img[0])):
                sub_region = masks_cleared[int(
                    i * ms_scale_factor):int((i + 1) * ms_scale_factor), int(j * ms_scale_factor):int((j + 1) * ms_scale_factor)]
                ids = set(sub_region[sub_region != 0].flatten().astype(np.int32))
                if len(ids) > 1:
                    if self.remove_ambiguous == False:
                        cell_area = np.count_nonzero(sub_region)
                        for cell_id in ids:
                            cell_areas[cell_id] += 1
                            area = np.count_nonzero(sub_region == cell_id) / cell_area
                            cells[cell_id, :] += self.maldi_canvas[i, j, :] * area
                if len(ids) == 1:
                    cell_id = next(iter(ids))
                    cells[cell_id, :] += self.maldi_canvas[i, j, :]
                    cell_areas[cell_id] += 1
                # render the process as animation frames
                if self.render_animation == True:
                    if i == 0 and j == 0:
                        import matplotlib as mpl
                        mpl.use('Agg')
                    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
                    ax[0].imshow(masks_cleared)
                    ax[1].imshow(self.maldi_canvas[:, :, 0], cmap='inferno',
                                 vmax=np.percentile(self.maldi_canvas[:, :, 0], 99.))
                    ax[1].add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0,
                                    edgecolor='red', facecolor='red', alpha=1))  # Render the whole pixel red
                    rect = patches.Rectangle((j * ms_scale_factor, i * ms_scale_factor), ms_scale_factor,
                                             ms_scale_factor, linewidth=1, edgecolor='red', facecolor='none', alpha=1)
                    ax[0].add_patch(rect)
                    text_cell_id = ax[1].text(
                        0, -0.1, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_intensity = ax[1].text(
                        0, -0.15, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_total_intensity = ax[1].text(
                        0, -0.2, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_total_cell_intensity = ax[1].text(
                        0, -0.25, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_coverage = ax[1].text(
                        0, -0.3, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    plt.tight_layout()
                    if len(ids) >= 1:
                        try:
                            text_cell_id.set_text(f"Cell ID: {cell_id}")
                            text_intensity.set_text(
                                f"Pixel Intensity: {self.maldi_canvas[i, j, 0]:.2f}")
                            text_total_intensity.set_text(
                                f"Total Intensity: {cells[cell_id, 0]:.2f}")
                            intensity = self.cell_intensities[cell_id][0]
                            text_total_cell_intensity.set_text(
                                f"Total Cell Intensity: {intensity:.2f}")
                            text_coverage.set_text(
                                f"GT deviation: {(intensity - cells[cell_id, 0]) * 100 / intensity:.2f}%")
                        except Exception as e:
                            continue
                    anim_dir = os.path.join(self.output, "animation")
                    os.makedirs(anim_dir, exist_ok=True)
                    plt.savefig(os.path.join(anim_dir, f"img_{k}.jpg"), dpi=300)
                    plt.close()
                k += 1

        canvas_to_mask = {}
        cells_missed = 0
        spectra_metrics_all_mz_mean = []
        spectra_metrics_all_mz_var = []
        cells_missed_all_mz = []
        deviation = np.zeros_like(cells)

        if self.plot or self.save_img:
            grid_size = int(np.ceil(np.sqrt(len(self.mass_list))))
            if grid_size > 5:
                grid_size = 5
            f_histograms, axes = plt.subplots(grid_size, grid_size, figsize=(6, 4), squeeze=False)
            axes = axes.flatten()

        self.measured_intensities = {}
        for key, _ in self.msi_coordinates.items():
            self.measured_intensities[key] = np.zeros(len(self.mass_list))

        for i in range(len(self.mass_list)):
            spectra_comparison = []
            for key, value in self.msi_coordinates.items():
                x_msi, y_msi = value
                x_global = x_msi * self.ms_spacing + self.msi_tl[0] * self.mic_spacing
                y_global = y_msi * self.ms_spacing + self.msi_tl[1] * self.mic_spacing
                x_pertubated, y_pertubated = self.final_transform.TransformPoint(
                    (x_global, y_global))
                x = x_pertubated / self.ms_spacing
                y = y_pertubated / self.ms_spacing
                mask_id = 0
                subregion = np.zeros(masks.shape, dtype=np.int64)
                subregion[mask_cut == key] = masks[mask_cut == key].astype(np.int64)
                counts = np.bincount(subregion[subregion != 0].flatten())
                if len(counts) >= 1:
                    mask_id = int(np.argmax(counts))
                    if mask_id in deleted_labels:
                        continue
                    canvas_to_mask[key] = mask_id
                    # print(f"at mz {mass_list[i]} cell {key} assigned to mask {mask_id}, true intensity: {self.cell_intensities[key][i]}, measured intensity: {cells[mask_id][i]}")
                    true_int = self.cell_intensities[key][i]
                    meas_int = cells[mask_id][i]
                    self.measured_intensities[key][i] = copy.copy(meas_int)

                    # if true_int != 0:
                    spectra_comparison.append(
                        (meas_int - true_int) / (true_int + meas_int))
                    # else:
                    #    deviation[mask_id][i] = meas_int
                    # print(f"cell: ({key},{mask_id}), x/y:({x}/{y}) ({x_msi}/{y_msi}), ground truth: {true_int}, measured: {meas_int}, difference: {true_int-meas_int}, relative difference: {(true_int-meas_int)/(true_int+meas_int):.2%}")
                else:
                    mask_id = 0
                    canvas_to_mask[key] = mask_id
                    true_int = self.cell_intensities[key][i]

                    meas_int = 0
                    self.measured_intensities[key][i] = copy.copy(meas_int)
                    cells_missed += 1
                    # if true_int != 0:
                    spectra_comparison.append(
                        (meas_int - true_int) / (true_int + meas_int))
                    # else:
                    #    deviation[mask_id][i] = meas_int
                    # print(f"cell: ({key},{mask_id}), x/y:({x}/{y}) ({x_msi}/{y_msi}), ground truth: {true_int}, measured: {meas_int}, difference: {true_int-meas_int}, relative difference: {(true_int-meas_int)/(true_int+meas_int):.2%}")

            spectra_comparison = np.array(spectra_comparison)
            spectra_comparison = spectra_comparison[~np.isnan(
                spectra_comparison) & ~np.isinf(spectra_comparison)]
            np.save(self.output + f"/spectra_comparison_{i}.npy", spectra_comparison)
            if self.plot or self.save_img:
                if i < 25:
                    axes[i].hist(spectra_comparison, bins=100, range=(-1, 1))
                    axes[i].set_title(f"m/z {self.mass_list[i]}")
                    axes[i].set_xlim([-1, 1])
            spectra_metrics_all_mz_mean.append(np.mean(np.abs(spectra_comparison)))
            spectra_metrics_all_mz_var.append(pow(np.std(spectra_comparison), 2))
            cells_missed_all_mz.append(np.sum(spectra_comparison == -1))
        if self.save_img:
            f_histograms.savefig(self.output + "/spectra_hist.svg",
                                 bbox_inches='tight', pad_inches=0, dpi=600)
        true_spectra = np.asarray(
            list(self.cell_intensities.values()), dtype=np.float32)
        measured_spectra = np.asarray(list(self.measured_intensities.values()))
        np.save(self.output + "/true_spectra.npy", true_spectra)
        np.save(self.output + "/measured_spectra.npy", measured_spectra)
        measured_spectra_sum = np.sum(measured_spectra, axis=1, keepdims=True)
        mask = measured_spectra_sum != 0
        mask = mask[:, 0]
        standardized_spectra = StandardScaler().fit_transform(
            measured_spectra[mask, :] / measured_spectra_sum[mask])
        np.save(self.output + "/standardized_spectra.npy", standardized_spectra)
        np.save(self.output + "/cell_labels.npy",
                np.array(list(self.cell_labels.values())))
        kmeans = KMeans(n_clusters=len(self.cell_types), random_state=self.seed)
        predicted_labels = kmeans.fit_predict(standardized_spectra)
        true_labels = np.array(list(self.cell_labels.values()))[mask]
        adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
        cont_matrix = contingency_matrix(true_labels, predicted_labels)
        self.cluster_metrics["contingency_matrix"] = cont_matrix
        self.cluster_metrics["adjusted_rand_index"] = adjusted_rand
        self.cluster_metrics["true_labels"] = true_labels
        self.cluster_metrics["predicted_labels"] = predicted_labels

        if (self.plot or self.save_img) and min(standardized_spectra.shape) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(standardized_spectra)
            unique_labels, true_labels_int = np.unique(true_labels, return_inverse=True)
            plt.figure(figsize=(6, 4))
            scatter = plt.scatter(
                pca_result[:, 0], pca_result[:, 1], c=true_labels_int, cmap='prism', alpha=0.7, linewidth=1.5)
            handles, _ = scatter.legend_elements()
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('PCA of Standardized Spectra')
            plt.legend(handles, unique_labels, title="True Label")
            if self.save_img:
                plt.savefig(self.output + "/pca.svg",
                            bbox_inches='tight', pad_inches=0, dpi=600)

        fpr = np.sum(deviation, axis=1) / (np.sum(cells, axis=1) +
                                           np.sum(deviation, axis=1)) * 100
        fpr = np.nan_to_num(fpr, 0)
        if self.plot or self.save_img:
            plt.figure(figsize=(6, 2))
            plt.hist(fpr, bins=100)
            plt.title(f"FPR")
            if self.save_img:
                plt.savefig(self.output + f"/fpr_hist.svg",
                            bbox_inches='tight', pad_inches=0, dpi=600)
        self.spectra_metrics["spectra_mean_deviation"] = spectra_metrics_all_mz_mean
        self.spectra_metrics["spectra_deviation_variance"] = spectra_metrics_all_mz_var
        self.spectra_metrics["missed_cells"] = cells_missed_all_mz
        self.spectra_metrics["FPR"] = np.mean(fpr)

        # print(f"Cells missed: {cells_missed}, Mean relative difference: {np.mean(spectra_comparison):.2%}")
        # print(f"FISCAS took {stop-start} s to run")
        # print("finish")

        if self.plot:
            plt.show()
        stop = time.perf_counter()
        self.spectra_metrics['time'] = stop - start

        with open(os.path.join(self.output, 'canvas_to_mask.pkl'), 'wb') as f:
            pickle.dump(canvas_to_mask, f)
        with open(os.path.join(self.output, 'cell_intensities.pkl'), 'wb') as f:
            pickle.dump(self.cell_intensities, f)
        with open(os.path.join(self.output, 'cells.pkl'), 'wb') as f:
            pickle.dump(cells, f)
        with open(os.path.join(self.output, 'cell_labels.pkl'), 'wb') as f:
            pickle.dump(self.cell_labels, f)

        return self.canvas_metrics, self.registration_metrics, self.segmentation_metrics, self.spectra_metrics, self.cluster_metrics


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate and evaluate synthetic MALDI data")
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON simulator configuration")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    settings["cell_types"] = [CellType(
        item["name"], {int(key) if key != "path" else key: value
                       for key, value in item["parameters"].items()}, item["population"])
        for item in settings["cell_types"]]
    Path(settings.get("output", "output")).mkdir(parents=True, exist_ok=True)
    import inspect

    evaluation_keys = set(inspect.signature(Validator).parameters) - {"canvas"}
    evaluation = {key: settings.pop(key)
                  for key in list(settings) if key in evaluation_keys}
    generation = CanvasSettings(**settings, **{
        key: evaluation[key] for key in ("output", "plot", "save_img") if key in evaluation})
    canvas = generate_and_plot_canvases(generation)
    Validator(canvas, **evaluation).main()
