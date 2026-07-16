#print("importing modules...")
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from PIL import Image
import random
import skimage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix 
from sklearn.metrics import adjusted_rand_score
import cv2
from math import radians, cos, sin
import SimpleITK as sitk
from cellpose import models
from tqdm import trange
import copy
import time
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
import pickle 
from sklearn.decomposition import PCA
from canvas_generation import generate_and_plot_canvases
import uuid

#matplotlib.use('Agg')

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
plt.rc('figure', autolayout=True)        # automatically adjust subplot parameters to fit the figure


class Validator:
    def __init__(self,
        cell_types, # define cell populations with CellType objects,
        output="./output",
        use_perfect_registration=False, # skip registration by using the inverse of perturbation transform
        use_perfect_segmentation=True, # skip segmentation by using the ground truth masks
        teaching_imprecision=10,  # accuracy of landmark registration in µm
        canvas_size=2000,  # in µm
        mic_spacing=0.6897,  # pixel size of microscope canvas in µm 10X ~ 0.65; 20X ~ 0.325; 50X ~ 0.13
        ms_spacing=1,  # pixel size of msi in µm
        cell_density=0.2,  # percentage of total canvas area
        max_cell_placement_iterations=1000, # max attempts to place cells on canvas
        rotation=0,  # in degree
        scale=1.0,
        mass_list=[369.35, 760.58, 744.55, 958.78], # list of m/z values to simulate
        seed=0, # random seed for reproducibility
        plot=True, # plot canvasses
        save_img=True, # save canvasses in output directory
        remove_ambiguous=True, # remove pixel that touch more than one cell
        cell_diameter=0, # parameter for cell segmentation
        render_animation=False, # render animation of spectra compilation
        texture="registration", # apply texture to msi canvas
        texture_path="./textures/texture.jpg", 
        texture_epsilon_add=0, #strength of texture; -1 < epsilon < 1
        texture_epsilon_multiply=0, #strength of texture; -1 < epsilon < 1
        diffusion_total_time=100.0,
        diffusion_dt=1,
        diffusion_coeff=[0.1,1,50,10],
        diffusion_barrier_alpha=[0.1,0.3,0.7,0.5],
        diffusion_leaky_outer_boundary=True):  

        self.cell_types = cell_types
        self.output = output
        self.use_perfect_registration = use_perfect_registration
        self.use_perfect_segmentation = use_perfect_segmentation
        self.teaching_imprecision = teaching_imprecision
        self.canvas_size = canvas_size
        self.mic_spacing = mic_spacing
        self.ms_spacing = ms_spacing
        self.cell_density = cell_density
        self.max_cell_placement_iterations = max_cell_placement_iterations
        self.rotation = rotation
        self.scale = scale
        self.mass_list = mass_list
        self.seed = seed
        self.plot = plot
        self.save_img = save_img
        self.remove_ambiguous = remove_ambiguous
        self.cell_diameter = cell_diameter
        self.canvas_metrics = {}
        self.registration_metrics = {}
        self.segmentation_metrics = {}
        self.spectra_metrics = {}
        self.mic_coordinates = {}
        self.msi_coordinates = {}
        self.cell_intensities = {}
        self.cell_labels = {}
        self.cluster_metrics = {}
        self.render_animation = render_animation
        self.texture = texture
        self.texture_path = texture_path
        self.texture_epsilon_add = texture_epsilon_add
        self.texture_epsilon_multiply = texture_epsilon_multiply

        self.diffusion_total_time=diffusion_total_time
        self.diffusion_dt=diffusion_dt
        self.diffusion_coeff=diffusion_coeff
        self.diffusion_barrier_alpha=diffusion_barrier_alpha
        self.diffusion_leaky_outer_boundary=diffusion_leaky_outer_boundary
        plt.rcParams.update({'figure.max_open_warning': 0})

        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.plot or self.save_img:
            n = len(self.mass_list)
            c = int(np.ceil(np.sqrt(n)))
            fig, ax = plt.subplots(c, c, figsize=(6,6)) 
            k = 0
            colors = ["red", "black", "blue", "green", "orange", "purple", "brown", "pink"]
            for mz in mass_list:
                for c, cell_type in enumerate(self.cell_types):
                    try:
                        m = cell_type.parameters[mz][1]
                        s = cell_type.parameters[mz][2]
                        min_x = m - 4*s
                        min_x = min_x if min_x >= 0 else 0
                        max_x = m + 4*s
                        x = np.linspace(min_x, max_x, 100)
                        if cell_type.parameters[mz][0] == "normal":
                            y = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)
                        if cell_type.parameters[mz][0] == "uniform":
                            y = np.where((x >= m - s) & (x <= m + s), 1/(2*s), 0)
                        ax[k//2,k%2].plot(x, y, label=f"{cell_type.name} - " + "{m/z}" + f"{mz} (μ={m}, σ={s})", color=colors[c])
                        ax[k//2, k%2].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=1)
                    except Exception as e:
                        continue
                k += 1
            plt.tight_layout()
            if self.save_img:
                plt.savefig(self.output + "/cell_type_distributions.svg", bbox_inches='tight', pad_inches=0, dpi=600)
        return None

    # function to return dict with settings for report generation with quarto
    def return_settings(self):
        cell_types = []
        for i in self.cell_types:
            cell_types.append(i.name)
        settings = {
            "ID": uuid.uuid4(),
            "cell_types": cell_types,
            "output": self.output,
            "use_perfect_registration": self.use_perfect_registration,
            "use_perfect_segmentation": self.use_perfect_segmentation,
            "teaching_imprecision": self.teaching_imprecision,
            "canvas_size": self.canvas_size,
            "mic_spacing": self.mic_spacing,
            "ms_spacing": self.ms_spacing,
            "cell_density": self.cell_density,
            "max_cell_placement_iterations": self.max_cell_placement_iterations,
            "rotation": self.rotation,
            "scale": self.scale,
            "mass_list": self.mass_list,
            "seed": self.seed,
            "plot": self.plot,
            "save_img": self.save_img,
            "remove_ambiguous": self.remove_ambiguous,
            "cell_diameter": self.cell_diameter,
            "texture": self.texture,
            "texture_path": self.texture_path,
            "texture_epsilon_add": self.texture_epsilon_add,
            "texture_epsilon_multiply": self.texture_epsilon_multiply,
            "diffusion_total_time": self.diffusion_total_time,
            "diffusion_dt": self.diffusion_dt,
            "diffusion_coeff": self.diffusion_coeff,
            "diffusion_barrier_alpha": self.diffusion_barrier_alpha,
            "diffusion_leaky_outer_boundary": self.diffusion_leaky_outer_boundary  
        }
        return settings


    # loads random cell images and their corresponding masks from folder with prepared cell images and applies random rotation and scale
    # returns randomized cell image with its corresponding mask
    def load_random_cell(self, cell_directory, mask_index=255, seed=None):
        all_images = glob.glob(os.path.join(cell_directory, '**', '*.*'), recursive=True)
        images = sorted([
            file for file in all_images if 'cell' in os.path.basename(file) and 'seg' not in os.path.basename(file)])
        masks = sorted([
            file for file in all_images if 'seg' in os.path.basename(file)])
        if seed is None:
            angle = random.randint(0, 360)
            scaling_factor = random.uniform(0.8, 1.2)
            cell_chosen = random.randint(0, len(images)-1)
        else:
            angle = 0
            scaling_factor = 1
            cell_chosen = seed
        cell_mask = np.array(Image.open(masks[cell_chosen]), dtype=np.uint16) # open corresponding cell mask
        mask_ids = np.unique(cell_mask)
        for mask_id in mask_ids:
            if mask_id != 0:
                cell_mask[cell_mask == mask_id] = mask_index
                mask_index += 1
        mask_index -= 1
        current_mask_index = mask_index
        #with Image.open(images[cell_chosen]) as img: 
        #    cell_image = np.zeros((img.size[1], img.size[0], img.n_frames))
        #    for i in range(img.n_frames):
        #       img.seek(i)
        #       cell_image[:, :, i] = np.array(img)
        #       cell_image[:, :, i] = skimage.exposure.equalize_adapthist(np.array(img))
        cell_image = skimage.io.imread(images[cell_chosen])
        cell_rotated = skimage.transform.rotate(cell_image, angle, resize=True)
        mask_rotated = skimage.transform.rotate(np.array(cell_mask), angle, order=0, resize=True)
        cell_scaled = skimage.transform.rescale(cell_rotated, scaling_factor, anti_aliasing=True, channel_axis=-1)
        mask_scaled = skimage.transform.rescale(mask_rotated, scaling_factor, order=0, anti_aliasing=False)
        return cell_scaled, mask_scaled.astype(np.uint16), current_mask_index




    # generate MALDI signal distribution for a given cell mask
    def generate_maldi_cell(self, cell_type, mask, sigma=4.0):
        mask = copy.deepcopy(mask)
        
        intensities_all_mz = np.zeros((len(mask), len(mask[0]), len(self.mass_list)))
        mask = mask.astype(bool)
        dist_to_boundary = distance_transform_edt(mask)
        dist_to_boundary[~mask] = np.nan
        d_max = np.nanmax(dist_to_boundary)          # größter Abstand = „Mitte“ der Form
        gradient_linear = dist_to_boundary / d_max + 0.1
        intensities = np.nan_to_num(gradient_linear)
        cell = cell_type
        for i in range(len(self.mass_list)):
            factor = cell.get_intensity(self.mass_list[i])
            rescaled_intensities = intensities * factor
            intensities_all_mz[:,:, i] = rescaled_intensities
        return intensities_all_mz, np.sum(intensities_all_mz, axis=(0,1))

    # calculate the density of cells on the mask canvas
    def calculate_cell_density(self):
        total_area = self.mask_canvas.shape[0] * self.mask_canvas.shape[1]
        cell_area = np.count_nonzero(self.mask_canvas)
        density = cell_area / total_area
        return density

    # interpret registration results (2d-affine)
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

    # helper function for SITK
    def get_image_center(self, image):
        size = image.GetSize()
        spacing = image.GetSpacing()
        center = [(index) * spacing[dim] for dim, index in enumerate(np.array(size) / 2)]
        return center

    # perform pertubations on MSI canvas using SITK
    def pertubate_img(self, image, inverse=False):
        if not isinstance(image, sitk.Image):
            img = sitk.GetImageFromArray(image)
        else:
            img = image
        scale = 1 / self.scale
        rotation_angle_degrees = self.rotation
        rotation_angle_radians = np.deg2rad(rotation_angle_degrees)
        center = self.get_image_center(img)
        scaling_transform = sitk.ScaleTransform(2)
        scaling_transform.SetScale((scale, scale))
        scaling_transform.SetCenter(center)
        rotation_transform = sitk.Euler2DTransform()
        rotation_transform.SetCenter(center)
        rotation_transform.SetAngle(rotation_angle_radians)
        pertubation_transform = sitk.CompositeTransform(2)
        pertubation_transform.AddTransform(rotation_transform)
        pertubation_transform.AddTransform(scaling_transform)
        if inverse == True:
            transformed_image = sitk.Resample(img, img, pertubation_transform.GetInverse(), sitk.sitkNearestNeighbor, 0.0, img.GetPixelID())
        else:
            transformed_image = sitk.Resample(img, img, pertubation_transform, sitk.sitkNearestNeighbor, 0.0, img.GetPixelID())
        if not isinstance(image, sitk.Image):
            return pertubation_transform, sitk.GetArrayFromImage(transformed_image)
        else:
            return pertubation_transform, transformed_image

    # perform pertubations on cell coordinates to evaluate registration performance
    def pertubate_coordinates(self, coordinates, center, inverse=False):
        scale = 1 / self.scale
        rotation_angle_degrees = self.rotation
        rotation_angle_radians = np.deg2rad(rotation_angle_degrees)
        scaling_transform = sitk.ScaleTransform(2)
        scaling_transform.SetScale((scale, scale))
        scaling_transform.SetCenter(center)
        rotation_transform = sitk.Euler2DTransform()
        rotation_transform.SetCenter(center)
        rotation_transform.SetAngle(rotation_angle_radians)
        pertubation_transform = sitk.CompositeTransform(2)
        pertubation_transform.AddTransform(rotation_transform)
        pertubation_transform.AddTransform(scaling_transform)
        pertubated_coordinates = []
        if inverse == True:
            for point in coordinates:
                transformed_point = pertubation_transform.GetInverse().TransformPoint(point)
                pertubated_coordinates.append(transformed_point)
        else:
            for point in coordinates:
                transformed_point = pertubation_transform.TransformPoint(point)
                pertubated_coordinates.append(transformed_point)
        return np.array(pertubated_coordinates)

    # interpret registration results (2d-affine)
    def interpret_registration_matrix(self, transformations):
        if isinstance(transformations, sitk.CompositeTransform):
            for i in range(transformations.GetNumberOfTransforms()):
                sub_transform = transformations.GetNthTransform(i)
                if isinstance(sub_transform, sitk.Euler2DTransform):
                    matrix = np.reshape(sub_transform.GetMatrix(), (2,2))
                    self.registration_metrics['matrix'] = matrix
                    self.registration_metrics['translation'] = sub_transform.GetTranslation()
                    theta_rad, theta_deg = self.calculate_rotation_angle(matrix)
                    self.registration_metrics['theta_deg'] = theta_deg
                    self.registration_metrics['theta_rad'] = theta_rad
                elif isinstance(sub_transform, sitk.AffineTransform):
                    matrix = sub_transform.GetMatrix()
                    self.registration_metrics['matrix'] = matrix
                    self.registration_metrics['translation'] = sub_transform.GetTranslation()
                    rot, scale, shear = self.decompose_affine_2d(np.reshape(np.asarray(matrix), (2,2)))
                    theta_rad, theta_deg = self.calculate_rotation_angle(rot)
                    self.registration_metrics['rot'] = rot
                    self.registration_metrics['scale'] = scale
                    self.registration_metrics['shear'] = shear
                    self.registration_metrics['theta_deg'] = theta_deg
                    self.registration_metrics['theta_rad'] = theta_rad
                elif isinstance(sub_transform, sitk.TranslationTransform):
                    self.registration_metrics['translation'] = sub_transform.GetOffset()
                elif isinstance(sub_transform, sitk.Similarity2DTransform):
                    matrix = np.reshape(sub_transform.GetMatrix(), (2,2))
                    self.registration_metrics['matrix'] = matrix
                    self.registration_metrics['translation'] = sub_transform.GetTranslation()
                    self.registration_metrics['scale'] = sub_transform.GetScale()
                    theta_rad, theta_deg = self.calculate_rotation_angle(matrix)
                    self.registration_metrics['theta_deg'] = theta_deg
                    self.registration_metrics['theta_rad'] = theta_rad
        else:
            if isinstance(transformations, sitk.AffineTransform):
                self.registration_metrics['matrix'] = transformations.GetMatrix()
                self.registration_metrics['translation'] = transformations.GetTranslation()
            elif isinstance(transformations, sitk.TranslationTransform):
                self.registration_metrics['translation'] = transformations.GetOffset()
            elif isinstance(transformations, sitk.Similarity2DTransform):
                matrix = np.reshape(transformations.GetMatrix(), (2,2))
                self.registration_metrics['matrix'] = matrix
                theta_rad, theta_deg = self.calculate_rotation_angle(matrix)
                self.registration_metrics['translation'] = transformations.GetTranslation()
                self.registration_metrics['scale'] = transformations.GetScale()
                self.registration_metrics['theta_deg'] = theta_deg
                self.registration_metrics['theta_rad'] = theta_rad
        return None

    def rotate_coordinates(self, coords, angle_deg, center_point):
        angle_rad = radians(angle_deg)
        rotation_matrix = np.array([
            [cos(angle_rad), -sin(angle_rad)],
            [sin(angle_rad), cos(angle_rad)]
        ])
        rotated_coords = []
        for x, y in coords:
            translated_coords = np.array([x - center_point[0], y - center_point[1]])
            rotated_translated_coords = np.dot(rotation_matrix, translated_coords)
            new_x, new_y = rotated_translated_coords + center_point
            rotated_coords.append((new_x, new_y))
        return rotated_coords

    def scale_coordinates(self, coords, scale_factors, center_point):
        scaled_coords = []
        for x, y in coords:
            translated_coords = np.array([x - center_point[0], y - center_point[1]])
            scaled_translated_coords = translated_coords * np.array(scale_factors)
            new_x, new_y = scaled_translated_coords + center_point
            scaled_coords.append((new_x, new_y))
        return np.array(scaled_coords)

    def evaluate_registration(self):
        error = []
        for key, value in self.msi_coordinates.items():
            x, y = value
            x_global, y_global = x * self.ms_spacing, y * self.ms_spacing
            x_pertubated, y_pertubated = self.pertubation_transform_maldi.GetInverse().TransformPoint((x_global, y_global))
            landmark_moving = self.final_transform.GetInverse().TransformPoint((x_pertubated, y_pertubated))
            adjusted_translation = (-self.ms_spacing / 2, -self.ms_spacing / 2)
            #adjusted_translation = (0,0)
            translation_transform = sitk.TranslationTransform(2)
            translation_transform.SetOffset(adjusted_translation)
            landmark_moving = np.array(translation_transform.TransformPoint(landmark_moving))
            landmark_fixed = np.array(self.mic_coordinates[key]) * self.mic_spacing
            error.append(landmark_fixed - landmark_moving)
        error = np.array(error)
        if self.plot or self.save_img:
            fig, ax = plt.subplots(figsize=(6,6))
            radii = [1,3,5,7,10]
            for radius in radii:
                circle = plt.Circle((0, 0), radius, color='grey', fill=False, linestyle='-', linewidth=0.5)
                ax.add_artist(circle)
                ax.text(0, -radius, str(radius), fontsize=12, ha='center', va='center', color='grey',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            ax.set_aspect('equal', 'box')
            limit = max(radii) + 1
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            #ax.set_xlabel('Error X in µm')
            #ax.set_ylabel('Error Y in µm')
            #ax.set_title('Registration Error Plot')
            ax.scatter(0,0, color='grey', s=150, marker='+')
            ax.scatter(error[:,0], error[:,1], color='blue', s=70, marker='.', zorder=10)
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/Registration_error_plot.svg", bbox_inches='tight', pad_inches=0, dpi=600)
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
    def match_masks(self, predictions, ground_truths):
        num_pred = len(predictions)
        num_gt = len(ground_truths)
        iou_matrix = np.zeros((num_pred, num_gt))
        for i, pred_mask in enumerate(predictions):
            for j, gt_mask in enumerate(ground_truths):
                iou_matrix[i, j] = self.calculate_iou(pred_mask, gt_mask)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Negative for maximization
        matches = [(i, j) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] > 0.5]
        return matches, iou_matrix
 
    # convert label image to binary masks
    def label_image_to_masks(self, label_image):
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels != 0]
        masks = [(label_image == label_id) for label_id in unique_labels]
        return masks


    def add_texture(self, image, texture_path="./texture.jpg"):
        if np.abs(self.texture_epsilon_add) > 1:
            self.texture_epsilon_add /= np.abs(self.texture_epsilon_add)  # catch epsilon values outside of [-1, 1]
        image = np.array(image)
        texture = np.array(Image.open(texture_path))
        if len(texture.shape) == 3:
            texture = np.mean(texture, axis=2)  # Convert to grayscale if RGB
        texture = skimage.transform.resize(texture, (image.shape[1], image.shape[0]))
        texture = texture / np.max(texture) * self.texture_epsilon_add # Normalize texture and weight it by epsilon
        intensity = np.percentile(image, 99)
        texture = skimage.exposure.rescale_intensity(texture, in_range=(0,1), out_range=(0, intensity))
        combined_image = np.zeros_like(image)
        combined_image = texture + image
        return combined_image

    def multiply_texture(self, image, texture_path="./texture.jpg"):
        if np.abs(self.texture_epsilon_multiply) > 1:
            self.texture_epsilon_multiply /= np.abs(self.texture_epsilon_multiply)  # catch epsilon values outside of [-1, 1]
        image = np.array(image)
        texture = np.array(Image.open(texture_path))
        if len(texture.shape) == 3:
            texture = np.mean(texture, axis=2)  # Convert to grayscale if RGB
        texture = skimage.transform.resize(texture, (image.shape[1], image.shape[0]))
        texture = (texture / np.max(texture)).astype(float)
        texture += self.texture_epsilon_multiply
        texture = texture / np.max(texture)# Normalize texture and weight it by epsilon
        combined_image = np.zeros_like(image)
        combined_image = texture * image
        return combined_image

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

        matches, iou_matrix = self.match_masks(predictions, ground_truths)

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
        # create & plot canvases in separate module
        generate_and_plot_canvases(self)

        #print("perform pertubations...")
        self.maldi_canvas = np.zeros(self.ms_canvas.shape)
        maldi_sitk = sitk.GetImageFromArray(self.ms_canvas[:, :, 0])
        maldi_sitk.SetSpacing((self.ms_spacing, self.ms_spacing))

        for i in range(len(self.mass_list)):
            img = sitk.GetImageFromArray(self.ms_canvas[:, :, i])
            img.SetSpacing((self.ms_spacing, self.ms_spacing))
            self.pertubation_transform_maldi, pertubated_img = self.pertubate_img(img)
            self.maldi_canvas[:, :, i] = sitk.GetArrayFromImage(pertubated_img)
            if self.texture == 'all':
                self.maldi_canvas[:, :, i] = self.multiply_texture(
                    self.maldi_canvas[:, :, i],
                    self.texture_path
                )
                self.maldi_canvas[:, :, i] = self.add_texture(
                    self.maldi_canvas[:, :, i],
                    self.texture_path
                )

        maldi_img = self.maldi_canvas[:, :, 0]
        if self.texture == 'registration':
            maldi_img = self.multiply_texture(maldi_img, self.texture_path)
            maldi_img = self.add_texture(maldi_img, self.texture_path)

        if self.plot or self.save_img:
            fig = plt.figure(figsize=(6,6))
            plt.imshow(
                maldi_img,
                vmax=np.percentile(maldi_img, 99.),
                cmap='inferno'
            )
            plt.axis('off')
            if self.save_img:
                fig.savefig(
                    self.output + "/pertubated_msi_canvas.svg",
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=600
                )

        # canvas_metrics["time"] was already set in generate_and_plot_canvases
        # continue with registration / segmentation / spectra as before...


        #img = np.zeros((len(self.mic_canvas), len(self.mic_canvas[0]), 3))
        #maldi_scaled = skimage.transform.rescale(self.ms_canvas[:,:,0], ms_to_mic, order=0, anti_aliasing=False)
        #start_y = msi_tl[1]
        #start_x = msi_tl[0]
        #img[start_y:start_y+len(maldi_scaled), start_x:start_x+len(maldi_scaled[0]), 0] = maldi_scaled / np.max(maldi_scaled)
        #img[start_y:start_y+len(maldi_scaled), start_x:start_x+len(maldi_scaled[0]), 1] = self.mic_canvas[start_y:start_y+len(maldi_scaled), start_x:start_x+len(maldi_scaled[0]),1]
        #img[:,:,2] = self.mic_canvas[:,:,2]
        #plt.figure()
        #plt.imshow(img)
        start = time.perf_counter()
        maldi_sitk = sitk.GetImageFromArray(maldi_img)
        mic_sitk = sitk.GetImageFromArray(self.mic_canvas[:,:,1])
        mask_sitk = sitk.GetImageFromArray(self.mask_canvas)
        maldi_sitk = sitk.Cast(maldi_sitk, sitk.sitkFloat32)
        mic_sitk = sitk.Cast(mic_sitk, sitk.sitkFloat32)
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkFloat32)
        maldi_sitk.SetSpacing([self.ms_spacing, self.ms_spacing])
        mic_sitk.SetSpacing([self.mic_spacing, self.mic_spacing])
        mask_sitk.SetSpacing([self.mic_spacing, self.mic_spacing])
        scaling_factor = int(maldi_sitk.GetSpacing()[0] / mic_sitk.GetSpacing()[0])
        self.moving_spacing = maldi_sitk.GetSpacing()[0] / scaling_factor
        moving_size = (maldi_sitk.GetSize()[0] * scaling_factor, maldi_sitk.GetSize()[1] * scaling_factor)
        moving_sitk = sitk.Resample(maldi_sitk, moving_size, interpolator=sitk.sitkNearestNeighbor, outputOrigin=(0.0, 0.0), outputSpacing=(self.moving_spacing, self.moving_spacing))

        mic_landmarks = [
            [self.msi_tl[0] * self.mic_spacing, self.msi_tl[1] * self.mic_spacing],
            [self.msi_tl[0] * self.mic_spacing, self.msi_br[1] * self.mic_spacing],
            [self.msi_br[0] * self.mic_spacing, self.msi_tl[1] * self.mic_spacing],
            [self.msi_br[0] * self.mic_spacing, self.msi_br[1] * self.mic_spacing],
        ]
        moving_landmarks = [
            [0.0, 0.0],
            [0.0, len(self.ms_canvas) * self.ms_spacing],
            [len(self.ms_canvas[0]) * self.ms_spacing, 0.0],
            [len(self.ms_canvas[0]) * self.ms_spacing, len(self.ms_canvas) * self.ms_spacing],
        ]

        center = np.array(self.ms_canvas.shape[:2], dtype=float)/2

        center = center * self.ms_spacing
        moving_pertubated_landmarks = self.pertubate_coordinates(moving_landmarks, center=center, inverse=True)
  
        if self.teaching_imprecision > 0:
            for i in range(len(mic_landmarks)):
                for j in range(2):
                    shift = random.uniform(-self.teaching_imprecision/2, self.teaching_imprecision/2)
                    mic_landmarks[i][j] += shift
                    shift = random.uniform(0, self.teaching_imprecision)
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
            composite_img = sitk.Compose([self.normalize_to_uint8(landmark_registration_sitk), self.normalize_to_uint8(landmark_registration_sitk), self.normalize_to_uint8(mic_sitk)])
            plt.figure(figsize=(6,6))
            plt.imshow(sitk.GetArrayViewFromImage(composite_img))
            plt.axis('off')
            #plt.title("Initial Transform for Registration")
            if self.save_img:
                plt.savefig(self.output + "/initial_transform.svg", bbox_inches='tight', pad_inches=0, dpi=600)

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        #registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
        #    numberOfIterations=100,
        #    maximumNumberOfCorrections=5,
        #    maximumNumberOfFunctionEvaluations=1000,
        #    costFunctionConvergenceFactor=1e+6
        #)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate = 1.0,
            numberOfIterations=2000,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=50
        )
        #registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(composite_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        if self.use_perfect_registration:
            translation_transform = sitk.TranslationTransform(2)
            center_fixed = self.get_image_center(mic_sitk)
            center_moving = self.get_image_center(moving_sitk)
            self.moving_array = sitk.GetArrayFromImage(moving_sitk)
            translation = np.array(center_fixed) - np.array(center_moving) + np.array((self.ms_spacing/2, self.ms_spacing/2))
            translation_transform.SetOffset(-translation)
            self.final_transform = sitk.CompositeTransform(2)
            self.pertubation_transform_moving, _ = self.pertubate_img(moving_sitk)
            self.final_transform.AddTransform(self.pertubation_transform_moving.GetInverse())
            self.final_transform.AddTransform(translation_transform)
            self.registration_metrics['optimizer_iterations'] = 0
        else:
            self.final_transform = registration_method.Execute(mic_sitk, moving_sitk)
            #self.registration_metrics['optimizer_stopping_condition'] = registration_method.GetOptimizerStopConditionDescription()
            self.registration_metrics['optimizer_iterations'] = registration_method.GetOptimizerIteration()
       
        if self.plot or self.save_img:
            test_sitk = sitk.Resample(mic_sitk,
                                     moving_sitk,
                                     self.final_transform.GetInverse(),
                                     sitk.sitkLinear,
                                     0.0,
                                     moving_sitk.GetPixelID()
                                     )
            composite_img = sitk.Compose([self.normalize_to_uint8(moving_sitk), self.normalize_to_uint8(moving_sitk), self.normalize_to_uint8(test_sitk)])
            self.registered_img = sitk.GetArrayFromImage(composite_img)
            plt.figure(figsize=(6,6))
            #plt.title("Registration Overlay")
            plt.imshow(sitk.GetArrayViewFromImage(composite_img))
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/registration_overlay.svg", bbox_inches='tight', pad_inches=0, dpi=600)

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
        self.registration_metrics['time'] = stop-start


        #print("Running Segmentation...")
        start = time.perf_counter()
        if self.use_perfect_segmentation:
            masks = mask_cut
        else:
            #TODO: add a table for cellpose settings in report files 
            model = models.Cellpose(gpu=False, model_type='cyto2')
            masks, flows, styles, diams = model.eval(mic_cut, diameter=self.cell_diameter)

        labels = set(np.unique(masks))
        masks_cleared = skimage.segmentation.clear_border(masks)
        reduced_labels = set(np.unique(masks_cleared))
        deleted_labels = labels.difference(reduced_labels)
        #print(f"Deleted labels: {deleted_labels}")
        mask_cut_cleared = skimage.segmentation.clear_border(mask_cut)
        predictions = self.label_image_to_masks(masks_cleared)
        ground_truths = self.label_image_to_masks(mask_cut_cleared)
        iou_all_masks, precision, recall, f1_score = self.evaluate_segmentation(predictions, ground_truths, iou_threshold=0.5)
        self.segmentation_metrics['mean IoU'] = np.mean(list(iou_all_masks.values()))
        self.segmentation_metrics['precision'] = precision
        self.segmentation_metrics['recall'] = recall
        self.segmentation_metrics['f1_score'] = f1_score


        if self.plot or self.save_img:
            overlay_img = np.zeros((len(masks), len(masks[0]), 3))
            overlay_img[:,:,0] = mask_cut_cleared
            overlay_img[:,:,1] = masks_cleared
            plt.figure(figsize=(6,6))
            #plt.title("Segmentation")
            plt.imshow(overlay_img)
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/segmentation_results.svg", bbox_inches='tight', pad_inches=0, dpi=600)
        stop = time.perf_counter()
        self.segmentation_metrics['time'] = stop-start


        #print("Calculating Single Cell Spectra...")
        
        start = time.perf_counter()
        ms_img = self.maldi_canvas[:,:,0]
        ms_scale_factor = self.ms_spacing / self.moving_spacing

        mask_max = np.max(masks_cleared)
        mask_min = np.min(masks_cleared)
        num_cells = int(mask_max - mask_min -1)
        cells = np.zeros((num_cells + 2, len(self.mass_list)))
        cell_areas = np.zeros((num_cells + 2, 1))
        k=0 # counter for animation frames
        # iterate over each ms pixel and assign its intensity to the corresponding cell(s)
        for i in trange(len(ms_img)):
            for j in range(len(ms_img[0])):
                sub_region = masks_cleared[int(i*ms_scale_factor):int((i+1)*ms_scale_factor), int(j*ms_scale_factor):int((j+1)*ms_scale_factor)]
                ids = set(sub_region[sub_region!=0].flatten().astype(np.int32))
                if len(ids) > 1:
                    if self.remove_ambiguous == False:
                        cell_area = np.count_nonzero(sub_region)
                        for cell_id in ids:
                            cell_areas[cell_id] += 1
                            area = np.count_nonzero(sub_region == cell_id) / cell_area
                            cells[cell_id, :] += self.maldi_canvas[i,j,:] * area
                if len(ids) == 1:
                    cell_id = next(iter(ids))
                    cells[cell_id,:] += self.maldi_canvas[i,j,:]
                    cell_areas[cell_id] += 1
                # render the process as animation frames    
                if self.render_animation == True:
                    if i == 0 and j == 0:
                        import matplotlib as mpl
                        mpl.use('Agg')
                    fig, ax = plt.subplots(1,2, figsize=(6,6))
                    ax[0].imshow(masks_cleared)
                    ax[1].imshow(self.maldi_canvas[:,:,0], cmap='inferno', vmax=np.percentile(self.maldi_canvas[:,:,0], 99.))
                    ax[1].add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=0, edgecolor='red', facecolor='red', alpha=1))  # Render the whole pixel red
                    rect = patches.Rectangle((j*ms_scale_factor, i*ms_scale_factor), ms_scale_factor, ms_scale_factor, linewidth=1, edgecolor='red', facecolor='none', alpha=1)
                    ax[0].add_patch(rect)
                    text_cell_id = ax[1].text(0, -0.1, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_intensity = ax[1].text(0, -0.15, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_total_intensity = ax[1].text(0, -0.2, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_total_cell_intensity = ax[1].text(0, -0.25, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    text_coverage = ax[1].text(0, -0.3, '0', transform=ax[1].transAxes, color='black', fontsize=10, va='top', ha='left')
                    plt.tight_layout()
                    if len(ids) >=1:
                        try:
                            text_cell_id.set_text(f"Cell ID: {cell_id}")
                            text_intensity.set_text(f"Pixel Intensity: {self.maldi_canvas[i,j,0]:.2f}")
                            text_total_intensity.set_text(f"Total Intensity: {cells[cell_id,0]:.2f}")
                            intensity = self.cell_intensities[cell_id][0]
                            text_total_cell_intensity.set_text(f"Total Cell Intensity: {intensity:.2f}")
                            text_coverage.set_text(f"GT deviation: {(intensity-cells[cell_id,0])*100/intensity:.2f}%")
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
            f_histograms, axes = plt.subplots(grid_size, grid_size, figsize=(6,4))
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
                x_pertubated, y_pertubated = self.final_transform.TransformPoint((x_global, y_global))
                x = x_pertubated / self.ms_spacing
                y = y_pertubated / self.ms_spacing
                mask_id = 0
                subregion = np.zeros(masks.shape, dtype=np.int64)
                subregion[mask_cut==key] = masks[mask_cut==key].astype(np.int64)
                counts = np.bincount(subregion[subregion!=0].flatten())
                if len(counts) >= 1:
                    mask_id = int(np.argmax(counts))
                    if mask_id in deleted_labels:
                        continue
                    canvas_to_mask[key] = mask_id
                    #print(f"at mz {mass_list[i]} cell {key} assigned to mask {mask_id}, true intensity: {self.cell_intensities[key][i]}, measured intensity: {cells[mask_id][i]}")
                    true_int = self.cell_intensities[key][i]
                    meas_int = cells[mask_id][i]
                    self.measured_intensities[key][i] = copy.copy(meas_int)
                    
                    if true_int != 0:
                        spectra_comparison.append((meas_int-true_int)/true_int)
                    else:
                        deviation[mask_id][i] = meas_int
                    #print(f"cell: ({key},{mask_id}), x/y:({x}/{y}) ({x_msi}/{y_msi}), ground truth: {true_int}, measured: {meas_int}, difference: {true_int-meas_int}, relative difference: {(true_int-meas_int)/true_int:.2%}")
                else:
                    mask_id = 0
                    canvas_to_mask[key] = mask_id
                    true_int = self.cell_intensities[key][i]
            
                    meas_int = 0
                    self.measured_intensities[key][i] = copy.copy(meas_int)
                    cells_missed += 1
                    if true_int != 0:
                        spectra_comparison.append((meas_int-true_int)/true_int)
                    else:
                        deviation[mask_id][i] = meas_int
                    #print(f"cell: ({key},{mask_id}), x/y:({x}/{y}) ({x_msi}/{y_msi}), ground truth: {true_int}, measured: {meas_int}, difference: {true_int-meas_int}, relative difference: {(true_int-meas_int)/true_int:.2%}")
                
            
            spectra_comparison = np.array(spectra_comparison) * 100
            spectra_comparison = spectra_comparison[~np.isnan(spectra_comparison) & ~np.isinf(spectra_comparison)]
            if self.plot or self.save_img:
                if i < 25:
                    axes[i].hist(spectra_comparison, bins=100)
                    axes[i].set_title(f"m/z {self.mass_list[i]}")
            spectra_metrics_all_mz_mean.append(np.mean(np.abs(spectra_comparison[spectra_comparison!=-100])))
            spectra_metrics_all_mz_var.append(pow(np.std(spectra_comparison[spectra_comparison!=-100]),2))
            cells_missed_all_mz.append(np.sum(spectra_comparison==-100))
        if self.save_img:
            f_histograms.savefig(self.output + "/spectra_hist.svg", bbox_inches='tight', pad_inches=0, dpi=600)
        true_spectra = np.asarray(list(self.cell_intensities.values()), dtype=np.float32)
        measured_spectra = np.asarray(list(self.measured_intensities.values()))
        np.save(self.output + "/true_spectra.npy", true_spectra)
        np.save(self.output + "/measured_spectra.npy", measured_spectra)
        measured_spectra_sum = np.sum(measured_spectra, axis=1, keepdims=True)
        mask = measured_spectra_sum != 0
        mask = mask[:,0]
        standardized_spectra = StandardScaler().fit_transform(measured_spectra[mask, :]/measured_spectra_sum[mask])
        np.save(self.output + "/standardized_spectra.npy", standardized_spectra)
        np.save(self.output + "/cell_labels.npy", np.array(list(self.cell_labels.values())))
        kmeans = KMeans(n_clusters=len(self.cell_types), random_state=self.seed)
        predicted_labels = kmeans.fit_predict(standardized_spectra)
        true_labels = np.array(list(self.cell_labels.values()))[mask]
        adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
        cont_matrix = contingency_matrix(true_labels, predicted_labels)
        self.cluster_metrics["contingency_matrix"] = cont_matrix
        self.cluster_metrics["adjusted_rand_index"] = adjusted_rand
        self.cluster_metrics["true_labels"] = true_labels
        self.cluster_metrics["predicted_labels"] = predicted_labels


        if self.plot or self.save_img:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(standardized_spectra)
            unique_labels, true_labels_int = np.unique(true_labels, return_inverse=True)
            plt.figure(figsize=(6,4))
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=true_labels_int, cmap='prism', alpha=0.7, linewidth=1.5)
            handles, _ = scatter.legend_elements()
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('PCA of Standardized Spectra')
            plt.legend(handles, unique_labels, title="True Label")
            if self.save_img:
                plt.savefig(self.output + "/pca.svg", bbox_inches='tight', pad_inches=0, dpi=600)


        fpr = np.sum(deviation, axis=1) / (np.sum(cells, axis=1) + np.sum(deviation, axis=1)) * 100
        fpr = np.nan_to_num(fpr, 0)
        if self.plot or self.save_img:
            plt.figure(figsize=(6,2))
            plt.hist(fpr, bins=100)
            plt.title(f"FPR")
            if self.save_img:
                plt.savefig(self.output + f"/fpr_hist.svg", bbox_inches='tight', pad_inches=0, dpi=600)
        self.spectra_metrics["spectra_mean_deviation"] = spectra_metrics_all_mz_mean
        self.spectra_metrics["spectra_deviation_variance"] = spectra_metrics_all_mz_var
        self.spectra_metrics["missed_cells"] = cells_missed_all_mz
        self.spectra_metrics["FPR"] = np.mean(fpr)


        #print(f"Cells missed: {cells_missed}, Mean relative difference: {np.mean(spectra_comparison):.2%}")
        #print(f"FISCAS took {stop-start} s to run")
        #print("finish")

        if self.plot:
            plt.show()
        stop = time.perf_counter()
        self.spectra_metrics['time'] = stop-start
        

        with open(os.path.join(self.output, 'canvas_to_mask.pkl'), 'wb') as f:
            pickle.dump(canvas_to_mask, f)
        with open(os.path.join(self.output, 'cell_intensities.pkl'), 'wb') as f:
            pickle.dump(self.cell_intensities, f)
        with open(os.path.join(self.output, 'cells.pkl'), 'wb') as f:
            pickle.dump(cells, f)
        with open(os.path.join(self.output, 'cell_labels.pkl'), 'wb') as f:
            pickle.dump(self.cell_labels, f)
            
        
        return self.canvas_metrics, self.registration_metrics, self.segmentation_metrics, self.spectra_metrics, self.cluster_metrics


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
    
    
    def get_intensity(self, mz):
        if self.parameters[mz][0] == "normal":
            intensity = np.abs(np.random.normal(loc=self.parameters[mz][1], scale=self.parameters[mz][2]))
        elif self.parameters[mz][0] == "uniform":
            low = self.parameters[mz][1] - self.parameters[mz][2]
            high = self.parameters[mz][1] + self.parameters[mz][2]
            intensity = np.abs(np.random.uniform(low=low, high=high))
        return intensity


    def get_directory(self):
        return self.parameters["path"]

if __name__== "__main__":
    mass_list = [744.55,760.58,369.35,958.78]
    parameters_A = {"path" : "./Cell_Library/DLD-1",
                    mass_list[0] : ["normal", 15, 5], 
                    mass_list[1] : ["normal", 15, 3],
                    mass_list[2] : ["normal", 15, 3],
                    mass_list[3] : ["normal", 10, 3]
                    }
    parameters_B = {"path" : "./Cell_Library/Colo205",
                    mass_list[0] : ["normal", 15, 5], 
                    mass_list[1] : ["normal", 20, 5],
                    mass_list[2] : ["normal", 11, 3],
                    mass_list[3] : ["normal", 22, 15]
                    }

    A = CellType("A", parameters_A, 0.6)
    B = CellType("B", parameters_B, 0.4)

    valid = Validator(
        use_perfect_segmentation=True, 
        use_perfect_registration=False, 
        remove_ambiguous=False,
        seed=43,  
        scale=1.05, 
        teaching_imprecision=10, 
        rotation=5.0, 
        ms_spacing=5,
        mic_spacing=0.6897, 
        plot=True, 
        save_img=False, 
        canvas_size=1000, 
        cell_density=0.2, 
        mass_list=mass_list,
        cell_types = [A,B],
        render_animation=False,
        diffusion_total_time=0.0,
        diffusion_dt=1,
        diffusion_coeff=[0.1,1,50,10],
        diffusion_barrier_alpha=[0.1,0.3,0.7,0.5],
        diffusion_leaky_outer_boundary=True,
        texture='registration',
        texture_path="./textures/noise_gradient_150x150.jpg",
        texture_epsilon_add=0.13,
        texture_epsilon_multiply=0.3,)

    a, b, c, d, e = valid.main()
    _ = valid.return_settings()