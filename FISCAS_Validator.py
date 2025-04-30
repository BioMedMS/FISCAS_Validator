#print("importing modules...")
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import skimage
import cv2
from math import radians, cos, sin
import math
import SimpleITK as sitk
from cellpose import models
from tqdm import trange
import copy
import time
from scipy.optimize import linear_sum_assignment

###################################################### Settings ########################################################
#path_sample_cells = "D:/Jan/fiscas-validationKit/Cell_images/microscopy/"
#output = "D:/Jan/fiscas-validationKit/output/"
#use_perfect_registration =  True
#use_perfect_segmentation = False
#teaching_imprecision = 5 # in µm
#canvas_size = 2000 # in µm
#mic_spacing = 0.325 # in µm 10X ~ 0.65; 20X ~ 0.325; 50X ~ 0.13
#ms_spacing = 1 # in µm
#cell_density = 0.05 # percentage of total canvas area
#max_cell_placement_iterations = 1000
#rotation = -0 # in degree
#scale = 1.0
#mass_list = [577.5, 760.68]
#diffusion = 0 # in µm
#noise_loc = 0.0 # mean of noise level; default 0.1
#noise_scale = 0.0 # spread of noise level; default 0.2
#seed = 0
#plot = True
#save_img = False
#remove_ambiguous = True
#cell_diameter = 120 # for segmentation
########################################################################################################################

class Validator:
    def __init__(self,
                 path_sample_cells="D:/Jan/fiscas-validationKit/Cell_images/microscopy/",
                 output="D:/Jan/fiscas-validationKit/output/",
                 use_perfect_registration=False,
                 use_perfect_segmentation=False,
                 teaching_imprecision=5,  # in µm
                 canvas_size=2000,  # in µm
                 mic_spacing=0.325,  # in µm 10X ~ 0.65; 20X ~ 0.325; 50X ~ 0.13
                 ms_spacing=1,  # in µm
                 cell_density=0.05,  # percentage of total canvas area
                 max_cell_placement_iterations=1000,
                 rotation=0,  # in degree
                 scale=1.0,
                 mass_list=[577.5, 760.68],
                 diffusion=0,  # in µm
                 noise_loc=0.0,  # mean of noise level; default 0.1
                 noise_scale=0.0,  # spread of noise level; default 0.2
                 seed=0,
                 plot=True,
                 save_img=False,
                 remove_ambiguous=True,
                 cell_diameter=120):  # for segmentation
        self.path_sample_cells = path_sample_cells
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
        self.diffusion = diffusion
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale
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
        self.msi_canvas_coordinates = {}
        self.cell_intensities = {}

    def return_settings(self):
        settings = {
            "path_sample_cells": self.path_sample_cells,
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
            "diffusion": self.diffusion,
            "noise_loc": self.noise_loc,
            "noise_scale": self.noise_scale,
            "seed": self.seed,
            "plot": self.plot,
            "save_img": self.save_img,
            "remove_ambiguous": self.remove_ambiguous,
            "cell_diameter": self.cell_diameter
        }
        return settings

    # loads random cell images and their corresponding masks from folder with prepared cell images and applies random rotation and scale
    # returns randomized cell image with its corresponding mask
    def load_random_cell(self, mask_index=255, seed=None):
        if seed is None:
            cell_chosen = random.randint(0, len(self.cell_image_paths)-1)
            angle = random.randint(0, 360)
            scaling_factor = random.uniform(0.8, 1.2)
        else:
            cell_chosen = seed
            angle = 0
            scaling_factor = 1
        cell_mask = np.array(Image.open(self.cell_masks_paths[cell_chosen]), dtype=np.uint16)
        cell_mask[cell_mask > 0] = mask_index
        with Image.open(self.cell_image_paths[cell_chosen]) as img:
            cell_image = np.zeros((img.size[1], img.size[0], img.n_frames))
            for i in range(img.n_frames):
                img.seek(i)
                cell_image[:, :, i] = skimage.exposure.equalize_adapthist(np.array(img))
        cell_rotated = skimage.transform.rotate(cell_image, angle, resize=True)
        mask_rotated = skimage.transform.rotate(np.array(cell_mask), angle, order=0, resize=True)
        cell_scaled = skimage.transform.rescale(cell_rotated, scaling_factor, anti_aliasing=True)
        mask_scaled = skimage.transform.rescale(mask_rotated, scaling_factor, order=0, anti_aliasing=False)
        cell_scaled[mask_scaled == 0] = 0
        return cell_scaled, mask_scaled.astype(np.uint16)

    # apply diffusion on cell mask
    def diffuse_mask(self, mask, diffusion):
        mask[mask > 0] = 255
        r = int(self.diffusion / self.ms_spacing)
        if r < 1:
            return mask
        kernel = np.fromfunction(lambda x, y: ((x-r)**2 + (y-r)**2 <= r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)
        padding = r
        mask_padded = np.zeros((len(mask)+2*padding, len(mask[0])+2*padding))
        mask_padded[padding:padding+len(mask), padding:padding+len(mask[0])] = mask
        dil_sample = cv2.dilate(np.float32(mask_padded), kernel, iterations = 1)
        dil_sample = dil_sample.astype('uint8')
        dil_sample = dil_sample * 255
        return dil_sample


    def generate_maldi_cell(self, mask, sigma=4.0, diffusion=0):
        mask = copy.deepcopy(mask)
        if self.diffusion != 0:
            mask = self.diffuse_mask(mask, diffusion)
        mask_scaled = skimage.transform.rescale(mask, (self.mic_spacing / self.ms_spacing), order=1)
        mask_bin = np.zeros_like(mask_scaled)
        mask_bin[mask_scaled > 0.7 * np.max(mask_scaled)]=1
        coords = np.argwhere(mask_bin > 0)
        center_of_mass = coords.mean(axis=0)
        cx, cy = center_of_mass
        intensities = np.zeros_like(mask_bin, dtype=float)
        for x, y in coords:
            distance = np.sqrt((x-cx)**2 + (y-cy)**2)
            intensities[x, y] = np.exp(-distance**2 / (2 * sigma**2)) + 0.1
        return intensities, np.sum(intensities)


    def calculate_cell_density(self):
        total_area = self.mask_canvas.shape[0] * self.mask_canvas.shape[1]
        cell_area = np.count_nonzero(self.mask_canvas)
        density = cell_area / total_area
        return density


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

        # Schritt 1: Punkte als NumPy-Arrays umwandeln
        fixed_points = np.array(fixed_points)
        moving_points = np.array(moving_points)

        # Schritt 2: Zentriere die Punktsets
        fixed_centroid = np.mean(fixed_points, axis=0)
        moving_centroid = np.mean(moving_points, axis=0)

        fixed_centered = fixed_points - fixed_centroid
        moving_centered = moving_points - moving_centroid

        # Schritt 3: Skalierung und Rotation mit SVD berechnen
        U, _, VT = np.linalg.svd(np.dot(moving_centered.T, fixed_centered))
        R = np.dot(U, VT)

        scaleratio = np.sum(fixed_centered**2) / np.sum(moving_centered**2)
        S = np.sqrt(scaleratio)

        # Mittelwert der Skalierung anzeigen
        S = S / np.linalg.norm(R)  # Skalierung normalisieren

        # Schritt 4: Translation zurückführen
        translation = fixed_centroid - S * np.dot(moving_centroid, R)

        return R, S, translation


    def get_image_center(self, image):
        size = image.GetSize()
        spacing = image.GetSpacing()
        #origin = image.GetOrigin()
        center = [(index) * spacing[dim] for dim, index in enumerate(np.array(size) / 2)]
        return center


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
            landmark_moving = np.array(self.final_transform.GetInverse().TransformPoint((x_pertubated, y_pertubated)))
            landmark_fixed = np.array(self.mic_coordinates[key]) * self.mic_spacing
            error.append(landmark_fixed - landmark_moving)
        error = np.array(error)
        if self.plot or self.save_img:
            fig, ax = plt.subplots()
            radii = [1,3,5,7,10]
            for radius in radii:
                circle = plt.Circle((0, 0), radius, color='grey', fill=False, linestyle='-', linewidth=0.5)
                ax.add_artist(circle)
                ax.text(0, -radius, str(radius), fontsize=8, ha='center', va='center', color='grey',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            ax.set_aspect('equal', 'box')
            limit = max(radii) + 1
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            #ax.set_xlabel('Error X in µm')
            #ax.set_ylabel('Error Y in µm')
            #ax.set_title('Registration Error Plot')
            ax.scatter(0,0, color='grey', s=100, marker='+')
            ax.scatter(error[:,0], error[:,1], color='blue', s=30, marker='.', zorder=10)
            #ax.scatter(np.mean(error[:,0]), np.mean(error[:,1]), facecolor='none', color='red', s=50, marker='o')
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/Registration_error_plot.svg", bbox_inches='tight', pad_inches=0)
        #print(f"Mean error in µm: {np.mean(error, axis=0)} +/- {np.std(error, axis=0)}")
        self.registration_metrics["mre"] = np.mean(error, axis=0)
        self.registration_metrics["mre_std"] = np.std(error, axis=0)
        return None


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


    def label_image_to_masks(self, label_image):
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels != 0]
        masks = [(label_image == label_id) for label_id in unique_labels]
        return masks


    def evaluate_segmentation(self, predictions, ground_truths, iou_threshold=0.5):
        matches, iou_matrix = self.match_masks(predictions, ground_truths)
        iou_all_masks = {}
        for pred_index, gt_index in matches:
            iou_all_masks[gt_index] = iou_matrix[pred_index, gt_index]
        tp = len(matches) # tp = true positives
        matched_pred_indices = {pred_index for pred_index, _ in matches}
        fp = len(predictions) - len(matched_pred_indices) # fp = false positives
        matched_gt_indices = {gt_index for _, gt_index in matches}
        fn = len(ground_truths) - len(matched_gt_indices) # fn = false negatives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return iou_all_masks, precision, recall, f1_score

    def main(self):
        # print("populating canvas...")
        start = time.perf_counter()
        self.mic_canvas = np.zeros(
            (int(self.canvas_size / self.mic_spacing), int(self.canvas_size / self.mic_spacing), 3))
        self.mask_canvas = np.zeros(
            (int(self.canvas_size / self.mic_spacing), int(self.canvas_size / self.mic_spacing)))
        self.ms_canvas = np.zeros((int(0.5*self.canvas_size / self.ms_spacing), int(
            0.5*self.canvas_size / self.ms_spacing), len(self.mass_list)))

        all_images = glob.glob(os.path.join(
            self.path_sample_cells, '**', '*cell*'), recursive=True)
        self.cell_image_paths = [
            file for file in all_images if 'mask' not in os.path.basename(file)]
        self.cell_masks_paths = [
            file for file in all_images if 'mask' in os.path.basename(file)]

        if self.seed != 0:
            random.seed(self.seed)
        density = 0
        i = 0
        cell_id = 1
        while density < self.cell_density:
            if i > self.max_cell_placement_iterations:
                # print(f"aborted populating canvas. Too many cell placement cycles ({i})")
                break
            try:
                i += 1
                # choose random canvas coordinate to place the cell
                x = random.randint(0, len(self.mic_canvas[0]))
                y = random.randint(0, len(self.mic_canvas))
                cell, mask = self.load_random_cell(mask_index=cell_id)
                # prevent cell images to violate canvas border
                top_left_x = x - cell.shape[1] // 2
                top_left_y = y - cell.shape[0] // 2
                bottom_right_x = x + cell.shape[1] // 2
                bottom_right_y = y + cell.shape[0] // 2
                if (top_left_x > 0) & (top_left_y > 0) & (bottom_right_x < len(self.mic_canvas[0])) & (bottom_right_y < len(self.mic_canvas)):
                    canvas_area = self.mask_canvas[top_left_y:top_left_y+len(
                        mask), top_left_x:top_left_x+len(mask[0])]
                    if not np.any(np.logical_and(canvas_area, mask)):
                        # draw cell on mic_canvas; += prevents the background to override existing cells
                        self.mic_canvas[top_left_y:top_left_y+len(cell), top_left_x:(
                            top_left_x+len(cell[0])), 1] += cell[:, :, 0]
                        self.mic_canvas[top_left_y:top_left_y+len(
                            cell), top_left_x:top_left_x+len(cell[0]), 2] += cell[:, :, 1]
                        self.mask_canvas[top_left_y:top_left_y +
                                         len(mask), top_left_x:top_left_x+len(mask[0])] += mask
                        # store the mic x,y coordinates
                        self.mic_coordinates[cell_id] = (x, y)
                        density = self.calculate_cell_density()
                        # add msi data
                        if (x > 0.25 * len(self.mic_canvas[0])) & (x < 0.75 * len(self.mic_canvas[0])) & (y > 0.25 * len(self.mic_canvas)) & (y < 0.75 * len(self.mic_canvas)):
                            # conversion to global ms coordinate system
                            x_ms_coord = round(
                                x/(self.ms_spacing/self.mic_spacing))
                            y_ms_coord = round(
                                y/(self.ms_spacing/self.mic_spacing))
                            offset_x = round(
                                len(self.mic_canvas[0])*0.25*self.mic_spacing/self.ms_spacing)
                            offset_y = round(
                                len(self.mic_canvas)*0.25*self.mic_spacing/self.ms_spacing)
                            x_ms = x_ms_coord - offset_x  # conversion to local ms coordinate system
                            y_ms = y_ms_coord - offset_y
                            msi_cell, cell_intensity = self.generate_maldi_cell(
                                mask, sigma=20/self.ms_spacing, diffusion=self.diffusion)
                            top_left_x_ms = x_ms - msi_cell.shape[1] // 2
                            top_left_y_ms = y_ms - msi_cell.shape[0] // 2
                            bottom_right_x_ms = x_ms + msi_cell.shape[1] // 2
                            bottom_right_y_ms = y_ms + msi_cell.shape[0] // 2
                            if (top_left_x_ms > 0) & (top_left_y_ms > 0) & (bottom_right_x_ms < len(self.ms_canvas[0])) & (bottom_right_y_ms < len(self.ms_canvas)):
                                for j in range(len(self.mass_list)):
                                    # add cell intensity to ms canvas; += to add intensities of overlapping diffusion
                                    self.ms_canvas[top_left_y_ms:top_left_y_ms+len(
                                        msi_cell), top_left_x_ms:top_left_x_ms+len(msi_cell[0]), j] += msi_cell
                                # store local ms coordinates
                                self.msi_coordinates[cell_id] = (x_ms, y_ms)
                                self.msi_canvas_coordinates[cell_id] = (
                                    x_ms_coord, y_ms_coord)  # store global ms_coordinates
                                # store total msi cell intensity
                                self.cell_intensities[cell_id] = cell_intensity
                cell_id += 1
            except Exception:
                continue

        #print(f"A total number of {i} cells have been placed")

        if self.plot or self.save_img:
            f_mic_canvas = plt.figure()
            #plt.title("Microscopy Canvas")
            plt.imshow(self.mic_canvas[:,:])
            plt.axis('off')
            f_mask_canvas = plt.figure()
            #plt.title("Ground Truth Mask Canvas")
            plt.imshow(self.mask_canvas)
            plt.axis('off')
            f_msi_canvas = plt.figure()
            #plt.title("MSI Canvas")
            plt.imshow(self.ms_canvas[:,:,0])
            plt.axis('off')
            x_coords = []
            y_coords = []
            for key, item in self.msi_coordinates.items():
                x_coords.append(item[0])
                y_coords.append(item[1])
            plt.scatter(x_coords, y_coords, color='red', s=10, marker='x')
            if self.save_img:
                f_mic_canvas.savefig(self.output + "/mic_canvas.svg", bbox_inches='tight', pad_inches=0)
                f_mask_canvas.savefig(self.output + "/mask_canvas.svg", bbox_inches='tight', pad_inches=0)
                f_msi_canvas.savefig(self.output + "/ms_canvas.svg", bbox_inches='tight', pad_inches=0)
        self.canvas_metrics["n_cells"] = len(self.mic_coordinates)
        self.canvas_metrics["n_msi_cells"] = len(self.cell_intensities)

        #print("perform pertubations...")
        self.maldi_canvas = np.zeros(self.ms_canvas.shape)
        mask_canvas_sitk = sitk.GetImageFromArray(self.mask_canvas[int(0.25*len(self.mask_canvas)):int(
            0.75*len(self.mask_canvas)), int(0.25*len(self.mask_canvas[0])):int(0.75*len(self.mask_canvas[0]))])
        mask_canvas_sitk.SetSpacing((self.mic_spacing, self.mic_spacing))
        mask_canvas_sitk.SetOrigin((0.0, 0.0))
        mask_canvas_sitk = sitk.Cast(mask_canvas_sitk, sitk.sitkFloat32)
        mask_canvas_pertubated = self.pertubate_img(mask_canvas_sitk)[1]
        maldi_sitk = sitk.GetImageFromArray(self.ms_canvas[:, :, 0])
        maldi_sitk.SetSpacing((self.ms_spacing, self.ms_spacing))
        mask_canvas_pertubated = sitk.Resample(
            mask_canvas_pertubated,
            maldi_sitk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            mask_canvas_pertubated.GetPixelID()
        )
        mask_canvas_pertubated = sitk.GetArrayFromImage(mask_canvas_pertubated)
        plt.figure()
        plt.imshow(mask_canvas_pertubated)

        for i in range(len(self.mass_list)):
            img = sitk.GetImageFromArray(self.ms_canvas[:,:,i])
            img.SetSpacing((self.ms_spacing, self.ms_spacing))
            self.pertubation_transform_maldi, pertubated_img = self.pertubate_img(img)
            self.maldi_canvas[:,:,i] = sitk.GetArrayFromImage(pertubated_img)

        for mask_id in self.cell_intensities.keys():
            self.cell_intensities[mask_id] = np.sum(self.maldi_canvas[mask_canvas_pertubated == mask_id, 0])

        maldi_img = self.maldi_canvas[:,:,0]

        if self.plot or self.save_img:
            fig = plt.figure()
            plt.imshow(maldi_img)
            #plt.title("Pertubated MS Canvas")
            plt.axis('off')
            if self.save_img:
                fig.savefig(self.output + "/pertubated_ms_canvas.svg", bbox_inches='tight', pad_inches=0)
        stop = time.perf_counter()
        self.canvas_metrics['time'] = stop - start
        #print("perform registration...")
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

        mic_landmarks = [[0.25 * len(self.mic_canvas[0])*self.mic_spacing, 0.25 * len(self.mic_canvas)*self.mic_spacing],
                          [0.25 * len(self.mic_canvas[0])*self.mic_spacing, 0.75 * len(self.mic_canvas)*self.mic_spacing],
                          [0.75 * len(self.mic_canvas[0])*self.mic_spacing, 0.25 * len(self.mic_canvas)*self.mic_spacing],
                          [0.75 * len(self.mic_canvas[0])*self.mic_spacing, 0.75 * len(self.mic_canvas)*self.mic_spacing]]
        moving_landmarks = [[0.0, 0.0],
                             [0.0, len(self.ms_canvas)*self.ms_spacing],
                             [len(self.ms_canvas[0])*self.ms_spacing, 0.0],
                             [len(self.ms_canvas[0])*self.ms_spacing, len(self.ms_canvas)*self.ms_spacing]]
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
                                         sitk.sitkLinear,
                                         0.0,
                                         moving_sitk.GetPixelID()
                                         )
            composite_img = sitk.Compose([self.normalize_to_uint8(landmark_registration_sitk), self.normalize_to_uint8(landmark_registration_sitk), self.normalize_to_uint8(mic_sitk)])
            plt.figure()
            plt.imshow(sitk.GetArrayViewFromImage(composite_img))
            plt.axis('off')
            #plt.title("Initial Transform for Registration")
            if self.save_img:
                plt.savefig(self.output + "/initial_transform.svg", bbox_inches='tight', pad_inches=0)

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
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
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
            translation = np.array(center_fixed) - np.array(center_moving)
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
            test_sitk = sitk.Resample(moving_sitk,
                                     mic_sitk,
                                     self.final_transform,
                                     sitk.sitkLinear,
                                     0.0,
                                     moving_sitk.GetPixelID()
                                     )
            self.moving_transformed_arr = sitk.GetArrayFromImage(sitk.Resample(moving_sitk,
                                     moving_sitk,
                                     self.final_transform,
                                     sitk.sitkLinear,
                                     0.0,
                                     moving_sitk.GetPixelID()
                                     ))
            composite_img = sitk.Compose([self.normalize_to_uint8(test_sitk), self.normalize_to_uint8(test_sitk), self.normalize_to_uint8(mic_sitk)])
            self.registered_img = sitk.GetArrayFromImage(composite_img)
            plt.figure()
            #plt.title("Registration Overlay")
            plt.imshow(sitk.GetArrayViewFromImage(composite_img))
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/registration_overlay.svg", bbox_inches='tight', pad_inches=0)


        self.interpret_registration_matrix(self.final_transform.GetNthTransform(0))

        self.evaluate_registration()

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
            model = models.Cellpose(gpu=False, model_type='cyto2')
            masks, flows, styles, diams = model.eval(mic_cut, diameter=self.cell_diameter)

        plt.figure()
        plt.imshow(masks)
        plt.figure()
        plt.imshow(mask_cut)
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
            plt.figure()
            #plt.title("Segmentation")
            plt.imshow(overlay_img)
            plt.axis('off')
            if self.save_img:
                plt.savefig(self.output + "/segmentation_results.svg", bbox_inches='tight', pad_inches=0)
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
        cell_areas = np.zeros((num_cells + 2, 1))        #print(cells.shape)
        for i in trange(len(ms_img)):
            for j in range(len(ms_img[0])):
                sub_region = masks_cleared[int(i*ms_scale_factor):int((i+1)*ms_scale_factor), int(j*ms_scale_factor):int((j+1)*ms_scale_factor)]
                ids = set(sub_region[sub_region!=0].flatten().astype(np.int))
                if len(ids) > 1:
                    if self.remove_ambiguous == False:
                        cell_area = np.count_nonzero(sub_region)
                        for cell_id in ids:
                            cell_areas[cell_id] += 1
                            area = np.count_nonzero(sub_region == cell_id) / cell_area#np.count_nonzero(sub_region)
                            cells[cell_id, :] += self.maldi_canvas[i,j,:] * area
                if len(ids) == 1:
                    cell_id = next(iter(ids))
                    cells[cell_id,:] += self.maldi_canvas[i,j,:]
                    cell_areas[cell_id] += 1


        canvas_to_mask = {}
        spectra_comparison = []
        cells_missed = 0
        
        for key, value in self.msi_coordinates.items():
            x_msi, y_msi = value
            x_global, y_global = x_msi * self.ms_spacing + 0.25 * len(self.mic_canvas[0]) * self.mic_spacing, y_msi * self.ms_spacing + 0.25 * len(self.mic_canvas) * self.mic_spacing
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
                true_int = self.cell_intensities[key]
                meas_int = cells[mask_id][0]
                spectra_comparison.append((true_int-meas_int)/true_int)
                #print(f"cell: ({key},{mask_id}), x/y:({x}/{y}) ({x_msi}/{y_msi}), ground truth: {true_int}, measured: {meas_int}, difference: {true_int-meas_int}, relative difference: {(true_int-meas_int)/true_int:.2%}")
            else:
                mask_id = 0
                canvas_to_mask[key] = mask_id
                true_int = self.cell_intensities[key]
                meas_int = 0
                cells_missed += 1
                spectra_comparison.append((true_int-meas_int)/true_int)
                #print(f"cell: ({key},{mask_id}), x/y:({x}/{y}) ({x_msi}/{y_msi}), ground truth: {true_int}, measured: {meas_int}, difference: {true_int-meas_int}, relative difference: {(true_int-meas_int)/true_int:.2%}")
            
        
        spectra_comparison = np.array(spectra_comparison) * 100
        spectra_comparison = spectra_comparison[~np.isnan(spectra_comparison)]
        if self.plot or self.save_img:
            plt.figure()
            plt.hist(spectra_comparison, bins=100)
            if self.save_img:
                plt.savefig(self.output + "/spectra_hist.svg", bbox_inches='tight', pad_inches=0)
        self.spectra_metrics["spectra_mean_precision"] = np.mean(spectra_comparison[spectra_comparison!=100])
        self.spectra_metrics["missed_cells"] = np.sum(spectra_comparison==100)

        #print(f"Cells missed: {cells_missed}, Mean relative difference: {np.mean(spectra_comparison):.2%}")
        #print(f"FISCAS took {stop-start} s to run")
        #print("finish")

        if self.plot:
            plt.show()
        stop = time.perf_counter()
        self.spectra_metrics['time'] = stop-start

        return self.canvas_metrics, self.registration_metrics, self.segmentation_metrics, self.spectra_metrics

A = Validator(use_perfect_segmentation=True, use_perfect_registration=False, seed=42, scale=1.01, teaching_imprecision=10, rotation=1, ms_spacing=1, plot=True, save_img=False, canvas_size=1500, cell_density=0.3)
A.main()
