import numpy as np
import imzy
import csv 
import os
import skimage
from coordinate_picker import pick_coordinates
import SimpleITK as sitk
import matplotlib.pyplot as plt
from cellpose import models
from tqdm import trange
import tifffile
import xml.etree.ElementTree as ET

class FISCAS:
    def __init__(self, input_dir, output_dir, debug=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ms_spacing = None
        self.mic_spacing = None
        self.scaling_factor = None
        self.debug = debug

    def load_msi(self, imzml_file, peaklist_file, tolerance=0.05):
        print("starting import")
        try:
            with open(peaklist_file) as data:
                csv_reader = csv.reader(data, delimiter=';')
                line = 0
                for row in csv_reader:
                    line += 1
                length_peaklist = line-9
                
            peaklist = np.zeros((length_peaklist, 2))
            with open(peaklist_file) as data:
                csv_reader = csv.reader(data, delimiter=';')
                line = 0
                for row in csv_reader:
                    if line > 8:
                        peaklist[line-9,0] = row[0]
                        peaklist[line-9,1] = row[1]
                    line += 1
        except Exception as e:
            print(e)
            print("Please provide the path to a valid peaklist.")
        reader = imzy.get_reader(imzml_file)
        self.ms_spacing = reader._imzml_cache.PX_SIZE_X
        self.msi_data = np.transpose(reader.get_ion_images(peaklist[:, 0], tol=tolerance), (1,2,0))
        self.feature_list = peaklist[:, 0]
        return None

    def register_data(self, ms_img, mic_img):

        # Define simple ITK images
        fixed_image = sitk.GetImageFromArray(mic_img.astype(np.float32))
        ms_image = sitk.GetImageFromArray(ms_img.astype(np.float32))
        fixed_image.SetSpacing([self.mic_spacing, self.mic_spacing])
        ms_image.SetSpacing([self.ms_spacing, self.ms_spacing])

        # scaling factor ensures that the microscopy data can be mapped to 
        # the msi data in integer steps
        self.scaling_factor = int(self.ms_spacing / self.mic_spacing)

        # moving_image is a surrogate for the registration to map the microscopy
        # image to the nearest whole number integer multiple of the msi image.
        moving_shape = (ms_image.GetSize()[0] * self.scaling_factor, 
                        ms_image.GetSize()[1] * self.scaling_factor)
        moving_spacing = self.ms_spacing / self.scaling_factor
        moving_image = sitk.Resample(ms_image,
                                     moving_shape, 
                                     interpolator=sitk.sitkNearestNeighbor,
                                     outputOrigin=(0,0),
                                     outputSpacing=(moving_spacing, moving_spacing))
        

        # Collect landmarks
        if self.debug:
            translation = (-627.1745198306462, -437.6500325458628)
        else:
            print("Please select corresponding landmarks in the MSI and microscopy images. Click on the same feature in both images. When finished, close the window.")
            coordinates = pick_coordinates(ms_img, mic_img)
            print(f"Collected coordinates: {coordinates}")
            ms_image_points = coordinates['MSI_Data']
            fixed_image_points = coordinates['Microscopy_Data']
            ms_image_points = [
                ms_image.TransformContinuousIndexToPhysicalPoint(pnt)
                for pnt in ms_image_points
            ]
            fixed_image_points = [
                fixed_image.TransformContinuousIndexToPhysicalPoint(pnt)
                for pnt in fixed_image_points
            ]
            diff = np.mean(np.asarray(fixed_image_points) - np.asarray(ms_image_points), axis=0)
            translation = (-diff[0], -diff[1])
            print(f"Calculated translation from landmarks: {translation}")


        # Perform landmark-based registration
        landmark_transform = sitk.TranslationTransform(2)
        landmark_transform.SetOffset(translation)
        
        # Use landmark registration as initial transform for a more robust registration
        composite_transform = sitk.CompositeTransform(2)
        composite_transform.AddTransform(landmark_transform)
        composite_transform.AddTransform(sitk.AffineTransform(2))
        
        # Define registration parameters
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=50)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=1000,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=20
        )
        registration_method.SetInitialTransform(
            composite_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetMetricSamplingStrategy(
            registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Execute registration
        print("Starting registration...")
        final_transform = registration_method.Execute(fixed_image, moving_image)
        print("Completed registration.")
        # Cut registered microscopy image to size of moving image
        fixed_to_moving = sitk.Resample(
                fixed_image,
                moving_image,
                final_transform.GetInverse(),
                sitk.sitkLinear,
                0.0,
                fixed_image.GetPixelID()
            )
        output_image = sitk.GetArrayFromImage(fixed_to_moving)

        # show registered image for visual inspection
        moving_arr = sitk.GetArrayFromImage(moving_image)
        cut_arr = sitk.GetArrayFromImage(fixed_to_moving)
        img = np.zeros([moving_arr.shape[0], moving_arr.shape[1], 3])
        img[:,:,0] = moving_arr / np.max(moving_arr)
        img[:,:,1] = cut_arr / np.max(cut_arr)

        return cut_arr
    
    def segment_cells(self, img):
        model = models.CellposeModel(gpu=False)
        print("Starting cell segmentation...")
        masks, _, _ = model.eval(
                img,
                diameter=120
            )
        print("Completed cell segmentation.")
        return masks 
    
    def compile_cell_spectra(self, remove_ambiguous=False):
        num_cells = np.max(self.seg + 1)
        cells = np.zeros((num_cells, len(self.feature_list)))

        print("Compiling single-cell spectra...")
        for i in trange(self.msi_data.shape[0]):
            for j in range(self.msi_data.shape[1]):
                sub_region = self.seg[
                    int(i*self.scaling_factor):int((i+1)*self.scaling_factor), 
                    int(j*self.scaling_factor):int((j+1)*self.scaling_factor)]
                ids = set(sub_region[sub_region!=0].flatten().astype(np.int32))
                if len(ids) > 1:
                    if remove_ambiguous == False:
                        cell_area = np.count_nonzero(sub_region)
                        for cell_id in ids:
                            area = np.count_nonzero(sub_region == cell_id) / cell_area
                            cells[cell_id, :] += self.msi_data[i,j,:] * area
                    else:
                        continue
                if len(ids) == 1:
                    cell_id = next(iter(ids))
                    cells[cell_id,:] += self.msi_data[i,j,:]
        
        print("Completed compiling single-cell spectra.")
        return cells

    def generate_single_cell_spectra(self):
        msi_file = os.path.join(self.input_dir, "msi_data_1.imzML")
        mic_file = os.path.join(self.input_dir, "mic_data_1.tif")
        feature_list_file = os.path.join(self.input_dir, "feature_list_1.csv")
        msi_npy = os.path.join(self.input_dir, "msi_data_1.npy")
        feature_list_npy = os.path.join(self.input_dir, "feature_list.npy")
        seg_npy = os.path.join(self.input_dir, "seg.npy")
        path_cut_img = os.path.join(self.output_dir, "registered_microscopy.tif")


        # Load Data
        if not os.path.exists(msi_npy):
            self.load_msi(msi_file, feature_list_file)
            np.save(msi_npy, self.msi_data)
            np.save(feature_list_npy, self.feature_list)
        else:
            print("loading preprocessed msi data from .npy file")
            self.msi_data = np.load(msi_npy)
            self.feature_list = np.load(feature_list_npy)
        mic_data = skimage.io.imread(mic_file)
        if mic_file.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(mic_file) as tif:
                # Access OME metadata if available
                ome_metadata = tif.ome_metadata
                if ome_metadata:
                    # Parse the OME XML metadata
                    try:
                        root = ET.fromstring(ome_metadata)
                        namespaces = {
                            'ome': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'}
                        # Extract physical size information from the XML
                        size_x = root.find(
                            './/ome:Pixels', namespaces).get('PhysicalSizeX')
                        # size_y = root.find('.//ome:Pixels', namespaces).get('PhysicalSizeY')
                        self.mic_spacing = float(size_x)
                    except Exception as e:
                        print(f"Error parsing OME metadata: {e}")


        # Extract 2D images for coordinate picking
        if mic_data.ndim == 3 and mic_data.shape[0] > 1:
            mic_img = mic_data[0, :, :]
        else:
            mic_img = mic_data

        peak = 760.58
        peak_idx = np.argmin(np.abs(self.feature_list - peak))
        ms_img = self.msi_data[:,:,peak_idx]

        # Collect image spacings
        if self.mic_spacing is None: 
            if self.debug:
                self.mic_spacing = 0.3448648441825844
            else:
                self.mic_spacing = float(input("Please enter the pixel size of the microscopy image in microns (e.g. 0.5): "))
        if self.ms_spacing is None:
            if self.debug:
                self.ms_spacing = 8
            else:
                self.ms_spacing = int(input("Please enter the pixel size of the mass spectrometry image in microns (e.g. 5): "))

        if not os.path.exists(path_cut_img):
            cut_img = self.register_data(ms_img, mic_img)
            skimage.io.imsave(os.path.join(self.output_dir, "registered_microscopy.tif"), cut_img.astype(np.float32))
        else:
            print("loading registered microscopy image from file")
            cut_img = skimage.io.imread(path_cut_img)
            self.scaling_factor = int(self.ms_spacing / self.mic_spacing)

        if not os.path.exists(seg_npy):
            self.seg = self.segment_cells(cut_img)
            np.save(seg_npy, self.seg)
        else:
            print("loading preprocessed segmentation from .npy file")
            self.seg = np.load(seg_npy, allow_pickle=True)

        
        self.cell_spectra = self.compile_cell_spectra(remove_ambiguous=False)


        return self.seg, self.feature_list, self.cell_spectra

    
