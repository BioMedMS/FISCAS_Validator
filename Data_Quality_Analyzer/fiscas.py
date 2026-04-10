import numpy as np
import imzy
import csv 
import os
import skimage
from coordinate_picker import pick_coordinates

class FISCAS:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def load_msi(self, imzml_file, peaklist_file, tolerance=0.05):
            ## imzy import
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
        self.img_all_int = np.transpose(reader.get_ion_images(peaklist[:, 0], tol=tolerance), (1,2,0))
        self.mass_list = peaklist[:, 0]
        return None

    def register_data(self):
        return None
    
    def segment_cells(self):
        return None
    
    def compile_cell_spectra(self):
        return None

    def generate_single_cell_spectra(self, input_dir, output_dir):
        msi_file = os.path.join(input_dir, "msi_data_1.imzML")
        msi_npy = os.path.join(input_dir, "msi_data_1.npy")
        feature_list_npy = os.path.join(input_dir, "feature_list.npy")
        mic_file = os.path.join(input_dir, "mic_data_1.tif")
        feature_list_file = os.path.join(input_dir, "feature_list_1.csv")

        # Load Data
        if not os.path.exists(msi_npy):
            feature_list, msi_data = self.load_msi(msi_file, feature_list_file)
            np.save(msi_npy, msi_data)
            np.save(feature_list_npy, feature_list)
        else:
            print("loading preprocessed msi data from .npy file")
            msi_data = np.load(msi_npy)
            feature_list = np.load(feature_list_npy)
        mic_data = skimage.io.imread(mic_file)

        # Extract 2D images for coordinate picking
        if mic_data.ndim == 3 and mic_data.shape[0] > 1:
            mic_img = mic_data[0, :, :]
        else:
            mic_img = mic_data

        peak = 760.58
        peak_idx = np.argmin(np.abs(feature_list - peak))
        ms_img = msi_data[:,:,peak_idx]

        # Pick coordinates for landmark registration
        coordinates = pick_coordinates(ms_img, mic_img)
        print(coordinates)

        cut_img = self.register_data(ms_img, mic_img, coordinates)
        seg = self.segment_cells(cut_img)
        cell_spectra = self.compile_cell_spectra(seg, msi_data)


        return seg, feature_list, cell_spectra