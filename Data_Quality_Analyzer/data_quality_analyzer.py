# Structure of the Data Quality Analyzer (DQA) module:

# 1. Load Input Data
# 2. Run FISCAS on Input Data (fiscas.py) -> Single-Cell Spectra, Segmentation 
# 3. Estimate diffusion parameters based on MSI data and segmentation results (diffusion_estimation.py) -> Diffusion Parameters
# 4. Simulate MSI canvas based on segmentation results and diffusion parameters (canvas_generation.py) -> Simulated MSI Canvas
# 5. Compare similarity between simulated MSI canvas and original MSI data
# 6. Calculate single-cell spectra (fiscas.py) of simulated MSI canvas and calculate error to single-cell spectra of original MSI data

from fiscas import load_imzml
import skimage
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
input_dir = os.path.join(base_dir, "input")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

msi_file = os.path.join(input_dir, "msi_data_1.imzML")
mic_file = os.path.join(input_dir, "mic_data_1.tif")
feature_list_file = os.path.join(input_dir, "feature_list_1.csv")

msi_data, feature_list = load_imzml(msi_file, feature_list_file)
mic_data = skimage.io.imread(mic_file)
