# Structure of the Data Quality Analyzer (DQA) module:

# 1. Load Input Data
# 2. Run FISCAS on Input Data (fiscas.py) -> Single-Cell Spectra, Segmentation 
# 3. Estimate diffusion parameters based on MSI data and segmentation results (diffusion_estimation.py) -> Diffusion Parameters
# 4. Simulate MSI canvas based on segmentation results and diffusion parameters (canvas_generation.py) -> Simulated MSI Canvas
# 5. Compare similarity between simulated MSI canvas and original MSI data
# 6. Calculate single-cell spectra (fiscas.py) of simulated MSI canvas and calculate error to single-cell spectra of original MSI data


import os
from fiscas import FISCAS

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
input_dir = os.path.join(base_dir, "input")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)


seg, feature_list, cell_spectra = FISCAS.generate_single_cell_spectra(input_dir, output_dir)



