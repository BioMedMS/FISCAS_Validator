import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from FISCAS_Validator import Validator
from canvas_generation import CanvasSettings, CellType, generate_and_plot_canvases
import numpy as np
import copy
import pandas as pd
import glob
import re
import warnings
warnings.filterwarnings("ignore")


def run_validator(configuration):
    generation, evaluation = configuration
    print(f"[{datetime.datetime.now().time()}] Starting run {generation.seed}")

    try:
        A = Validator(generate_and_plot_canvases(generation), **evaluation)
        canvas_metrics, registration_metrics, segmentation_metrics, spectra_metrics, cluster_metrics = A.main()
    except Exception as e:
        print(f"[{datetime.datetime.now().time()}] Error in run {generation.seed}: {e}")
        return generation.seed, 1

    settings = A.return_settings()
    seed = A.seed
    setting_columns = list(settings.keys())
    columns = copy.copy(setting_columns)
    parameters = ["MRE X", "STD MRE X", "MRE Y", "STD MRE Y", "m/z 0", "STD m/z 0",
                  "m/z 1", "STD m/z 1", "m/z 2", "STD m/z 2", "m/z 3", "STD m/z 3", "FPR", "ARI"]
    for param in parameters:
        columns.append(param)
    df = pd.DataFrame(columns=columns)
    values = []
    for setting_value in settings.values():
        values.append(setting_value)

    for i in range(2):
        values.append(float(registration_metrics['mre'][i]))
        values.append(float(registration_metrics['mre_std'][i]))

    for i in range(4):
        values.append(float(spectra_metrics['spectra_mean_deviation'][i]))
        values.append(np.sqrt(float(spectra_metrics['spectra_deviation_variance'][i])))

    values.append(float(spectra_metrics['FPR']))
    values.append(float(cluster_metrics['adjusted_rand_index']))
    df = pd.concat([df, pd.DataFrame([values], columns=df.columns)],
                   axis=0, ignore_index=True)

    df.to_csv(
        output_dir + f"/results_summary_seed_{seed}_{settings['ID']}.csv", index=False)

    return A.seed, 0


base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# Use a *relative* base for output (matches paths used inside the QMD)
output_base_rel = "output"
output_base = os.path.join(base_dir, output_base_rel)
os.makedirs(output_base, exist_ok=True)

output_dir = os.path.join(output_base, "reports")
os.makedirs(output_dir, exist_ok=True)

textures_base_rel = "textures"
textures_dir = os.path.join(base_dir, textures_base_rel)
noise_texture = os.path.join(textures_dir, "noise_150x150.jpg")
noise_gradient_texture = os.path.join(textures_dir, "noise_gradient_150x150.jpg")

# Validator parameters
output = "./output/"
use_perfect_registration = False
use_perfect_segmentation = True
teaching_imprecision = 10
canvas_size = 1000
mic_spacing = 0.6897
ms_spacing = 5
cell_density = 0.2
max_cell_placement_iterations = 1000
rotation = 0
scale = 1.00
mass_list = [1, 2, 3, 4]
seed = 42
plot = False
save_img = True
remove_ambiguous = True
cell_diameter = 120
texture = None
texture_path = ""
texture_epsilon_add = 0.13
texture_epsilon_multiply = 0.3
diffusion_total_time = 0.0
diffusion_dt = 1
diffusion_coeff = [0.1, 1, 10, 50]
diffusion_barrier_alpha = [0.1, 0.3, 0.5, 0.7]
diffusion_leaky_outer_boundary = True

params_best_case = {
    "diffusion_total_time": 0,
    "remove_ambiguous": False,
    "rotation": 0,
    "scale": 1.0,
    "ms_spacing": 5,
    "texture": None,
    "texture_path": None,
    "texture_epsilon_add": 0.0,
    "texture_epsilon_multiply": 0.0,
}

params_real_case = {
    "diffusion_total_time": 50,
    "remove_ambiguous": False,
    "rotation": 0.2,
    "scale": 1.005,
    "ms_spacing": 10,
    "texture": "registration",
    "texture_path": noise_texture,
    "texture_epsilon_add": 0.13,
    "texture_epsilon_multiply": 0.3,
}

params_worst_case = {
    "diffusion_total_time": 100,
    "remove_ambiguous": False,
    "rotation": 5,
    "scale": 1.05,
    "ms_spacing": 20,
    "texture": "registration",
    "texture_path": noise_gradient_texture,
    "texture_epsilon_add": 0.25,
    "texture_epsilon_multiply": 0.5,
}

diffusion_total_time, remove_ambiguous, rotation, scale, ms_spacing, texture, texture_path, texture_epsilon_add, texture_epsilon_multiply = params_best_case.values()

validator_instances = []
result_files = glob.glob(output_dir + "/*.csv")
finished_seeds = []
seeds = list(range(1, 21))
for file in result_files:
    m = re.search(r'seed_(\d+)', file)
    seed_num = int(m.group(1)) if m else None
    if seed_num is not None:
        finished_seeds.append(seed_num)

for k in finished_seeds:
    if k in seeds:
        seeds.remove(k)

for j in seeds:
    # set seed for all random number generators
    seed = j
    if seed == 0:
        seed = random.randint(1, 10000)

    # define cell populations
    parameters_A = {"path": "./Cell_Library/DLD-1",
                    mass_list[0]: ["normal", 15, 5],
                    mass_list[1]: ["normal", 22, 5],
                    mass_list[2]: ["normal", 33, 3],
                    mass_list[3]: ["normal", 15, 5]
                    }
    parameters_B = {"path": "./Cell_Library/Colo205",
                    mass_list[0]: ["normal", 15, 5],
                    mass_list[1]: ["normal", 10, 3],
                    mass_list[2]: ["normal", 40, 20],
                    mass_list[3]: ["normal", 0, 1]
                    }

    A = CellType("A", parameters_A, 0.5)
    B = CellType("B", parameters_B, 0.5)

    cell_types = [A, B]

    run_dir = os.path.join(output_base, f"run_{j}")
    os.makedirs(run_dir, exist_ok=True)

    # Create validator object
    generation = CanvasSettings(
        output=run_dir,
        seed=seed,
        scale=scale,
        rotation=rotation,
        ms_spacing=ms_spacing,
        plot=plot,
        save_img=save_img,
        canvas_size=canvas_size,
        cell_density=cell_density,
        mic_spacing=mic_spacing,
        cell_types=cell_types,
        diffusion_total_time=diffusion_total_time,
        diffusion_dt=diffusion_dt,
        diffusion_coeff=diffusion_coeff,
        diffusion_barrier_alpha=diffusion_barrier_alpha,
        diffusion_leaky_outer_boundary=diffusion_leaky_outer_boundary,
        texture=texture,
        texture_path=texture_path,
        texture_epsilon_add=texture_epsilon_add,
        texture_epsilon_multiply=texture_epsilon_multiply)

    evaluation = dict(output=run_dir, use_perfect_segmentation=use_perfect_segmentation,
                      use_perfect_registration=use_perfect_registration,
                      remove_ambiguous=remove_ambiguous, teaching_imprecision=teaching_imprecision,
                      plot=plot, save_img=save_img)
    validator_instances.append((generation, evaluation))

slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus:
    max_workers = max(1, int(slurm_cpus))
else:
    max_workers = 4

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_validator, A)
               for A in validator_instances]
    for f in as_completed(futures):
        idx, returncode = f.result()
        status = "OK" if returncode == 0 else f"FAILED ({returncode})"
        print(f"[{datetime.datetime.now().time()}] Run {idx} finished: {status}")
