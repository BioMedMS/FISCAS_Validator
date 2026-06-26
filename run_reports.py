import os
import sys
import shutil
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

original_quarto_file = "Report.qmd"
output_base_rel = "output"
output_base = os.path.join(base_dir, output_base_rel)
os.makedirs(output_base, exist_ok=True)

output_dir = os.path.join(output_base, "reports")
os.makedirs(output_dir, exist_ok=True)

textures_base_rel = "textures"
textures_dir = os.path.join(base_dir, textures_base_rel)
noise_texture = os.path.join(textures_dir, "noise_150x150.jpg")
noise_gradient_texture = os.path.join(textures_dir, "noise_gradient_150x150.jpg")

commands = []
i = 0

<<<<<<< HEAD


for j in range(1,6):
=======
for j in range(1,2):
>>>>>>> f7cf7599c8a3622b74cd621ab0bc0a58c6b3c764
    for remove_ambiguous in [True, False]:

        # Build parameter overrides as a list (no shell quoting issues)
        param_overrides_best_case = [
            "-P", "allow_multiplets:True",
            "-P", "diffusion_total_time:0",
            "-P", f"remove_ambiguous:{remove_ambiguous}",
            "-P", "rotation:0",
            "-P", "scale:1.0",
            "-P", "ms_spacing:5",
            "-P", "texture:None",
            "-P", f"texture_path:None",
            "-P", "texture_epsilon_add:0",
            "-P", "texture_epsilon_multiply:0",
            "-P", f"seed:{j}",    
        ]

        param_overrides_real_case = [
            "-P", "allow_multiplets:True",
            "-P", "diffusion_total_time:0",
            "-P", f"remove_ambiguous:{remove_ambiguous}",
            "-P", "rotation:0.2",
            "-P", "scale:1.005",
            "-P", "ms_spacing:8",
            "-P", f"texture_path:{noise_texture}",
            "-P", "texture_epsilon_add:0.13",
            "-P", "texture_epsilon_multiply:0.3",
            "-P", f"seed:{j}",
        ]

        param_overrides_worst_case = [
            "-P", "allow_multiplets:True",
            "-P", "diffusion_total_time:0",
            "-P", f"remove_ambiguous:{remove_ambiguous}",
            "-P", "rotation:5",
            "-P", "scale:1.05",
            "-P", "ms_spacing:20",
            "-P", f"texture_path:{noise_gradient_texture}",
            "-P", "texture_epsilon_add:0.3",
            "-P", "texture_epsilon_multiply:0.6",
            "-P", f"seed:{j}",
        ]

        param_settings = [param_overrides_real_case, param_overrides_worst_case]

        for param_override in param_settings:
            # Make a unique copy of the QMD for this run (dynamic naming)
            temp_qmd = f"Report_run_{i}.qmd"
            shutil.copy(
                os.path.join(base_dir, original_quarto_file),
                os.path.join(base_dir, temp_qmd),
            )

            # Per-run *relative* output dir: "output/run_i"
            run_output_rel = os.path.join(output_base_rel, f"run_{i}")
            run_output_abs = os.path.join(base_dir, run_output_rel)
            os.makedirs(run_output_abs, exist_ok=True)

            param_override.append("-P")
            param_override.append(f"output:{run_output_rel}")
            output_file = f"Report_ms{i}.pdf"

            cmd = [
                "quarto", "render", temp_qmd,
                *param_override,
                "--output", output_file,
                "--output-dir", output_dir,
            ]

            commands.append((i, cmd, temp_qmd))
            i += 1

def run_quarto(task):
    idx, cmd, temp_qmd = task
    print(f"[{datetime.datetime.now().time()}] Starting run {idx}")
    env = os.environ.copy()
    env["QUARTO_PYTHON"] = sys.executable
    result = subprocess.run(cmd, cwd=base_dir, env=env)
    base_name = os.path.splitext(temp_qmd)[0]
    for ext in (".quarto_ipynb", ".ipynb"):
        nb_path = os.path.join(base_dir, base_name + ext)
        if os.path.exists(nb_path):
            os.remove(nb_path)
    if os.path.exists(os.path.join(base_dir, temp_qmd)):
        os.remove(os.path.join(base_dir, temp_qmd))

    return idx, result.returncode

<<<<<<< HEAD
# Save commands to a file for debugging or later use
commands_file = os.path.join(output_base, "commands.txt")
with open(commands_file, "w") as f:
    for idx, cmd, temp_qmd in commands:
        f.write(f"Run {idx}:\n")
        f.write(" ".join(cmd) + "\n")
        f.write("\n")
max_workers = 4#os.cpu_count()

=======
slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus:
    max_workers = max(1, int(slurm_cpus))
else:
    max_workers = 2
>>>>>>> f7cf7599c8a3622b74cd621ab0bc0a58c6b3c764

with ThreadPoolExecutor(max_workers=max_workers) as executor:
   futures = [executor.submit(run_quarto, task) for task in commands]
   for f in as_completed(futures):
       idx, returncode = f.result()
       status = "OK" if returncode == 0 else f"FAILED ({returncode})"
       print(f"[{datetime.datetime.now().time()}] Run {idx} finished: {status}")