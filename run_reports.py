import os
import sys
import shutil
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

cell_path_prefix = "./Cell_images/microscopy"
original_quarto_file = "Report.qmd"

# Use a *relative* base for output (matches paths used inside the QMD)
output_base_rel = "output"                              # <--- NEW
output_base = os.path.join(base_dir, output_base_rel)
os.makedirs(output_base, exist_ok=True)

output_dir = os.path.join(output_base, "reports")
os.makedirs(output_dir, exist_ok=True)

commands = []
i = 0

for cell_line in ["HuTu80"]:
    for diff_coeff in [0, 1]:
        for diff_time in [10, 20, 100]:
            for j in range(1):
                # Build cell path safely (handles trailing / etc.)
                cell_path = os.path.join(cell_path_prefix, cell_line)

                # Make a unique copy of the QMD for this run (dynamic naming)
                temp_qmd = f"Report_run_{i}.qmd"
                shutil.copy(
                    os.path.join(base_dir, original_quarto_file),
                    os.path.join(base_dir, temp_qmd),
                )

                # Per-run *relative* output dir: "output/run_i"
                run_output_rel = os.path.join(output_base_rel, f"run_{i}")   # <--- CHANGED
                run_output_abs = os.path.join(base_dir, run_output_rel)
                os.makedirs(run_output_abs, exist_ok=True)

                # Build parameter overrides as a list (no shell quoting issues)
                param_overrides = [
                    "-P", f"path_sample_cells:{cell_path}",
                    "-P", "allow_multiplets:True",
                    "-P", f"diffusion_coeff:{diff_coeff}",
                    "-P", f"diffusion_total_time:{diff_time}",
                    "-P", "teaching_imprecision:5",
                    "-P", "remove_ambiguous:False",
                    "-P", "rotation:0",
                    "-P", "ms_spacing:5",
                    "-P", f"seed:{j}",
                    "-P", "use_perfect_segmentation:True",
                    "-P", "scale:1",
                    "-P", "use_perfect_registration:True",
                    "-P", "save_img:True",
                    "-P", "canvas_size:1000",
                    "-P", "cell_density:0.2",
                    "-P", f"output:{run_output_rel}",        # <--- pass RELATIVE path
                ]

                output_file = f"Report_ms{i}.pdf"

                cmd = [
                    "quarto", "render", temp_qmd,
                    *param_overrides,
                    "--output", output_file,
                    "--output-dir", output_dir,
                ]

                commands.append((i, cmd, temp_qmd))
                i += 1


def run_quarto(task):
    idx, cmd, temp_qmd = task
    print(f"[{datetime.datetime.now().time()}] Starting run {idx}")

    # Use the same Python as this script (your micromamba venv)
    env = os.environ.copy()
    env["QUARTO_PYTHON"] = sys.executable

    # Run from the project directory (where FISCAS_Validator.py is)
    result = subprocess.run(cmd, cwd=base_dir, env=env)

    # Optional cleanup: remove the temp qmd and its notebook artifacts
    base_name = os.path.splitext(temp_qmd)[0]
    for ext in (".quarto_ipynb", ".ipynb"):
        nb_path = os.path.join(base_dir, base_name + ext)
        if os.path.exists(nb_path):
            os.remove(nb_path)
    if os.path.exists(os.path.join(base_dir, temp_qmd)):
        os.remove(os.path.join(base_dir, temp_qmd))

    return idx, result.returncode


max_workers = os.cpu_count()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_quarto, task) for task in commands]
    for f in as_completed(futures):
        idx, returncode = f.result()
        status = "OK" if returncode == 0 else f"FAILED ({returncode})"
        print(f"[{datetime.datetime.now().time()}] Run {idx} finished: {status}")