import os
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

cell_path_prefix = "D:/Jan/fiscas-validationKit/Cell_images/microscopy/"
quarto_file = "Report.qmd"
output_dir = "D:/Jan/fiscas-validationKit/reports/"

commands = []
i = 0

for cell_line in ["HuTu80/"]:
    for diff_coeff in [0, 1, 3, 5, 10]:
        for diff_time in [0, 100, 300, 500]:
            for j in range(1):
                cell_path = cell_path_prefix + cell_line
                param_overrides = (
                    f"-P path_sample_cells:{cell_path} "
                    f"-P allow_multiplets:True "
                    f"-P diffusion_coeff:{diff_coeff} "
                    f"-P diffusion_total_time:{diff_time} "
                    f"-P teaching_imprecision:5 "
                    f"-P remove_ambiguous:False "
                    f"-P rotation:0 "
                    f"-P ms_spacing:5 "
                    f"-P seed:{j} "
                    f"-P use_perfect_segmentation:True "
                    f"-P scale:1 "
                    f"-P use_perfect_registration:True "
                    f"-P save_img:True "
                    f"-P canvas_size:1000 "
                    f"-P cell_density:0.2"
                )

                output_file = f"Report_ms{i}.pdf"
                command = (
                    f"quarto render {quarto_file} {param_overrides} "
                    f"--output {output_file} --output-dir {output_dir}"
                )
                commands.append((i, command))
                i += 1

def run_quarto(task):
    idx, cmd = task
    print(f"[{datetime.datetime.now().time()}] Starting run {idx}")
    result = subprocess.run(cmd, shell=True)
    return idx, result.returncode

max_workers = os.cpu_count()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_quarto, task) for task in commands]

    for f in as_completed(futures):
        idx, returncode = f.result()
        status = "OK" if returncode == 0 else f"FAILED ({returncode})"
        print(f"[{datetime.datetime.now().time()}] Run {idx} finished: {status}")