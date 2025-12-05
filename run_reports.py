import os
import datetime


# Define the parameters and their values for each run
i = 0
cell_path_prefix = "./Cell_images/microscopy/"
for cell_line in ["HuTu80/"]:
    for diff_coeff in [0, 1, 3, 5, 10]:
        for diff_time in [0, 100, 300, 500]:
            for j in range(1):
                print(f"Starting run {i} at {datetime.datetime.now().time()}")
                cell_path = cell_path_prefix + cell_line
                # Construct the parameter overrides for the command
                param_overrides = f"-P path_sample_cells:{cell_path} -P allow_multiplets:True -P diffusion_coeff:{diff_coeff} -P diffusion_total_time:{diff_time} -P teaching_imprecision:5 -P remove_ambiguous:False -P rotation:0 -P ms_spacing:5 -P seed:{j} -P use_perfect_segmentation:True -P scale:1 -P use_perfect_registration:True -P save_img:True -P canvas_size:1000 -P cell_density:0.2"
                # Path to the Quarto file
                quarto_file = "Report.qmd"
                output_dir = "./reports/"
                output_file = f"Report_ms{i}.pdf"
                i += 1
                # Construct the Quarto render command with the output option
                command = f"quarto render {quarto_file} {param_overrides} --output {output_file} --output-dir {output_dir}"
                
                # Print the command for debugging
                print(f"Running: {command}")
                
                # Execute the command
                os.system(command)

    
