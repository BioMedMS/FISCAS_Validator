import os

# Define the parameters and their values for each run
i = 0
cell_path_prefix = "D:/Jan/fiscas-validationKit/Cell_images/microscopy/"
for cell_line in ["HuTu80/"]:
    for diffusion in [0, 1, 3, 5, 10]:
        for remove_ambiguous in [True, False]:
            for j in range(1,4):
                cell_path = cell_path_prefix + cell_line
                # Construct the parameter overrides for the command
                param_overrides = f"-P path_sample_cells:{cell_path} -P allow_multiplets:True -P diffusion:{diffusion} -P teaching_imprecision:5 -P remove_ambiguous:{remove_ambiguous} -P rotation:0 -P ms_spacing:5 -P seed:{j} -P use_perfect_segmentation:True -P scale:1 -P use_perfect_registration:True -P save_img:True -P canvas_size:1200 -P cell_density:0.2"
                # Path to the Quarto file
                quarto_file = "Report.qmd"
                output_dir = "D:/Jan/fiscas-validationKit/reports/"
                output_file = f"Report_ms{i}.pdf"
                i += 1
                # Construct the Quarto render command with the output option
                command = f"quarto render {quarto_file} {param_overrides} --output {output_file} --output-dir {output_dir}"
                
                # Print the command for debugging
                print(f"Running: {command}")
                
                # Execute the command
                os.system(command)

    for remove_ambiguous in [True, False]:
        for ms_spacing in [1, 2, 5, 10, 20]:
            for j in range(1,4):
                cell_path = cell_path_prefix + cell_line
                # Construct the parameter overrides for the command
                param_overrides = f"-P path_sample_cells:{cell_path} -P allow_multiplets:True -P diffusion:0 -P teaching_imprecision:5 -P remove_ambiguous:{remove_ambiguous} -P rotation:0 -P ms_spacing:{ms_spacing} -P seed:{j} -P use_perfect_segmentation:True -P scale:1 -P use_perfect_registration:True -P save_img:True -P canvas_size:1200 -P cell_density:0.2"
                # Path to the Quarto file
                quarto_file = "Report.qmd"
                output_dir = "D:/Jan/fiscas-validationKit/reports/"
                output_file = f"Report_ms{i}.pdf"
                i += 1
                # Construct the Quarto render command with the output option
                command = f"quarto render {quarto_file} {param_overrides} --output {output_file} --output-dir {output_dir}"
                
                # Print the command for debugging
                print(f"Running: {command}")
                
                # Execute the command
                os.system(command)

    for remove_ambiguous in [True, False]:
        for diffusion in [0, 1, 3, 5, 10]:
            for j in range(1,4):
                cell_path = cell_path_prefix + cell_line
                # Construct the parameter overrides for the command
                param_overrides = f"-P path_sample_cells:{cell_path} -P allow_multiplets:True -P diffusion:{diffusion} -P teaching_imprecision:10 -P remove_ambiguous:{remove_ambiguous} -P rotation:2 -P ms_spacing:5 -P seed:{j} -P use_perfect_segmentation:True -P scale:1.01 -P use_perfect_registration:False -P save_img:True -P canvas_size:1200 -P cell_density:0.2"
                # Path to the Quarto file
                quarto_file = "Report.qmd"
                output_dir = "D:/Jan/fiscas-validationKit/reports/"
                output_file = f"Report_ms{i}.pdf"
                i += 1
                # Construct the Quarto render command with the output option
                command = f"quarto render {quarto_file} {param_overrides} --output {output_file} --output-dir {output_dir}"
                
                # Print the command for debugging
                print(f"Running: {command}")
                
                # Execute the command
                os.system(command)
