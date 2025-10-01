import os

# Define the parameters and their values for each run
i = 0
cell_path_prefix = "D:/Jan/fiscas-validationKit/Cell_images/microscopy/"
for cell_line in ["HuTu80/"]:
    for j in range(1,6):
        for teaching_imprecision in [0, 5, 10, 20, 50]:
            print(teaching_imprecision)
            cell_path = cell_path_prefix + cell_line
            print(cell_path)
            # Construct the parameter overrides for the command
            param_overrides = f"-P path_sample_cells:{cell_path} -P allow_multiplets:True -P diffusion:0 -P teaching_imprecision:{teaching_imprecision} -P remove_ambiguous:True -P rotation:10 -P ms_spacing:5 -P seed:{j} -P use_perfect_segmentation:True -P scale:1 -P use_perfect_registration:False -P save_img:True -P canvas_size:1500 -P cell_density:0.2"
            print(param_overrides)
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
