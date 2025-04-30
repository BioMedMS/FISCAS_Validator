import os

# Define the parameters and their values for each run
for ms_spacing in [1, 5, 20]:
    for rotation in [0, 1, 5, 10]:
        for scale in [1, 0.99, 1.1]:
            # Construct the parameter overrides for the command
            param_overrides = f"-P ms_spacing:{ms_spacing} -P rotation:{rotation} -P scale:{scale} -P seed:42"
            
            # Path to the Quarto file
            quarto_file = "Report.qmd"
            
            output_file = f"output/Report_ms{ms_spacing}_rot{rotation}_scale{scale}.pdf"
            
            # Construct the Quarto render command with the output option
            command = f"quarto render {quarto_file} {param_overrides} --output {output_file}"
            
            # Print the command for debugging
            print(f"Running: {command}")
            
            # Execute the command
            os.system(command)
