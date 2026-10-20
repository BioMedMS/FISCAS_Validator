import os
import ast
import math
import pandas as pd
import subprocess
import yaml

def clean_value(val):
    """Safely converts stringified lists to lists, and NaN to None."""
    # 1. Handle NaN values
    if isinstance(val, float) and math.isnan(val):
        return None
    
    # 2. Handle stringified lists like "[1, 2, 3]" or "['A', 'B']"
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val # Fallback to string if parsing fails
            
    return val

def main(file_path):
    df = pd.read_csv(file_path)
    
    columns_to_extract = [
        "cell_types", "use_perfect_registration", "use_perfect_segmentation", 
        "teaching_imprecision", "canvas_size","mic_spacing", "ms_spacing", 
        "cell_density", "max_cell_placement_iterations", "rotation", "scale", "mass_list",
        "seed", "plot", "save_img", "remove_ambiguous", "cell_diameter", "texture", 
        "texture_path", "texture_epsilon_add", "texture_epsilon_multiply", 
        "diffusion_total_time", "diffusion_dt", "diffusion_coeff", 
        "diffusion_barrier_alpha", "diffusion_leaky_outer_boundary"  
    ]
    
    # Extract the dictionary
    raw_dict = df[columns_to_extract].to_dict(orient='records')[0]
    output_dir = f"./output/generated_reports/seed_{raw_dict['seed']}"
    os.makedirs(output_dir, exist_ok=True)
    # Clean every value in the dictionary
    params_dict = {k: clean_value(v) for k, v in raw_dict.items()}
    params_dict["output"] = output_dir  # Use the CSV filename (without extension) as the output folder name
    # Save to a perfectly formatted .yml file
    #params_dict["texture"] = "all"
    params_dict["texture_path"] = "./textures/noise_100x100_v2.jpg"
    #params_dict["rotation"] = 5
    params_file = "./params.yml"
    if "output" in params_dict and isinstance(params_dict["output"], str):
        params_dict["output"] = params_dict["output"].replace("\\", "/")
    with open(params_file, 'w') as f:
        yaml.dump(params_dict, f, default_flow_style=False)
    

    # Pass to Quarto
    try:
        subprocess.run(
            ["quarto", "render", "Report.qmd", "--execute-params", params_file], 
            capture_output=True, 
            text=True, 
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during Quarto execution: {e.stderr}")
        raise e
    return None

if __name__ == "__main__":
    main(os.sys.argv[1])