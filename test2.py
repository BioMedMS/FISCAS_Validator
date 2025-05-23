import imageio 
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def images_to_video(image_folder, output_video, fps=30):

    # Get all image files in the folder and sort by filename
    image_files = sorted(Path(image_folder).glob("*.jpg"), key=lambda x: int(x.stem.split('_')[-1]))  # Adjust split logic if needed
    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Read images and write to video
    writer = imageio.get_writer(output_video, fps=fps)
    for image_file in image_files:
        image = imageio.imread(image_file)
        writer.append_data(image)
    writer.close()

# Example usage
image_folder = "D:/Jan/fiscas-validationKit/output/animation"
output_video = "output_video.mp4"
images_to_video(image_folder, output_video, fps=5)