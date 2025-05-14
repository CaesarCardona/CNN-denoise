import os
import numpy as np
from PIL import Image

# Noise level: 0.1 means 10% Gaussian noise
NOISE_LEVEL = 0.1

INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "dataset-with-noise"

def add_gaussian_noise(image, noise_level):
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    noisy_array = img_array + noise
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def process_images(input_folder, output_folder, noise_level):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_path):
            try:
                with Image.open(input_path) as img:
                    noisy_img = add_gaussian_noise(img, noise_level)
                    noisy_img.save(output_path)
                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, NOISE_LEVEL)

