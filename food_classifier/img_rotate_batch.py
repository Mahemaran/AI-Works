from PIL import Image
import os
import random
from concurrent.futures import ThreadPoolExecutor

# Function to rotate and save the image
def rotate_and_save(image, angle, output_folder, base_filename="omeletss"):
    rotated_image = image.rotate(angle)
    rotated_image.save(f"{output_folder}/{base_filename}_{angle}.jpg")
    print(f"Saved {base_filename}_{angle}.jpg")

# Load your image
image_path = r"D:\Maran\image9-2023-09-02T194411-22.webp" # Replace with your image path
original_image = Image.open(image_path)

# Convert to RGB if the image has an alpha channel (RGBA)
if original_image.mode == 'RGBA':
    original_image = original_image.convert('RGB')

# Resize the image to a consistent size (optional, can be customized)
original_image = original_image.resize((224, 224))  # Example resize to 224x224

# Define output folder
output_folder = r"D:\Maran\image_training_data\Food_images_datasets\Omelets"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Use ThreadPoolExecutor for parallel processing (if needed)
with ThreadPoolExecutor() as executor:
    # Loop through random angles, or specify how many random rotations you want
    for angle in range(1, 361):  # You can randomize the range if needed
        executor.submit(rotate_and_save, original_image, angle, output_folder)

print("Rotation complete!")
