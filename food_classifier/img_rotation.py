from PIL import Image
import os

# Load your image
image_path = "C:\\Users\\DELL\\Downloads\\Gym photos\\circle_with_arrow.JPEG"  # Replace with your image path
original_image = Image.open(image_path)

# Convert to RGB if the image has an alpha channel (RGBA)
if original_image.mode == 'RGBA':
    original_image = original_image.convert('RGB')

output_folder = "C:\\Users\\DELL\\Downloads\\rotated_image"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through 1 degree to 360 degrees
for angle in range(1, 361):
    rotated_image = original_image.rotate(angle)
    # Save the rotated image to the specified folder
    rotated_image.save(f"{output_folder}/rotated_image_{angle}.jpg")
    print(f"Saved rotated_image_{angle}.jpg in {output_folder}")

print("Rotation complete!")