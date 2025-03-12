from PIL import Image
import os

# Load your image
image_path = "C:\\Users\\DELL\\Downloads\\Gym photos\\whatsapp.png"  # Replace with your image path
original_image = Image.open(image_path)

# Shear factors (you can experiment with different values)
shear_factor_xy = 0.01  # Shearing along the X-axis
shear_factor_yx = 0.01  # Shearing along the Y-axis

if original_image.mode == 'RGBA':
    original_image = original_image.convert('RGB')

output_folder = "C:\\Users\\DELL\\Downloads\\sheared_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through shear factors to create multiple transformations (optional)
for shear_value in range(1, 11):  # This will apply shear values from 1 to 10
    # Create an affine transformation matrix for shearing along X-axis
    shear_matrix_x = (1, shear_factor_xy * shear_value, 0,
                      0, 1, 0)  # Shear along X

    shear_matrix_y = (1, 0, 0,
                      shear_factor_yx * shear_value, 1, 0)  # Shear along Y

    # Apply shear transformation using the matrix
    sheared_image_x = original_image.transform(original_image.size, Image.AFFINE, shear_matrix_x)
    sheared_image_y = original_image.transform(original_image.size, Image.AFFINE, shear_matrix_y)

    # Save the transformed images
    sheared_image_x.save(f"{output_folder}/sheared_x_{shear_value}.jpg")
    sheared_image_y.save(f"{output_folder}/sheared_y_{shear_value}.jpg")

    print(f"Saved sheared_x_{shear_value}.jpg and sheared_y_{shear_value}.jpg")

print("Shear transformation complete!")
