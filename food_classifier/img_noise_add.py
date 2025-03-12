import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r"C:\Users\DELL\Downloads\Boiled Egg_00.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Add Gaussian noise
mean = 0
sigma = 25  # Standard deviation
gaussian_noise = np.random.normal(mean, sigma, img.shape)
noisy_img = np.clip(img + gaussian_noise, 0, 255).astype(np.uint8)

# Show the noisy image
plt.imshow(noisy_img)
plt.show()
