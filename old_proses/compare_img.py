import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(image1, image2):
    # Compute the Mean Squared Error (MSE)
    return np.sum((image1 - image2) ** 2) / float(image1.shape[0] * image1.shape[1])

def compare_images(image1, image2):
    # Compute MSE and SSIM
    m = mse(image1, image2)
    s = ssim(image1, image2)
    return m, s

# Read the images
image1 = cv2.imread("img/img-1B.jpg")
image2 = cv2.imread("compressed_image/compressed_parallel1.jpg")

# Check if images are loaded properly
if image1 is None or image2 is None:
    print("Error loading images!")
else:
    # Resize images to match (optional)
    image1_resized = cv2.resize(image1, (300, 300))  # Resize to same size
    image2_resized = cv2.resize(image2, (300, 300))
    
    # Convert images to grayscale (optional, for simpler comparison)
    image1_gray = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)
    
    # Compare the images
    mse_value, ssim_value = compare_images(image1_gray, image2_gray)
    
    # Output comparison results
    print(f"MSE: {mse_value}")
    print(f"SSIM: {ssim_value}")
