import numpy as np
import cv2
from PIL import Image
import os
import time
import multiprocessing as mp

# 1. Function to split the image into individual color channels
def split_channels(image):
    # Convert BGR to RGB (OpenCV reads images as BGR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Split the image into 3 channels (Red, Green, Blue)
    return cv2.split(img_rgb)

# 2. Function to apply FFT to a color channel
def apply_fft(channel):
    # Apply 2D FFT
    f_transform = np.fft.fft2(channel)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Mask to keep only part of the frequencies
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)

    # Keeping a central portion of frequencies (e.g., 50%)
    r = int(rows * 0.5)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

    # Apply the mask and perform inverse FFT
    f_transform_shifted_masked = f_transform_shifted * mask
    f_ishift = np.fft.ifftshift(f_transform_shifted_masked)
    channel_compressed = np.fft.ifft2(f_ishift)

    # Return the absolute value of the transformed (compressed) channel
    return np.abs(channel_compressed)

# 3. Function to merge the channels back after FFT compression
def merge_channels(channels):
    # Merge the 3 channels back into a single RGB image
    return cv2.merge(channels)

# Function to process one image with parallelized FFT on each channel
def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
    # Step 1: Split the image into color channels
    channels = split_channels(img)

    # Step 2: Apply FFT to each channel in parallel
    with mp.Pool(processes=3) as pool:
        compressed_channels = pool.map(apply_fft, channels)

    # Step 3: Merge the channels back together after FFT
    compressed_image = merge_channels(compressed_channels)

    # Convert back to BGR for saving and clip to [0, 255], then convert to uint8
    compressed_image_bgr = np.uint8(np.clip(compressed_image, 0, 255))

    return compressed_image_bgr

# Function to save the image to the disk
def save_image(image, output_path):
    img_to_save = Image.fromarray(image)
    img_to_save.save(output_path)

if __name__ == "__main__":
    # Example input
    image_files = input("Enter the paths of the images to compress, separated by commas: ").split(',')
    image_files = [file.strip() for file in image_files]

    # Output directory setup
    output_dir = "compressed_image"
    os.makedirs(output_dir, exist_ok=True)

    # Process and save each image
    for image_file in image_files:
        output_file = os.path.join(output_dir, f"compressed_{os.path.basename(image_file)}")
        print(f"Processing {image_file}...")
        start_time = time.time()
        try:
            compressed_image = process_image(image_file)
            save_image(compressed_image, output_file)
            print(f"Saved compressed image to {output_file}.")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        end_time = time.time()
        print(f"Time taken for {image_file}: {end_time - start_time:.2f} seconds.")
