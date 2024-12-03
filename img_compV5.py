import numpy as np
import cv2
from PIL import Image
import os
import time
import multiprocessing as mp

# 1. Function to split the image into individual color channels
def split_channels(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.split(img_rgb)

# 2. Function to apply FFT to a color channel
def apply_fft(args):
    import os
    print(f"Processing channel in PID: {os.getpid()}")
    channel, compression_ratio = args
    print(f"Processing channel with compression ratio {compression_ratio}")
    f_transform = np.fft.fft2(channel)
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)

    # Keeping a central portion of frequencies
    r = int(rows * compression_ratio)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

    f_transform_shifted_masked = f_transform_shifted * mask
    f_ishift = np.fft.ifftshift(f_transform_shifted_masked)
    channel_compressed = np.fft.ifft2(f_ishift)

    return np.abs(channel_compressed)

# 3. Function to merge the channels back after FFT compression
def merge_channels(channels):
    return cv2.merge(channels)

# Function to process one image in serial mode
def process_image_serial(image_path, compression_ratio):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
    channels = split_channels(img)
    compressed_channels = [apply_fft((channel, compression_ratio)) for channel in channels]
    compressed_image = merge_channels(compressed_channels)
    return np.uint8(np.clip(compressed_image, 0, 255))

# Function to process one image in parallel mode
def process_image_parallel(image_path, compression_ratio):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
    channels = split_channels(img)
    with mp.Pool(processes=3) as pool:
        compressed_channels = pool.map(apply_fft, [(channel, compression_ratio) for channel in channels])
    compressed_image = merge_channels(compressed_channels)
    return np.uint8(np.clip(compressed_image, 0, 255))

# Function to save the image to the disk
def save_image(image, output_path):
    img_to_save = Image.fromarray(image)
    img_to_save.save(output_path)

# Main processing loop with timing for both serial and parallel
if __name__ == "__main__":
    image_files = input("Enter the paths of the images to compress, separated by commas: ").split(',')
    image_files = [file.strip() for file in image_files]

    compression_ratio = 0.3

    output_dir = "compressed_image"
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in serial
    print("\n--- Serial Processing ---")
    serial_start_time = time.time()
    for i, image_file in enumerate(image_files):
        output_file = os.path.join(output_dir, f"compressed_serial_{i+1}.jpg")
        print(f"Processing {image_file} in serial mode...")
        start_time = time.time()
        try:
            compressed_image = process_image_serial(image_file, compression_ratio)
            save_image(compressed_image, output_file)
            print(f"Saved compressed image to {output_file}.")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        print(f"Time for {image_file}: {time.time() - start_time:.2f} seconds.")
    print(f"Total time for serial processing: {time.time() - serial_start_time:.2f} seconds.")

    # Process each image in parallel
    print("\n--- Parallel Processing ---")
    parallel_start_time = time.time()
    for i, image_file in enumerate(image_files):
        output_file = os.path.join(output_dir, f"compressed_parallel_{i+1}.jpg")
        print(f"Processing {image_file} in parallel mode...")
        start_time = time.time()
        try:
            compressed_image = process_image_parallel(image_file, compression_ratio)
            save_image(compressed_image, output_file)
            print(f"Saved compressed image to {output_file}.")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        print(f"Time for {image_file}: {time.time() - start_time:.2f} seconds.")
    print(f"Total time for parallel processing: {time.time() - parallel_start_time:.2f} seconds.")
