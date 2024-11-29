import numpy as np
import cv2
import multiprocessing as mp
import os
from PIL import Image
import time

def compress_channel(channel, compression_ratio):
    f_transform = np.fft.fft2(channel)
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r = int(rows * compression_ratio)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

    f_transform_shifted_masked = f_transform_shifted * mask
    f_ishift = np.fft.ifftshift(f_transform_shifted_masked)
    channel_compressed = np.fft.ifft2(f_ishift)
    return np.abs(channel_compressed)

def compress_image(image_path, compression_ratio, parallel=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channels = cv2.split(img)

    if parallel:
        with mp.Pool(processes=len(channels)) as pool:
            compressed_channels = pool.map(lambda ch: compress_channel(ch, compression_ratio), channels)
    else:
        compressed_channels = [compress_channel(channel, compression_ratio) for channel in channels]

    compressed_image = cv2.merge(compressed_channels)
    return compressed_image

def save_image(image_array, output_path):
    img_to_save = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
    img_to_save.save(output_path)

def process_images(image_files, output_files, compression_ratio, parallel=False):
    start_time = time.time()
    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(compress_and_save, zip(image_files, output_files, [compression_ratio]*len(image_files)))
    else:
        for image_file, output_file in zip(image_files, output_files):
            compress_and_save(image_file, output_file, compression_ratio)
    end_time = time.time()
    processing_type = "parallel" if parallel else "serial"
    print(f"Total time for {processing_type} processing: {end_time - start_time:.2f} seconds.")

def compress_and_save(image_file, output_file, compression_ratio):
    try:
        compressed_image = compress_image(image_file, compression_ratio, parallel=False)
        save_image(compressed_image, output_file)
        print(f"Saved compressed image to {output_file}.")
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    # Input image files
    image_files = input("Enter the paths of the images to compress, separated by commas: ").split(',')
    image_files = [file.strip() for file in image_files]

    # Output directory setup
    output_dir = "compressed_image"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output filenames
    output_file_serial = [os.path.join(output_dir, f'compressed_serial{i+1}.jpg') for i in range(len(image_files))]
    output_file_parallel = [os.path.join(output_dir, f'compressed_parallel{i+1}.jpg') for i in range(len(image_files))]
    compression_ratio = 0.25  # Keep 50% of frequencies

    # Serial processing
    print("\nStarting serial processing...")
    process_images(image_files, output_file_serial, compression_ratio, parallel=False)

    # Parallel processing
    print("\nStarting parallel processing...")
    process_images(image_files, output_file_parallel, compression_ratio, parallel=True)
