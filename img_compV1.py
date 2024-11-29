import numpy as np
import cv2
from PIL import Image
import concurrent.futures
import time
import os

def compress_image_color(image_path, compression_ratio):
    # Read the image in color (BGR format)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Split into individual color channels
    channels = cv2.split(img)
    compressed_channels = []

    for channel in channels:
        # Apply FFT
        f_transform = np.fft.fft2(channel)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Create a mask to keep certain frequencies
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)

        # Keep only a fraction of the high-frequency components
        r = int(rows * compression_ratio)
        mask[crow - r:crow + r, ccol - r:ccol + r] = 1

        # Apply mask to the frequency domain
        f_transform_shifted_masked = f_transform_shifted * mask

        # Inverse FFT to get the compressed channel
        f_ishift = np.fft.ifftshift(f_transform_shifted_masked)
        channel_compressed = np.fft.ifft2(f_ishift)
        compressed_channels.append(np.abs(channel_compressed))

    # Merge compressed channels back into an RGB image
    compressed_image = cv2.merge(compressed_channels)
    return compressed_image

def save_image(image_array, output_path):
    # Convert to uint8, ensure values are clipped to [0, 255]
    img_to_save = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
    img_to_save.save(output_path)

def process_images_serial(image_files, output_files, compression_ratio):
    for image_file, output_file in zip(image_files, output_files):
        start_time = time.time()
        compressed_image = compress_image_color(image_file, compression_ratio)
        save_image(compressed_image, output_file)
        end_time = time.time()
        print(f"Serialized compression for {image_file} took {end_time - start_time:.2f} seconds.")

def process_images_parallel(image_files, output_files, compression_ratio):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for image_file, output_file in zip(image_files, output_files):
            start_time = time.time()
            futures.append(executor.submit(compress_and_save, image_file, output_file, compression_ratio, start_time))
        
        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # This will also raise exceptions if any occurred

def compress_and_save(image_file, output_file, compression_ratio, start_time):
    compressed_image = compress_image_color(image_file, compression_ratio)
    save_image(compressed_image, output_file)
    end_time = time.time()
    print(f"Parallel compression for {image_file} took {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Input image files from user
    image_files = input("Enter the paths of the images to compress, separated by commas: ").split(',')
    image_files = [file.strip() for file in image_files]  # Clean up whitespace
    
    output_dir = "compressed_image"

    os.makedirs(output_dir, exist_ok=True)
    
    output_fileSerial = [os.path.join(output_dir, f'compressed_serial{i+1}.jpg')  for i in range(len(image_files))]
    output_filePararell = [os.path.join(output_dir, f'compressed_parallel{i+1}.jpg') for i in range(len(image_files))]
    compression_ratio = 0.05  # Keep 10% of the frequencies

    # Serial processing
    print("Starting serial processing...")
    start_serial = time.time()
    process_images_serial(image_files, output_fileSerial, compression_ratio)
    end_serial = time.time()
    print(f"Total time for serial processing: {end_serial - start_serial:.2f} seconds.")

    # Parallel processing
    print("\nStarting parallel processing...")
    start_parallel = time.time()
    process_images_parallel(image_files, output_filePararell, compression_ratio)
    end_parallel = time.time()
    print(f"Total time for parallel processing: {end_parallel - start_parallel:.2f} seconds.")

# img/img-1.jpg, img/img-2.jpg, img/img-3.jpg, img/img-4.jpg, img/img-5.jpg, img/img-6.jpg, img/img-7.jpg

# img/img-1B.jpg, img/img-2B.jpg, img/img-3B.jpg, img/img-4B.jpg, img/img-5B.jpg
