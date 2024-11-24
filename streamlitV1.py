import streamlit as st
import numpy as np
import cv2
from PIL import Image
from img_compV1 import *

def compress_image_color(image, compression_ratio):
    # Convert to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to NumPy array
    img = np.array(image)

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

def main():
    st.title("Image Compressor with FFT")
    st.write("Upload an image and set the compression ratio to compress the image using FFT.")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    compression_ratio = st.slider("Compression Ratio", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

    if uploaded_file is not None:
        # Display original image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Compress image
        compressed_image_array = compress_image_color(original_image, compression_ratio)
        
        # Normalize the compressed image to [0, 255] and convert to uint8
        compressed_image_array = np.clip(compressed_image_array, 0, 255).astype(np.uint8)
        compressed_image = Image.fromarray(compressed_image_array)

        # Display compressed image
        st.image(compressed_image, caption=f"Compressed Image (Ratio: {compression_ratio})", use_column_width=True)
        

if __name__ == "__main__":
    main()
