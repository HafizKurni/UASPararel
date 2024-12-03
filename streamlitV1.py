import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import time
import multiprocessing as mp
from io import BytesIO
from img_compV5 import *

# Define a function to load an image from an uploaded file
def load_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def save_image_to_bytes(image):
    output = BytesIO()
    img = Image.fromarray(image)
    img.save(output, format="JPEG")
    return output.getvalue()

# Streamlit app
st.title("Image Compression with FFT")
st.write("Upload an image, set a compression ratio, and choose processing mode (serial or parallel).")

# Upload image
uploaded_files = st.file_uploader("Upload Image Files", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Compression ratio input
compression_ratio = st.slider(
    "Compression Ratio (0.0 to 1.0):", min_value=0.0, max_value=1.0, value=0.3, step=0.1
)

# Processing mode selection
processing_mode = st.radio(
    "Processing Mode:", options=["Serial", "Parallel"], index=0
)

# Process button
if st.button("Process Images"):
    if not uploaded_files:
        st.warning("Please upload at least one image file.")
    else:
        output_dir = "compressed_images"
        os.makedirs(output_dir, exist_ok=True)

        st.write("### Processing Images")
        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"Processing `{uploaded_file.name}`...")

            # Load the image
            try:
                img = load_image(uploaded_file)

                # Process the image
                start_time = time.time()
                if processing_mode == "Serial":
                    compressed_image = process_image_serial(img, compression_ratio)
                else:
                    compressed_image = process_image_parallel(img, compression_ratio)
                elapsed_time = time.time() - start_time

                # Save and display the result
                compressed_image_bytes = save_image_to_bytes(compressed_image)
                st.image(compressed_image, caption=f"Compressed Image ({uploaded_file.name})", use_column_width=True)

                st.write(f"Processing time: {elapsed_time:.2f} seconds")

                # Download link
                st.download_button(
                    label=f"Download {uploaded_file.name}",
                    data=compressed_image_bytes,
                    file_name=f"compressed_{i+1}.jpg",
                    mime="image/jpeg",
                )

            except Exception as e:
                st.error(f"Error processing `{uploaded_file.name}`: {e}")
