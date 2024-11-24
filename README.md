# Image Compressor with FFT

## Overview

This program allows users to upload an image and apply compression using Fast Fourier Transform (FFT). The user can adjust the compression ratio through a slider, and the app will display both the original and compressed images. The compression is done in the frequency domain by filtering out high-frequency components of the image, leading to a loss of finer details but smaller file sizes.

## Features

- **Image Upload**: Users can upload an image file in `.jpg`, `.jpeg`, or `.png` format.
- **Compression Ratio Slider**: Adjust the compression ratio using a slider to control the amount of high-frequency components to retain.
- **Real-time Image Display**: The app will display both the original and compressed images after applying the selected compression.
- **Frequency Domain Compression**: Uses FFT to convert the image into its frequency components and applies a mask to retain only a fraction of the frequency data.

## Requirements

This program is built using the following libraries:
- `streamlit`: For building the web interface.
- `numpy`: For numerical operations, particularly FFT.
- `opencv-python`: For image processing and frequency domain manipulation.
- `Pillow`: For image handling and conversion.

To install the required dependencies, run:

```bash
pip install streamlit numpy opencv-python Pillow
```

## How It Works

1. **Upload Image**: Users upload an image (JPG, PNG, or JPEG format).
2. **Set Compression Ratio**: A slider allows users to select the compression ratio (between 0.01 and 0.5). This ratio determines how many of the image's high-frequency components will be retained.
3. **FFT Compression**: The image is converted to its frequency domain using FFT, and a mask is applied to filter out high-frequency components based on the compression ratio.
4. **Display Results**: The app displays both the original and compressed images. The compression results will have less detail, but the file size will be smaller.

## Running the Application

1. Clone or download this repository to your local machine.
2. Install the dependencies using the following command:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app using the following command:
    ```bash
    streamlit run image_compressor.py
    ```
4. Open the provided URL in your browser to start using the app.

## Usage

1. **Upload Image**: Click the "Upload an image" button to select an image from your local machine.
2. **Adjust Compression Ratio**: Use the slider to set the compression ratio, which controls how much detail is retained in the compressed image.
3. **View Compressed Image**: The original image and the compressed image will be displayed side by side.

## Example

- Original Image:
  ![Original Image](img/img-1.jpg)

- Compressed Image (Compression Ratio: 0.1):
  ![Compressed Image](compressed_image/compressed_serial1.jpg)

## Limitations

- The compression ratio is applied globally to the entire image, which means finer details in the image will be lost as the compression ratio increases.
- The image quality may degrade as more high-frequency components are removed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of [Streamlit](https://streamlit.io/) for providing a simple interface for building web apps.
- The image processing is based on the Fast Fourier Transform (FFT) algorithm, a widely used technique for frequency analysis.
- Also Special Thanks to [Github joshuapjacob](https://github.com/joshuapjacob/fast-fourier-transform-image-compression) for providing the idea and inspiration to implement Fast Fourier Transform (FFT) image compression in Python.
