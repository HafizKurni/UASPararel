# Image Compressor with FFT
This project is part of the **UAS Project for Sistem Pararel dan Terdistribusi**

## Group Members:
1. **Hafiz Muhammad Kurniawan** - 09021182126003
2. **Wisnu Wardana** - 09021282126086

## Overview

This program allows users to upload an image and apply compression using Fast Fourier Transform (FFT). The user can adjust the compression ratio through a slider, and the app will display both the original and compressed images. The compression is done in the frequency domain by filtering out high-frequency components of the image, leading to a loss of finer details but smaller file sizes.

## Features

- **Image Upload**: Users can upload an image file in `.jpg`, `.jpeg`, or `.png` format.
- **Compression Ratio Slider**: Adjust the compression ratio using a slider to control the amount of high-frequency components to retain.
- **Real-time Image Display**: The app will display both the original and compressed images after applying the selected compression.
- **Frequency Domain Compression**: Uses FFT to convert the image into its frequency components and applies a mask to retain only a fraction of the frequency data.

## Requirements

This program is built using the following libraries:
- `numpy`: For numerical operations, particularly FFT.
- `opencv-python`: For image processing and frequency domain manipulation.
- `Pillow`: For image handling and conversion.

To install the required dependencies, run:

```bash
pip install numpy opencv-python Pillow
```

## Example

- Original Image:
  ![Original Image](img/img-1B.jpg)

- Compressed Image (Compression Ratio: 0.1):
  ![Compressed Image](compressed_image/compressed_serial_1.jpg)

## Limitations

- The compression ratio is applied globally to the entire image, which means finer details in the image will be lost as the compression ratio increases.
- The image quality may degrade as more high-frequency components are removed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The image processing is based on the Fast Fourier Transform (FFT) algorithm, a widely used technique for frequency analysis.
- Also Special Thanks to [Github joshuapjacob](https://github.com/joshuapjacob/fast-fourier-transform-image-compression) for providing the idea and inspiration to implement Fast Fourier Transform (FFT) image compression in Python.
