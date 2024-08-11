# Image2VideoDiffuser

Transform static images into dynamic video sequences using advanced AI-powered diffusion models.


## Overview

**Image2VideoDiffuser** is a powerful tool that converts static images into dynamic video sequences using state-of-the-art diffusion models. With just a single image, you can generate captivating video animations that breathe life into your visuals. The tool is built using the `DiffusionPipeline` from the `diffusers` library and leverages PyTorch for efficient processing.

## Features

- **Image to Video Conversion:** Transform any image into a smooth video sequence.
- **Customizable Settings:** Adjust frame size, output resolution, FPS, and other parameters to fit your needs.
- **Error Handling & Logging:** Built-in logging and error handling for a robust user experience.
- **Reproducibility:** Set a random seed to ensure consistent results across different runs.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- diffusers
- Transformers
- PIL (Pillow)
- NumPy
- tqdm

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/filipa131/Image2VideoDiffuser.git
   cd Image2VideoDiffuser
   ```

2. **Install dependencies:**
    ```bash
   pip install torch diffusers transformers pillow numpy tqdm
   ```

### Usage

1. **Prepare your image:**
Ensure that your input image (e.g., puppy.png) is available in the project directory or provide the path to the image.

2. **Run the program:**
Run the following command in your terminal or directly within a Python environment like Google Colab:
```bash
python main.py --image_path='images/puppy.png' --output_video='output_videos/puppy.mp4' --frame_size=1024 576 --fps=7 --seed=42
```

3. **Customize parameters:**
- image_path: Path to your input image.
- output_video: The name/path of the output video file.
- frame_size: Desired resolution for the video frames (width, height).
- fps: Frames per second for the output video.
- seed: Random seed for reproducibility.
