import torch
import transformers
import logging
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(image_path='images/puppy.png', output_video='output_videos/puppy.mp4', 
         frame_size=(1024, 576), seed=42, fps=7, decode_chunk_size=8, model_variant='fp16'):
    
    try:
        # Load and set up the diffusion pipeline
        logger.info("Loading diffusion pipeline...")
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant=model_variant
        )
        pipeline.enable_model_cpu_offload()

        # Load and resize the input image
        logger.info("Loading and resizing image...")
        image = load_image(image_path)
        image_resized = image.resize(frame_size, Image.ANTIALIAS)

        # Set the random seed for reproducibility
        generator = torch.manual_seed(seed)
        logger.info(f"Random seed set to: {seed}")

        # Generate video frames with a progress bar
        logger.info("Generating video frames...")
        frames = pipeline(image_resized, decode_chunk_size=decode_chunk_size, generator=generator).frames[0]

        # Export the frames to a video file
        logger.info(f"Exporting video to {output_video}...")
        export_to_video(frames, output_video, fps=fps)
        logger.info("Video generation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
