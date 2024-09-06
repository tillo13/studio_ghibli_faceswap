import os
import time
import torch
import random
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from huggingface_hub import hf_hub_download
import faceid_utils  # Import the faceid_utils module
from enhance_image_via_import import enhance_image
from datetime import datetime

# Python logging for better debug messages
import logging

logging.basicConfig(level=logging.DEBUG)

# Ensure required files are downloaded
faceid_utils.download_required_files()

# Constants
NUMBER_OF_LOOPS = 100
DEFAULT_MODEL = "DucHaiten/DucHaitenAnime"
INCOMING_IMAGES_PATH = "incoming_images"
GENERATED_IMAGES_PATH = "generated_images"
ENHANCED_IMAGES_PATH = "enhanced_images"
MALE_PROMPT = "a highly detailed image of a man in a studio ghibli style environment wearing a suit"
FEMALE_PROMPT = "a highly detailed image of a woman in a studio ghibli style environment wearing a dress"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
SEED = 1060
CFG_SCALE = 1.13
NUMBER_OF_STEPS = 40
RANDOMIZE_SEED_VALUE = True

# Ensure directories exist
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
os.makedirs(ENHANCED_IMAGES_PATH, exist_ok=True)

# Download the IP-Adapter checkpoint if it doesn't exist
v2 = True
ip_ckpt_filename = "ip-adapter-faceid-plusv2_sd15.bin" if v2 else "ip-adapter-faceid-plus_sd15.bin"
if not os.path.exists(ip_ckpt_filename):
    logging.info(f"IP-Adapter checkpoint not found. Downloading...")
    ip_ckpt_filename = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename=ip_ckpt_filename)
logging.info(f"IP-Adapter checkpoint available at: {ip_ckpt_filename}")

# Load Stable Diffusion pipeline
logging.info("Setting up Stable Diffusion pipeline and VAE.")
vae_model_path = "stabilityai/sd-vae-ft-mse"
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    DEFAULT_MODEL,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
).to("cuda")
logging.info(f"Stable Diffusion pipeline set up using model: {DEFAULT_MODEL}")

from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

# Load IP-Adapter
logging.info("Loading IP-Adapter.")
device = "cuda"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt_filename, device)
logging.info("IP-Adapter loaded successfully.")

# Helper functions
def load_image(image_path):
    logging.debug(f"Loading image from {image_path}")
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error loading image from {image_path}: {e}")
        return None

def save_image(image, path):
    logging.debug(f"Saving generated image to {path}")
    image.save(path)

# Helper function to process images
def process_images(image_files, prompt, loop, timestamp):
    global total_images_generated, total_generation_time
    for i, selected_image_path in enumerate(image_files):
        image_start_time = time.time()
        selected_basename = os.path.basename(selected_image_path).split('.')[0]

        # Extract embeddings for the selected image
        logging.debug(f"Extracting embedding for {selected_image_path}")
        embedding, aligned_face = faceid_utils.extract_face_embedding(selected_image_path)

        if embedding is None:
            logging.warning(f"Skipping {selected_image_path} due to failed embedding extraction.")
            continue  # Skip to the next iteration if no face was detected

        if RANDOMIZE_SEED_VALUE:
            random_seed = random.randint(0, 100000)
            generator = torch.manual_seed(random_seed)
            logging.info(f"Generating image {loop * len(image_files) + i + 1} with random seed {random_seed} using {selected_image_path}.")
        else:
            fixed_seed = SEED + loop * len(image_files) + i
            generator = torch.manual_seed(fixed_seed)
            logging.info(f"Generating image {loop * len(image_files) + i + 1} with fixed seed {fixed_seed} using {selected_image_path}.")

        try:
            images = ip_model.generate(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                faceid_embeds=embedding,
                face_image=aligned_face,
                shortcut=v2,
                s_scale=CFG_SCALE,
                num_samples=1,
                width=1024,
                height=1024,
                num_inference_steps=NUMBER_OF_STEPS,
                seed=random_seed if RANDOMIZE_SEED_VALUE else fixed_seed
            )
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            continue

        # Create a more precise timestamp for uniqueness
        precise_timestamp = f"{timestamp}_{time.time()*1000:.0f}"
        result_image_path = f"{GENERATED_IMAGES_PATH}/{faceid_utils.sanitize_filename(selected_basename)}_{faceid_utils.sanitize_filename(DEFAULT_MODEL)}_{precise_timestamp}.png"
        save_image(images[0], result_image_path)
        logging.info(f"Generated image {loop * len(image_files) + i + 1} saved at {result_image_path}")

        # Enhance the generated image
        try:
            enhance_image(result_image_path, ENHANCED_IMAGES_PATH)
            logging.info(f"Enhanced image {loop * len(image_files) + i + 1} and saved to {ENHANCED_IMAGES_PATH}")
        except Exception as e:
            logging.error(f"Error enhancing image: {result_image_path}, error: {e}")

        total_images_generated += 1
        elapsed_time = time.time() - image_start_time
        total_generation_time += elapsed_time

        average_time_per_image = total_generation_time / total_images_generated
        estimated_time_remaining = average_time_per_image * (NUMBER_OF_LOOPS * len(image_files) - total_images_generated)

        logging.info(f"Elapsed time for current image: {elapsed_time:.2f} seconds")
        logging.info(f"Total images generated: {total_images_generated}")
        logging.info(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")

# List all image files in incoming_images and divide them into male and female subfolders
logging.debug("Listing all image files...")
male_image_files = [os.path.join(INCOMING_IMAGES_PATH, "male", f) for f in os.listdir(os.path.join(INCOMING_IMAGES_PATH, "male")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
female_image_files = [os.path.join(INCOMING_IMAGES_PATH, "female", f) for f in os.listdir(os.path.join(INCOMING_IMAGES_PATH, "female")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not male_image_files and not female_image_files:
    raise ValueError(f"No images found in {INCOMING_IMAGES_PATH}/male or {INCOMING_IMAGES_PATH}/female.")

timestamp = time.strftime("%H%M%S")

total_start_time = time.time()
total_images_generated = 0
total_generation_time = 0

# Generate images by looping NUMBER_OF_LOOPS times over all images in incoming_images
for loop in range(NUMBER_OF_LOOPS):
    logging.info(f"Starting loop {loop + 1} of {NUMBER_OF_LOOPS}")

    # Process male images
    logging.info("Processing male images...")
    process_images(male_image_files, MALE_PROMPT, loop, timestamp)

    # Process female images
    logging.info("Processing female images...")
    process_images(female_image_files, FEMALE_PROMPT, loop, timestamp)

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time
average_time_per_image = total_elapsed_time / total_images_generated

# Summary
logging.info("=== SUMMARY ===")
logging.info(f"Total images generated: {total_images_generated}")
logging.info(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
logging.info(f"Average time per image: {average_time_per_image:.2f} seconds")

logging.info("All images have been generated and enhanced.")