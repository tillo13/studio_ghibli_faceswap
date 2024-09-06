import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from huggingface_hub import hf_hub_download
from faceid_utils import extract_face_embedding, add_padding
import requests

# Directories Setup
input_dirs = {
    "male": "./incoming_images/male",
    "female": "./incoming_images/female"
}
output_dir = "./model_posed"

os.makedirs(output_dir, exist_ok=True)
os.makedirs("ip_adapter", exist_ok=True)

# Download necessary files from GitHub if they're missing
def download_file_from_github(repo_url, file_path, save_dir):
    file_url = f"{repo_url}/raw/main/{file_path}"
    local_path = os.path.join(save_dir, file_path.replace('/', os.sep))
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {file_path} from GitHub...")
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_path} to {local_path}")
        else:
            raise Exception(f"Error downloading {file_path}: {response.status_code}")
    else:
        print(f"{file_path} already exists in {save_dir}. Skipping download.")
    return local_path

# Constants
DEFAULT_MODEL = "DucHaiten/DucHaitenAnime"
MALE_PROMPT = "a detailed anime-style portrait of a man with expressive facial details in studio ghibli style"
FEMALE_PROMPT = "a detailed anime-style portrait of a woman with expressive facial details in studio ghibli style"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
CFG_SCALE = 1.13
NUMBER_OF_STEPS = 40
GITHUB_REPO = "https://github.com/tencent-ailab/IP-Adapter"
REQUIRED_FILES = [
    "ip_adapter/__init__.py",
    "ip_adapter/attention_processor.py",
    "ip_adapter/attention_processor_faceid.py",
    "ip_adapter/custom_pipelines.py",
    "ip_adapter/ip_adapter.py",
    "ip_adapter/ip_adapter_faceid.py",
    "ip_adapter/ip_adapter_faceid_separate.py",
    "ip_adapter/resampler.py",
    "ip_adapter/test_resampler.py",
    "ip_adapter/utils.py"
]

# Download required files from GitHub
for file in REQUIRED_FILES:
    download_file_from_github(GITHUB_REPO, file, ".")

# Download the IP-Adapter checkpoint if it doesn't exist
v2 = True
ip_ckpt_filename = "ip-adapter-faceid-plusv2_sd15.bin" if v2 else "ip-adapter-faceid-plus_sd15.bin"
if not os.path.exists(ip_ckpt_filename):
    print(f"IP-Adapter checkpoint not found. Downloading...")
    ip_ckpt_filename = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename=ip_ckpt_filename)

# Load Stable Diffusion pipeline
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

# Load IP-Adapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

device = "cuda"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt_filename, device)

# Functions for processing images
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def save_image(image, path):
    image.save(path)

def resize_and_pad(image, target_size):
    original_size = image.size
    ratio = float(target_size) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    image = image.resize(new_size, Image.LANCZOS)
    new_image = Image.new("RGB", (target_size, target_size))
    new_image.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))
    return new_image

def process_image(img_path, gender, ip_model, prompts, output_dir):
    img_filename = os.path.basename(img_path)
    base_image = load_image(img_path)

    # Extract face embedding
    embedding, aligned_face = extract_face_embedding(img_path)
    if embedding is None:
        print(f"Skipping {img_filename} due to failed embedding extraction.")
        return

    # Resize and pad image
    base_image = resize_and_pad(base_image, 512)

    # Set seed for consistency if needed
    generator = torch.manual_seed(1060)

    # Generate image using IP-Adapter with face embedding
    images = ip_model.generate(
        prompt=prompts[gender],
        negative_prompt=NEGATIVE_PROMPT,
        faceid_embeds=embedding,
        face_image=aligned_face,
        shortcut=v2,
        s_scale=CFG_SCALE,
        num_samples=1,
        width=512,
        height=512,
        num_inference_steps=NUMBER_OF_STEPS,
        seed=1060
    )

    stylized_image_path = os.path.join(output_dir, f"anime_style_{img_filename}")
    save_image(images[0], stylized_image_path)

if __name__ == "__main__":
    prompts = {
        "male": MALE_PROMPT,
        "female": FEMALE_PROMPT
    }

    for gender, input_dir in input_dirs.items():
        print(f"Processing {gender} images from {input_dir}...")
        for img_filename in os.listdir(input_dir):
            if img_filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                input_image_path = os.path.join(input_dir, img_filename)
                process_image(input_image_path, gender, ip_model, prompts, output_dir)