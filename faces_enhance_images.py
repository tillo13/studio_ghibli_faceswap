import os
import cv2
import requests
from PIL import Image, ImageEnhance, ExifTags
import numpy as np
import time
import shutil
import psutil
from datetime import datetime
from mtcnn import MTCNN
import dlib

#### GLOBAL IMAGE VALUES YOU CAN EDIT ####

# TARGET_SIZE:
# This controls the maximum size of the longest dimension of the image (width or height).
# Any image dimension exceeding this value will be resized down to this value, maintaining aspect ratio.
TARGET_SIZE = 2048

# DENOISE_STRENGTH:
# These four values control the strength of the denoising process applied to the face regions.
# They are (h, hForColorComponents, templateWindowSize, searchWindowSize):
# - h: Filter strength for luminance component.
# - hForColorComponents: Same as h but for color images, usually equals h.
# - templateWindowSize: Size in pixels of the template patch used for denoising.
# - searchWindowSize: Size in pixels of the window used to compute weighted average for given pixel.
DENOISE_STRENGTH = (3, 3, 7, 21)

# SHARPEN_AMOUNT:
# This value controls the intensity/amount of sharpening applied using the unsharp mask.
# Lower values yield subtler sharpening; higher values yield stronger sharpening.
SHARPEN_AMOUNT = 0.5

# SHARPEN_SIGMA:
# This value is the standard deviation for Gaussian kernel used in the unsharp mask.
# It determines how much the image is blurred before applying the sharpening.
SHARPEN_SIGMA = 1.0

# SHARPEN_KERNEL_SIZE:
# This tuple represents the width and height of the Gaussian kernel used in the unsharp mask.
# Larger values yield more blurred image used for sharpening.
SHARPEN_KERNEL_SIZE = (5, 5)

# SHARPNESS_ENHANCE:
# This value controls the overall sharpness enhancement applied to the entire image.
# Values greater than 1 enhance sharpness, values less than 1 reduce it.
SHARPNESS_ENHANCE = 1.01

# CONTRAST_ENHANCE:
# This value controls the overall contrast enhancement applied to the entire image.
# Values greater than 1 enhance contrast, values less than 1 reduce it.
CONTRAST_ENHANCE = 1.05

# COLOR_ENHANCE:
# This value controls the overall color enhancement/saturation applied to the entire image.
# Values greater than 1 make colors more vibrant, values less than 1 make them more muted.
COLOR_ENHANCE = 1.05

##########################################

# Add Comparisons: Turn on/off image comparison functionality
ADD_COMPARISONS = False  # Set to False to turn off comparison images

# Save Original: Turn on/off saving of the original image
SAVE_ORIGINAL = False  # Set to False to not save the original unedited image

def memory_available():
    return psutil.virtual_memory().available

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print(f'Downloaded {local_path}')
    else:
        print(f'Failed to download {url}')

def denoise_image(img_cv):
    print('Applying subtle denoising...')
    h, hForColorComponents, templateWindowSize, searchWindowSize = DENOISE_STRENGTH
    return cv2.fastNlMeansDenoisingColored(img_cv, None, h, hForColorComponents, templateWindowSize, searchWindowSize)

def unsharp_mask(img_cv, kernel_size=SHARPEN_KERNEL_SIZE, sigma=SHARPEN_SIGMA, amount=SHARPEN_AMOUNT, threshold=0):
    print('Applying unsharp mask for subtle sharpening...')
    blurred = cv2.GaussianBlur(img_cv, kernel_size, sigma)
    sharpened = float(amount + 1) * img_cv - float(amount) * blurred
    sharpened = np.maximum(sharpened, 0)
    sharpened = np.minimum(sharpened, 255)
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img_cv - blurred) < threshold
        np.copyto(sharpened, img_cv, where=low_contrast_mask)
    return sharpened

def safe_save(filepath, img):
    """Save the image ensuring no overwrites. Append timestamp if file already exists."""
    base, ext = os.path.splitext(filepath)
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_base = f"{timestamp}_{os.path.basename(base)}"
        filepath = os.path.join(os.path.dirname(base), new_base + ext)
    img.save(filepath)
    print(f'Saved: {filepath}')

def process_image(filepath, face_detector, landmark_predictor):
    try:
        print(f'Opening image: {filepath}')
        with Image.open(filepath) as img:
            try:
                # Handle image rotation based on EXIF tags
                print('Checking for EXIF orientation data...')
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    exif = dict(exif.items())
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        print('Rotating image 180 degrees due to EXIF orientation...')
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        print('Rotating image 270 degrees due to EXIF orientation...')
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        print('Rotating image 90 degrees due to EXIF orientation...')
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                print('No EXIF orientation data found or error reading EXIF data.')
                pass

            orig_size = img.size
            print(f'Original size: {orig_size} ({os.path.getsize(filepath) / (1024 * 1024):.2f} MB)')

            # Default unedited_filepath to the original image path
            if SAVE_ORIGINAL:
                unedited_filepath = os.path.join('results', f"{os.path.splitext(os.path.basename(filepath))[0]}_unedited.png")
                os.makedirs('results', exist_ok=True)
                safe_save(unedited_filepath, img)
            else:
                unedited_filepath = None

            # Convert image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            print('Converted image to OpenCV (BGR) format.')

            # Now, we will use MTCNN to detect faces
            print('Detecting faces using MTCNN...')
            faces = face_detector.detect_faces(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            print(f'Number of faces detected: {len(faces)}')

            # Process each detected face
            for face in faces:
                x, y, w, h = face['box']
                face_img = img_cv[y:y+h, x:x+w]
                print(f'Processing face region: x={x}, y={y}, width={w}, height={h}')

                # Subtle denoising
                face_img = denoise_image(face_img)
                
                # Facial landmarks
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
                landmarks = landmark_predictor(gray, rect)

                # Subtle sharpening
                face_img = unsharp_mask(face_img)

                img_cv[y:y+h, x:x+w] = face_img

            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            print('Converted image back to PIL format.')

            # Apply general subtle enhancements to the entire image
            img = ImageEnhance.Sharpness(img).enhance(SHARPNESS_ENHANCE)
            print('Applied general subtle sharpness enhancement.')
            img = ImageEnhance.Contrast(img).enhance(CONTRAST_ENHANCE)
            print('Applied general subtle contrast enhancement.')
            img = ImageEnhance.Color(img).enhance(COLOR_ENHANCE)
            print('Applied general subtle color enhancement.')

            # Resize the final image to ensure the longest side is at most target_size
            if max(img.size) > TARGET_SIZE:
                scale_factor = TARGET_SIZE / max(img.size)
                new_dimensions = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                img = img.resize(new_dimensions, Image.LANCZOS)
                print(f'Resized final image to ensure longest side is at most {TARGET_SIZE}: {img.size}')

            # Save enhanced image to results directory with _enhanced appended to filename
            os.makedirs('results', exist_ok=True)
            enhanced_filepath = os.path.join('results', f"{os.path.splitext(os.path.basename(filepath))[0]}_enhanced.png")
            safe_save(enhanced_filepath, img)

            return img, unedited_filepath, enhanced_filepath

    except IOError as e:
        print(f"Error processing file {filepath}: {e}, skipping...")
        return None

def main():
    image_directory = '.'

    face_predictor_path = 'shape_predictor_68_face_landmarks.dat'
    
    if not os.path.exists(face_predictor_path):
        download_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        print(f"{face_predictor_path} not found. Downloading from {download_url}...")
        download_file(download_url, face_predictor_path)
    
    landmark_predictor = dlib.shape_predictor(face_predictor_path)
    face_detector = MTCNN()

    all_files = [f for f in sorted(os.listdir(image_directory)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    processed_files = 0
    processing_times = []

    print('Starting to process images...')

    for filename in all_files:
        old_filepath = os.path.join(image_directory, filename)
        new_filepath_comparison = os.path.join('results', f"{os.path.splitext(os.path.basename(filename))[0]}_comparison.png")

        print(f"\nProcessing file: {filename}...")
        start_time = time.time()

        result = process_image(old_filepath, face_detector, landmark_predictor)
        if result:
            processed_img, unedited_filepath, enhanced_filepath = result
            if ADD_COMPARISONS and unedited_filepath:
                # Make comparison image
                with Image.open(unedited_filepath) as orig_img:
                    common_height = min(orig_img.height, processed_img.height)
                    orig_img_resized = orig_img.resize((int(orig_img.width * common_height / orig_img.height), common_height), Image.LANCZOS)
                    processed_img_resized = processed_img.resize((int(processed_img.width * common_height / processed_img.height), common_height), Image.LANCZOS)

                    comparison_img = Image.new('RGB', (orig_img_resized.width + processed_img_resized.width, common_height))
                    comparison_img.paste(orig_img_resized, (0, 0))
                    comparison_img.paste(processed_img_resized, (orig_img_resized.width, 0))

                    safe_save(new_filepath_comparison, comparison_img)

        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        processed_files += 1

        print(f"Time taken for {filename}: {processing_time:.2f} seconds")
        remaining_images = len(all_files) - processed_files
        avg_time_per_image = sum(processing_times) / len(processing_times)
        estimated_time_left = avg_time_per_image * remaining_images
        estimated_minutes = int(estimated_time_left // 60)
        estimated_seconds = int(estimated_time_left % 60)
        print(f"Estimated time to completion: {estimated_minutes} mins {estimated_seconds} seconds")

    total_processing_time = sum(processing_times)
    total_minutes = int(total_processing_time // 60)
    total_seconds = int(total_processing_time % 60)
    print(f'Total images processed: {processed_files}')
    print(f'Total time taken: {total_minutes} mins {total_seconds} seconds')

if __name__ == '__main__':
    main()