import os
import cv2
from PIL import Image, ImageEnhance, ExifTags
import numpy as np
import time
import shutil
import psutil

def memory_available():
    return psutil.virtual_memory().available

def process_image(filepath, sr_model, initial_try_size=2048, min_sr_size=1024, shrink_factor=0.9, target_dim=2048, min_allowable_size=512):
    try:
        with Image.open(filepath) as img:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    exif = dict(exif.items())
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            orig_size = img.size
            print(f'Original size: {orig_size} ({os.path.getsize(filepath) / (1024 * 1024):.2f} MB)')

            if max(img.size) > initial_try_size:
                scale_factor = initial_try_size / max(img.size)
                img = img.resize((int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)), Image.LANCZOS)

            while max(img.size) > min_sr_size:
                try:
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    if memory_available() < img_cv.nbytes * 2:
                        raise MemoryError("Not enough memory to proceed with super-resolution.")
                    print('Starting super-resolution...')
                    img_cv = sr_model.upsample(img_cv)
                    print('Super-resolution completed.')
                    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    break
                except (cv2.error, MemoryError) as e:
                    print(f"Super-resolution failed: {e}. Retrying with smaller size...")
                    initial_try_size = int(initial_try_size * shrink_factor)
                    if initial_try_size < min_allowable_size:
                        print('Reached minimum allowable size. Skipping super-resolution.')
                        break
                    scale_factor = initial_try_size / max(img.size)
                    img = img.resize((int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)), Image.LANCZOS)

            if min(img.size) < target_dim:
                scale_factor = target_dim / min(img.size)
                img = img.resize((int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)), Image.LANCZOS)
                print(f'Upscaled to target size: {img.size} after super-resolution.')

            img = ImageEnhance.Sharpness(img).enhance(1.2)
            print('Applied general sharpness enhancement.')
            img = ImageEnhance.Contrast(img).enhance(1.1)
            print('Applied general contrast enhancement.')
            img = ImageEnhance.Color(img).enhance(1.1)
            print('Applied general color enhancement.')

            return img

    except IOError as e:
        print(f"Error processing file {filepath}: {e}, skipping...")
        return None

def main():
    image_directory = '.'
    initial_try_size = 2048
    min_sr_size = 1024
    shrink_factor = 0.9
    target_size = 2048

    path_to_lap_srn_model = r"C:\kumori\prod\tools\enhance_images\LapSRN_x8.pb"
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path_to_lap_srn_model)
    sr.setModel('lapsrn', 8)

    all_files = [f for f in sorted(os.listdir(image_directory)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    processed_files = 0
    processing_times = []

    print('Starting to process images...')

    for filename in all_files:
        old_filepath = os.path.join(image_directory, filename)
        new_filename = f'{os.path.splitext(filename)[0]}_enhanced{os.path.splitext(filename)[1]}'
        new_filepath = os.path.join(image_directory, new_filename)

        print(f"\nProcessing file: {filename}...")
        start_time = time.time()

        processed_img = process_image(old_filepath, sr, initial_try_size, min_sr_size, shrink_factor, target_size)
        if processed_img:
            processed_img.save(new_filepath, 'PNG')

            unedited_images_directory = os.path.join(image_directory, "unedited_images")
            os.makedirs(unedited_images_directory, exist_ok=True)
            shutil.move(old_filepath, os.path.join(unedited_images_directory, filename))

            comparison_directory = os.path.join(image_directory, "comparisons")
            os.makedirs(comparison_directory, exist_ok=True)

            with Image.open(os.path.join(unedited_images_directory, filename)) as orig_img:
                common_height = min(orig_img.height, processed_img.height)
                orig_img_resized = orig_img.resize((int(orig_img.width * common_height / orig_img.height), common_height), Image.LANCZOS)
                processed_img_resized = processed_img.resize((int(processed_img.width * common_height / processed_img.height), common_height), Image.LANCZOS)

                comparison_img = Image.new('RGB', (orig_img_resized.width + processed_img_resized.width, common_height))
                comparison_img.paste(orig_img_resized, (0, 0))
                comparison_img.paste(processed_img_resized, (orig_img_resized.width, 0))

                comparison_filepath = os.path.join(comparison_directory, f'{os.path.splitext(filename)[0]}_comparison{os.path.splitext(filename)[1]}')
                comparison_img.save(comparison_filepath, 'PNG')
                print(f'Comparison image saved as {comparison_filepath}')

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