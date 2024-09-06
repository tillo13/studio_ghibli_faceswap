# Studio Ghibli Cartoon Face Likeness App

Welcome to the Studio Ghibli Cartoon Face Likeness App! This app processes images of faces and enhances them by giving them a Studio Ghibli-like cartoon effect. Below, you'll find detailed information about the project structure, the Python scripts it includes, and how to set up and run the application.

## Project Structure

The project consists of 7 Python files and 8 directories. Here is the directory structure:

```plaintext
.
|-- __pycache__
|-- incoming_images
|   |-- incoming_images/female
|   |-- incoming_images/male
|-- model_posed
|-- ip_adapter
|   |-- ip_adapter/__pycache__
List of Python File Paths
.\enhance_image_via_import.py
.\faceid_utils.py
.\faces_enhance_images.py
.\gather_pythons.py
.\ghibli_profile_portraits_using_faceid_utils.py
.\loop_random_image_ghibli_using_faceid_utils.py
.\old_enhance_images.py
Descriptions of Python Files
.\enhance_image_via_import.py
This script enhances an input image by denoising it, applying unsharp mask, adjusting sharpness, contrast, and color. The enhanced image is then saved. You can find the full implementation in the .\enhance_image_via_import.py file.

.\faceid_utils.py
Contains utilities for face identification, including functions for downloading required files, adding padding to images, and extracting face embeddings. You can see the details in the .\faceid_utils.py file.

.\faces_enhance_images.py
Processes a set of images by enhancing and optionally comparing them, handling EXIF orientation, and applying facial landmarks. Full implementation is available in the .\faces_enhance_images.py file.

.\gather_pythons.py
Gathers all .py files within the specified root directory and its subdirectories, excluding specified directories, and writes the gathered data to a file. Check the implementation in the .\gather_pythons.py file.

.\ghibli_profile_portraits_using_faceid_utils.py
This script generates anime-style portraits using IP-Adapter and Stable Diffusion Pipeline. It processes images of males and females separately. The complete code is available in the .\ghibli_profile_portraits_using_faceid_utils.py file.

.\loop_random_image_ghibli_using_faceid_utils.py
Runs a loop to generate a specified number of images with Studio Ghibli-style enhancements and then optionally enhances them further. You can find the full details in the .\loop_random_image_ghibli_using_faceid_utils.py file.

.\old_enhance_images.py
Enhances images using a DNN-based super-resolution model and applies sharpness, contrast, and color enhancements. Detailed implementation is in the .\old_enhance_images.py file.

### How to Run the App

1. **Install Dependencies**

   Make sure you have Python installed. Then, install the required dependencies using the following command:

   ```bash
   pip install -r requirements.txt
Prepare Input Images

Place the images you want to process in the incoming_images/female or incoming_images/male directories.

Run the Scripts

You can run each script from the command line. For example, to run the main enhancement script, use:

python enhance_image_via_import.py
Similarly, you can run the other scripts as needed.

Check Output

The enhanced images and any generated comparisons will be saved in the appropriate output directories as specified in the scripts.

We hope you find this app useful and easy to use. Enjoy enhancing your images with Studio Ghibli-like cartoon effects!