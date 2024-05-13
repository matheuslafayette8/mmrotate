import os
import shutil

def downsample_images(source_folder, destination_folder, sample_rate):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_files = [f for f in sorted(os.listdir(source_folder))]
    selected_images = image_files[::sample_rate]

    copied_count = 0

    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.copy(source_path, destination_path)
        copied_count += 1
    
    print(f"New downsampled batch size: {copied_count}")

source_folder = "falsePositives/imgs"
destination_folder = "01-11_withNewIndiaDownSampled/imgs"
sample_rate = 5

downsample_images(source_folder, destination_folder, sample_rate)