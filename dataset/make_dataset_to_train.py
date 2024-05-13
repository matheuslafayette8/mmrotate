import cv2
import os
import shutil

base_path = "dataset/global_model/600/sub_datasets/"
datasets_name = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

dataset_name = "02_05"
output_path = f"dataset/global_model/600/complete_datasets/{dataset_name}/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "imgs"))
    os.makedirs(os.path.join(output_path, "labels"))

for dataset_name in datasets_name:
    
    base_path = f"dataset/global_model/600/sub_datasets/{dataset_name}/"

    igi_imgs_folder = base_path + "low_high_igi_imgs/"
    labels_folder = base_path + "labels/"

    igi_files = os.listdir(igi_imgs_folder)
    igi_files.sort()

    for igi_file in igi_files:
        igi_name, igi_ext = os.path.splitext(igi_file)
        label_file = igi_name + ".xml"
        label_path = os.path.join(labels_folder, label_file)

        if os.path.exists(label_path):
            shutil.copy(os.path.join(igi_imgs_folder, igi_file), os.path.join(output_path, "imgs", f"{dataset_name}_{igi_file}"))
            shutil.copy(label_path, os.path.join(output_path, "labels", f"{dataset_name}_{label_file}"))