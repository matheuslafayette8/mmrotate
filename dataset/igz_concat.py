import cv2
import os
import numpy as np

base_path = "dataset/global_model/600/sub_datasets/"
datasets_name = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

for dataset_name in datasets_name:

    base_path = f"dataset/global_model/600/sub_datasets/{dataset_name}/"

    low_i_view_folder = base_path + "low_i_imgs/"
    g_view_folder = base_path + "g_imgs/"
    high_i_view_folder = base_path + "high_i_imgs/"
    output_path = base_path + "low_high_igi_imgs/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    low_i_view_files = os.listdir(low_i_view_folder)
    g_view_files = os.listdir(g_view_folder)
    high_i_view_files = os.listdir(high_i_view_folder)

    low_i_view_files.sort()
    g_view_files.sort()
    high_i_view_files.sort()

    for low_i_file, g_file, high_i_file in zip(low_i_view_files, g_view_files, high_i_view_files):

        low_i_image = cv2.imread(os.path.join(low_i_view_folder, low_i_file), cv2.IMREAD_GRAYSCALE)
        g_image = cv2.imread(os.path.join(g_view_folder, g_file), cv2.IMREAD_GRAYSCALE)
        high_i_image = cv2.imread(os.path.join(high_i_view_folder, high_i_file), cv2.IMREAD_GRAYSCALE)
        
        merged_image = np.zeros((low_i_image.shape[0], low_i_image.shape[1], 3))

        merged_image[:, :, 0] = low_i_image
        merged_image[:, :, 1] = g_image
        merged_image[:, :, 2] = high_i_image
        
        cv2.imwrite(os.path.join(output_path, low_i_file), merged_image)