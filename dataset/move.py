import os
import shutil

def copy_files(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        
        # Verifica se o arquivo começa com "pagemill" ou "santaclara"
        if filename.startswith(("pagemill", "santaclara")) and os.path.isfile(source_path):
            destination_path = os.path.join(destination_folder, filename)
            
            # Copia o arquivo para o destino
            shutil.copy(source_path, destination_path)
            print(f"Copiado: {filename}")

if __name__ == "__main__":
    source_folder = "/home/luminar/mmrotate/dataset/withNewImagesAndAllFalsePositives/labels"
    destination_folder = "/home/luminar/mmrotate/dataset/falsePositives/labels"

    copy_files(source_folder, destination_folder)