import os

def find_missing_xml_png(xml_folder, png_folder):
    # Lista de arquivos PNG na pasta png_folder
    png_files = [file.split('.')[0] for file in os.listdir(png_folder) if file.endswith('.png')]

    # Lista de arquivos XML na pasta xml_folder
    xml_files = [file.split('.')[0] for file in os.listdir(xml_folder) if file.endswith('.xml')]

    # Encontra os arquivos PNG que não têm um correspondente na pasta XML
    missing_png_files = [png for png in png_files if png not in xml_files]

    return missing_png_files

# Caminhos dos diretórios
xml_folder_path = "/persistent/mmrotate/global_model/metrics/labels"
png_folder_path = "/persistent/mmrotate/global_model/metrics/imgs"

# Encontra os arquivos PNG que não têm correspondente na pasta XML
missing_png_files = find_missing_xml_png(xml_folder_path, png_folder_path)

# Imprime os nomes dos arquivos PNG que não têm correspondente na pasta XML
print("Arquivos PNG sem correspondente na pasta de XML:")
for file in missing_png_files:
    print(file + ".png")
