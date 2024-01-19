import os
import shutil
import xml.etree.ElementTree as ET

# Diretórios de origem e destino
origem_dir = "/home/openmmlab/mmrotate/dataset/26_12_negatives_iris_plus/down_west_gate_route_1/imgs"
#destino_imgs_dir = "/home/matheuslafayette/lane-vec-end-to-end/mmrotate/dataset/falsePositives/imgs"
destino_labels_dir = "/home/openmmlab/mmrotate/dataset/26_12_negatives_iris_plus/down_west_gate_route_1/labels"

# Itera sobre os arquivos no diretório de origem
for filename in os.listdir(origem_dir):
    if filename.endswith(".png"):
        # Constrói os caminhos completos
        origem_path = os.path.join(origem_dir, filename)
        
        # Copia a imagem para o diretório de destino
        #shutil.copy(origem_path, destino_img_path)
        print(f"Copiada imagem: {filename}")
        
        # Cria o nome do arquivo XML correspondente
        xml_filename = os.path.splitext(filename)[0] + ".xml"
        destino_xml_path = os.path.join(destino_labels_dir, xml_filename)
        
        # Parse do XML modelo
        tree = ET.parse("/home/openmmlab/mmrotate/dataset/26_12_with_iris_plus/labels/church_candidate_1_i_3_img.xml")  # Substitua "modelo.xml" pelo caminho real do seu arquivo modelo XML
        root = tree.getroot()
        
        # Atualiza o folder no XML
        folder_elem = root.find("folder")
        folder_elem.text = filename.split("bag")[0]
        
        # Atualiza o filename e o path no XML
        filename_elem = root.find("filename")
        filename_elem.text = filename
        path_elem = root.find("path")
        path_elem.text = origem_path
        
        # Salva o novo XML no diretório de destino
        tree.write(destino_xml_path)
        print(f"Criado arquivo XML: {xml_filename}")
