import os
import xml.etree.ElementTree as ET

def process_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for obj in root.findall('.//object'):
        name_element = obj.find('name')
        if name_element is not None and name_element.text == 'Parking_space':
            print(file_path)
            name_element.text = 'parking_space'

    tree.write(file_path)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            process_xml(file_path)

if __name__ == "__main__":
    folder_path = "12_12_all_images/labels"  # Substitua pelo caminho real para sua pasta
    process_folder(folder_path)
