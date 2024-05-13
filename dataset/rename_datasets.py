import os
import shutil

dataset = "iris_plus_sphere_route_3"

# Diretórios de origem e destino
diretorio_origem = "sub_datasets/03_01/bev_imgs_sphere_route_3/"
pasta_png_destino = "complete_datasets/03_01/imgs"
pasta_xml_destino = "complete_datasets/03_01/labels"

# Prefixo a ser adicionado
prefixo = dataset + "_"

# Cria as pastas de destino se não existirem
os.makedirs(pasta_png_destino, exist_ok=True)
os.makedirs(pasta_xml_destino, exist_ok=True)

# Itera sobre os arquivos no diretório de origem
for filename in os.listdir(diretorio_origem):
    # Caminho completo do arquivo de origem
    caminho_origem = os.path.join(diretorio_origem, filename)
    pasta_destino = pasta_png_destino if filename.endswith(".png") else pasta_xml_destino

    novo_nome = f"{prefixo}{filename}"
        
    # Copia o arquivo para a pasta_destino
    shutil.copy2(caminho_origem, os.path.join(pasta_destino, novo_nome))
    print(f"Copiado {filename} para {pasta_destino} como {novo_nome}")

print("Concluído.")