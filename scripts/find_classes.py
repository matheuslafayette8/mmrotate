import os

# Pasta onde os arquivos estão localizados
pasta = "dataset/global_model/dota/22_02/train/"

# Dicionário para armazenar os datasets onde a classe 'crosswalk_continental' está inclusa e a contagem de ocorrências
datasets_com_crosswalk_continental = {}

# Itera sobre todos os arquivos na pasta
for arquivo in os.listdir(pasta):
    if arquivo.endswith(".txt"):  # Verifica se é um arquivo de texto
        # Lê o conteúdo do arquivo
        with open(os.path.join(pasta, arquivo), 'r') as f:
            linhas = f.readlines()
            # Itera sobre cada linha do arquivo
            for linha in linhas:
                # Separa os elementos da linha
                elementos = linha.strip().split()
                # Pega a classe
                classe = elementos[-2]
                # Pega o nome do dataset do nome do arquivo
                nome_dataset = arquivo.split("_i_")[0]
                # Se a classe for 'crosswalk_continental', adiciona ao contador do dataset
                if classe == 'arrow_forward_right':
                    if nome_dataset not in datasets_com_crosswalk_continental:
                        datasets_com_crosswalk_continental[nome_dataset] = 1
                    else:
                        datasets_com_crosswalk_continental[nome_dataset] += 1

# Imprime os datasets onde a classe 'crosswalk_continental' está inclusa e a contagem de ocorrências
print("Datasets com a classe 'crosswalk_continental':")
for dataset, contagem in datasets_com_crosswalk_continental.items():
    print(f"Dataset: {dataset}, Contagem: {contagem}")
