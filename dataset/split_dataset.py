import os
import random
import shutil
import argparse

def split_dataset(dataset, model, train_percentage, random_seed):
    # Configuração dos caminhos
    input_folder = f"dataset/{model}/600/dota/{dataset}/"
    train_output_folder = f"dataset/{model}/600/dota/{dataset}/train/"
    val_output_folder = f"dataset/{model}/600/dota/{dataset}/val/"

    # Criar pastas de treino e validação
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(val_output_folder, exist_ok=True)

    # Listar arquivos de imagens na pasta de entrada
    image_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]
    
    # Definir semente para reproduzibilidade
    random.seed(random_seed)

    # Embaralhar a lista de arquivos
    random.shuffle(image_files)

    # Calcular o número de amostras para treino
    num_train_samples = int(train_percentage * len(image_files))

    # Separar os arquivos de imagem e labels para treino e validação
    train_files = image_files[:num_train_samples]
    val_files = image_files[num_train_samples:]

    for file in train_files:
        # Mover arquivos de imagem para a pasta de treino
        shutil.move(os.path.join(input_folder, file), os.path.join(train_output_folder, file))
        # Mover arquivos de labels para a pasta de treino
        label_file = file.replace(".png", ".txt")
        shutil.move(os.path.join(input_folder, label_file), os.path.join(train_output_folder, label_file))

    for file in val_files:
        # Mover arquivos de imagem para a pasta de validação
        shutil.move(os.path.join(input_folder, file), os.path.join(val_output_folder, file))
        # Mover arquivos de labels para a pasta de validação
        label_file = file.replace(".png", ".txt")
        shutil.move(os.path.join(input_folder, label_file), os.path.join(val_output_folder, label_file))

if __name__ == "__main__":
    # Configurar argparse para receber argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Split dataset into train and val sets.")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--train_percentage", type=float, required=True, help="Train percentage")

    args = parser.parse_args()

    # Chamar a função para dividir o dataset
    split_dataset(args.dataset, args.model, args.train_percentage, random_seed=42)
