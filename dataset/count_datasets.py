import os

diretorio = "dataset/complete_datasets/05_01/imgs"
datasets_qtd = {}

for filename in os.listdir(diretorio):
    if filename.endswith(".png"):
        partes_nome = filename.split("_i_")
        name_dataset = partes_nome[0]
        if name_dataset not in datasets_qtd:
            datasets_qtd[name_dataset] = 1
        else:
            datasets_qtd[name_dataset] += 1

for dataset, qtd in sorted(datasets_qtd.items()):
    print(f"{dataset}: {qtd}")
