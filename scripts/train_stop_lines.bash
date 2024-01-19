#!/bin/bash

MMROTATE_DIR="/home/openmmlab/mmrotate"
cd $MMROTATE_DIR

DATASET="17_01"
MODEL="stop_lines"
TRAIN_PERCENTAGE=0.7

# Convert labels
python dataset/convert_labels_sl.py --dataset $DATASET

# Split dataset
python dataset/split_dataset.py --dataset $DATASET --model $MODEL --train_percentage $TRAIN_PERCENTAGE

# Remover a pasta data/split_ms_dota
rm -r data/split_ms_dota

# Atualizar o JSON para o conjunto de validação
jq '.img_dirs = ["dataset/'"$MODEL"'/dota/'"$DATASET"'/val"]' tools/data/dota/split/split_configs/ms_train.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_train.json
jq '.ann_dirs = ["dataset/'"$MODEL"'/dota/'"$DATASET"'/val"]' tools/data/dota/split/split_configs/ms_train.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_train.json

# Atualizar o JSON para o conjunto de treinamento
jq '.img_dirs = ["dataset/'"$MODEL"'/dota/'"$DATASET"'/train"]' tools/data/dota/split/split_configs/ms_val.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_val.json
jq '.ann_dirs = ["dataset/'"$MODEL"'/dota/'"$DATASET"'/train"]' tools/data/dota/split/split_configs/ms_val.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_val.json

# Executar o script de divisão de imagens
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ms_train.json
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ms_val.json

# Treinar o modelo
python tools/train.py configs/rotated_rtmdet/lane_vec_rotated_rtmdet_s-stop_lines.py