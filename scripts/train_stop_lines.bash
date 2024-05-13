#!/bin/bash

MMROTATE_DIR="/home/openmmlab/mmrotate"
cd $MMROTATE_DIR

DATASET="02_05"
MODEL="global_model"
TRAIN_PERCENTAGE=0.75

# Convert labels
python dataset/convert_labels_sl.py --dataset $DATASET

# Split dataset
#python dataset/split_dataset.py --dataset $DATASET --model $MODEL --train_percentage $TRAIN_PERCENTAGE

# Remover a pasta data/split_ms_dota
rm -r data/split_ms_dota

# Atualizar o JSON para o conjunto de treino
jq '.img_dirs = ["dataset/'"$MODEL"'/600/dota/'"$DATASET"'/train"]' tools/data/dota/split/split_configs/ms_train.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_train.json
jq '.ann_dirs = ["dataset/'"$MODEL"'/600/dota/'"$DATASET"'/train"]' tools/data/dota/split/split_configs/ms_train.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_train.json

# Atualizar o JSON para o conjunto de validaçao
jq '.img_dirs = ["dataset/'"$MODEL"'/600/dota/'"$DATASET"'/val"]' tools/data/dota/split/split_configs/ms_val.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_val.json
jq '.ann_dirs = ["dataset/'"$MODEL"'/600/dota/'"$DATASET"'/val"]' tools/data/dota/split/split_configs/ms_val.json > tmp.json && mv tmp.json tools/data/dota/split/split_configs/ms_val.json

# Executar o script de divisão de imagens
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ms_train.json
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ms_val.json

# Treinar o modelo
#python tools/train.py work_dirs/global_model/lane_vec/lane_vec_new_res.py