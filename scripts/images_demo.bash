#!/bin/bash

MMROTATE_DIR="/home/openmmlab/mmrotate"
cd $MMROTATE_DIR
MODEL="parking_spaces"

SRC_FOLDER='/persistent/mmrotate/parking_spaces/datasets/parking_spaces/08_02_test'
MODEL_CONFIG='configs/rotated_rtmdet/lane_vec_rotated_rtmdet_s-3x-dota_ms.py'
MODEL_CHECKPOINT='work_dirs/parking_spaces/08_02/epoch_36.pth'
DST_FOLDER='/persistent/mmrotate/parking_spaces/results/08_02'

python demo/images_demo.py $SRC_FOLDER $MODEL_CONFIG $MODEL_CHECKPOINT $DST_FOLDER