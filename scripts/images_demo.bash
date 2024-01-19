#!/bin/bash

MMROTATE_DIR="/home/openmmlab/mmrotate"
cd $MMROTATE_DIR

SRC_FOLDER='/persistent/bags/vegas/bev_imgs_church_to_center'
MODEL_CONFIG='configs/rotated_rtmdet/lane_vec_rotated_rtmdet_s-stop_lines.py'
MODEL_CHECKPOINT='work_dirs/stop_lines/17_01/epoch_36.pth'
DST_FOLDER='/persistent/mmrotate/stop_lines/results/church_to_center'

python demo/images_demo.py $SRC_FOLDER $MODEL_CONFIG $MODEL_CHECKPOINT  $DST_FOLDER