# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_folder', help='Source folder containing input images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('dst_folder', help='Destination folder for saving results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then passed to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # create destination folder if it doesn't exist
    os.makedirs(args.dst_folder, exist_ok=True)

    # list all files in the source folder
    img_files = os.listdir(args.src_folder)

    for img_file in img_files:
        # construct the full path to the input image
        img_path = os.path.join(args.src_folder, img_file)

        # test a single image
        result = inference_detector(model, img_path)

        # construct the full path to the output result image
        result_img_path = os.path.join(args.dst_folder, f'result_{img_file}')
        
        # show the results
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=result_img_path,
            pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
