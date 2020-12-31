# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: build_rcnn_mmap
@time: 2020/12/31 14:37
@desc: Gather every image.jpg.npy file to form a final objects.mmap file, which increased reading speed

"""

import argparse
import os
from typing import List

import numpy as np
from tqdm import tqdm

from video_dialogue_model.data.utils import (
    object_file,
    object_mask_file
)

RCNN_FEATURE_DIM = 2048
MAX_OBJECTS = 20


def iterate_img_dir(img_dir: str) -> List[str]:
    """iterate all images inside img_dir"""
    idx = 0
    output = []
    template = os.path.join(img_dir, "{}.jpg")
    while os.path.exists(template.format(idx)):
        output.append(template.format(idx))
    return output


def main():
    parser = argparse.ArgumentParser(description='video-data pre-processing.')

    parser.add_argument('--origin-dir', required=True,
                        help='origin data directory.')
    parser.add_argument('--output-dir', required=True,
                        help='output directory.')
    parser.add_argument('--split', type=str, default="train",
                        help='split of dataset, train/valid/test')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = iterate_img_dir(os.path.join(args.origin_dir, f"{args.split}_images"))
    total_num = len(image_files)
    print(f"Computing {RCNN_FEATURE_DIM}-dim features for a maximum of {MAX_OBJECTS} object in {total_num} pictures")

    objects = np.memmap(object_file(args.output_dir, args.split), dtype=np.float32, mode='w+',
                        shape=(total_num, MAX_OBJECTS, RCNN_FEATURE_DIM))
    objects_mask = np.memmap(object_mask_file(args.output_dir, args.split), dtype=np.bool, mode='w+',
                             shape=(total_num, MAX_OBJECTS))
    success = [False] * total_num
    while not all(success):
        for img_idx, img_file in tqdm(enumerate(iterate_img_dir(os.path.join(args.origin_dir, f"{args.split}_images"))),
                                      desc="Gathering Faster-RCNN feature"):
            try:
                if success[img_idx]:
                    continue
                npy_file = img_file + ".npy"
                rcnn_features = np.load(npy_file, allow_pickle=True)[()]
                objects_features = rcnn_features["features"]
                num_object = objects_features.shape[0]
                objects[img_idx][: num_object] = objects_features
                objects_mask[img_idx][: num_object] = True
                objects_mask[img_idx][num_object:] = False
                success[img_idx] = True
            except Exception as e:
                print(e)
                continue
