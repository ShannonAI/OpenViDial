# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: truncate_data.py
@time: 2020/12/25 11:04
@desc: Truncate mmap file to smaller ones for training acceleration.

"""
import numpy as np
from video_dialogue_model.data.utils import object_file, object_mask_file
from tqdm import tqdm

MAX_OBJECTS, RCNN_FEATURE_DIM = 100, 2048
TRUNCATE_OBJECTS = 20

output_dir = "./preprocessed_data"
for split, total_num in zip(["train", "valid", "test"], [865694, 109109, 111346]):
    objects = np.memmap(object_file(output_dir, split), dtype=np.float32, mode='r',
                        shape=(total_num, MAX_OBJECTS, RCNN_FEATURE_DIM))
    objects_mask = np.memmap(object_mask_file(output_dir, split), dtype=np.bool, mode='r',
                             shape=(total_num, MAX_OBJECTS))

    truncate_objects = np.memmap(object_file(output_dir, split)+f".{TRUNCATE_OBJECTS}", dtype=np.float32, mode='w+',
                                 shape=(total_num, TRUNCATE_OBJECTS, RCNN_FEATURE_DIM))
    truncate_objects_mask = np.memmap(object_mask_file(output_dir, split)+f".{TRUNCATE_OBJECTS}", dtype=np.bool, mode='w+',
                                      shape=(total_num, TRUNCATE_OBJECTS))
    offset = 0
    chunk_size = 1000
    bar = tqdm(total=total_num // chunk_size)
    while offset < total_num:
        bar.update()
        end = offset + chunk_size
        truncate_objects[offset: end] = objects[offset: end, :TRUNCATE_OBJECTS, :]
        truncate_objects_mask[offset: end] = objects_mask[offset: end, :TRUNCATE_OBJECTS]
        offset = end
