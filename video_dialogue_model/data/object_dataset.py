# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: feature_dataset
@time: 2020/11/14 12:07
@desc: Read Faster-RCNN object dataset

"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from video_dialogue_model.data.utils import sent_num_file, offsets_file, object_file, object_mask_file, warmup_mmap_file


class ObjectDataset(Dataset):
    """Load Object dataset"""
    def __init__(self, data_dir, split="train"):
        self.data_dir = data_dir
        self.sent_num = np.load(sent_num_file(data_dir, split))
        self.offsets = np.load(offsets_file(data_dir, split))
        self.total_sent_num = self.offsets[-1] + self.sent_num[-1]
        self.dim = 2048  # todo add x,y,w,h
        self.max_obj = 20
        warmup_mmap_file(object_file(data_dir, split))
        self.objects = np.memmap(object_file(data_dir, split), dtype=np.float32, mode='r',
                                 shape=(self.total_sent_num, self.max_obj, self.dim))
        warmup_mmap_file(object_mask_file(data_dir, split))
        self.objects_mask = np.memmap(object_mask_file(data_dir, split), dtype=np.bool, mode='r',
                                      shape=(self.total_sent_num, self.max_obj))

    def __getitem__(self, item):
        """
        Returns:
            1. object features, [self.max_object, self.dim]
            2. object_mask, [self.max_object], 0 means no object
        """
        return self.objects[item], self.objects_mask[item]

    def __len__(self):
        return self.total_sent_num


def test_object_dataset():
    from tqdm import tqdm

    # for debug
    np.memmap(object_file("../../sample_data/preprocessed_data", "train"),
              dtype=np.float32, mode="w+",
              shape=(2, 20, 2048))
    np.memmap(object_mask_file("../../sample_data/preprocessed_data/", "train"),
              dtype=np.bool, mode="w+",
              shape=(2, 20))

    d = ObjectDataset(data_dir="../../sample_data/preprocessed_data")
    for x in tqdm(d):
        print(x[0].shape, x[1].shape)
        print(x)


if __name__ == '__main__':
    test_object_dataset()
