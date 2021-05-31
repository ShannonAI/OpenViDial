# encoding: utf-8

import numpy as np
from torch.utils.data import Dataset

from video_dialogue_model.data.utils import sent_num_file, offsets_file, object_file, object_mask_file, warmup_mmap_file

class ObjectDataset(Dataset):
    MAX_OBJ = 20  # max-obj in mmap file
    """Load Object dataset"""
    def __init__(self, data_dir, split="train", max_obj=20):
        self.data_dir = data_dir
        self.sent_num = np.load(sent_num_file(data_dir, split))
        self.offsets = np.load(offsets_file(data_dir, split))
        self.total_sent_num = self.offsets[-1] + self.sent_num[-1]
        self.dim = 2048  # todo add x,y,w,h
        self.max_obj = max_obj  # max-obj when getting item
        warmup_mmap_file(object_file(data_dir, split, 0))
        print(self.total_sent_num, self.MAX_OBJ, self.dim)
        self.objects = np.memmap(object_file(data_dir, split, 0), dtype=np.float32, mode='r',
                                 shape=(self.total_sent_num, self.MAX_OBJ, self.dim))
        warmup_mmap_file(object_mask_file(data_dir, split, 0))
        self.objects_mask = np.memmap(object_mask_file(data_dir, split, 0), dtype=np.bool, mode='r',
                                      shape=(self.total_sent_num, self.MAX_OBJ))

    def __getitem__(self, item):
        return self.objects[item][: self.max_obj], self.objects_mask[item][: self.max_obj]

    def __len__(self):
        return self.total_sent_num