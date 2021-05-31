# encoding: utf-8

import numpy as np
from torch.utils.data import Dataset
from video_dialogue_model.data.utils import sent_num_file, offsets_file, feature_file, warmup_mmap_file


class FeatureDataset(Dataset):
    """Load Feature dataset"""
    def __init__(self, data_dir, split="train"):
        self.data_dir = data_dir
        self.sent_num = np.load(sent_num_file(data_dir, split))
        self.offsets = np.load(offsets_file(data_dir, split))
        self.dim = 1000
        self.total_num = self.offsets[-1] + self.sent_num[-1]
        warmup_mmap_file(feature_file(data_dir, split))
        self.features = np.memmap(feature_file(data_dir, split), dtype='float32', mode='r',
                                  shape=(self.total_num, self.dim))

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return self.total_num