# encoding: utf-8
import numpy as np
from torch.utils.data import Dataset
from utils import warmup_mmap_file, feature_file, sent_num_file, offsets_file, read_sents


class FeatureDataset(Dataset):
    """Load Feature dataset"""
    def __init__(self, data_dir, split="train"):
        self.data_dir = data_dir
        self.dim = 1000
        self.sents = read_sents(data_dir, split)
        if (split == 'train' or split == 'valid'):
            self.sent_num = np.load(sent_num_file(data_dir, split))
            self.offsets = np.load(offsets_file(data_dir, split))
            self.total_num = self.sent_num[-1] + self.offsets[-1]
            self.pair_id = self.get_train_dialogue(data_dir)
        else:
            self.total_num = len(self.sents) - 1
            self.pair_id = [i for i in range(1, len(self.sents))]
        warmup_mmap_file(feature_file(data_dir, split))
        self.features = np.memmap(feature_file(data_dir, split), dtype='float32', mode='r',
                                  shape=(self.total_num, self.dim))

    def __getitem__(self, item):
        return self.sents[self.pair_id[item]], self.features[self.pair_id[item]-1]

    def __len__(self):
        return len(self.pair_id)
    
    def get_train_dialogue(self, data_dir):
        tmp = []
        start_ = 0
        for dialogue_id in range(self.sent_num.shape[0]):
            num = int(self.sent_num[dialogue_id])
            for i in range(1, num):
                tmp.append(start_+i)
            start_ += num
        return tmp