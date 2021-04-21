# encoding: utf-8
import numpy as np
import config
from torch.utils.data import Dataset
from utils import sent_num_file, offsets_file, object_file, object_mask_file, warmup_mmap_file, read_sents

class ObjectDataset(Dataset):
    """Load Object dataset"""
    def __init__(self, data_dir, split="train"):
        self.data_dir = data_dir
        self.dim = 2048  # todo add x,y,w,h
        self.max_obj = config.max_obj  # max-obj when getting item
        self.sents = read_sents(data_dir, split)
        if (split == 'train' or split == 'valid'):
            self.sent_num = np.load(sent_num_file(data_dir, split))
            self.offsets = np.load(offsets_file(data_dir, split))
            self.total_sent_num = self.offsets[-1] + self.sent_num[-1]
            self.pair_id = self.get_train_dialogue(data_dir)
        else:
            self.total_sent_num = len(self.sents) - 1
            self.pair_id = [i for i in range(1, len(self.sents))]
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
        return self.sents[self.pair_id[item]], self.objects[self.pair_id[item]-1][: self.max_obj], self.objects_mask[self.pair_id[item]-1][: self.max_obj]

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
