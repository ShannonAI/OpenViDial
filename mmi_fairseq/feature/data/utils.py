# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: path_utils
@time: 2020/11/14 12:13
@desc: 

"""
import os


def sent_num_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.sent_num.npy")


def offsets_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.offsets.npy")


def feature_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.features.mmap")


def object_file(data_dir, split, truncate=0):
    return os.path.join(data_dir, f"{split}.objects.mmap")+(f".{truncate}" if truncate else "")


def object_mask_file(data_dir, split, truncate=0):
    return os.path.join(data_dir, f"{split}.objects_mask.mmap")+(f".{truncate}" if truncate else "")


def src_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.src.txt")


def nmt_src_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.src-tgt.src")


def nmt_tgt_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.src-tgt.tgt")


def text_bin_file(data_dir, split):
    return os.path.join(data_dir, split)


def img_file(data_dir, group_idx, sent_idx):
    return os.path.join(data_dir, f"img_dir{group_idx}", f"{sent_idx}.jpg")


def warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(10 * 1024 * 1024):
            pass
