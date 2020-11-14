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


def sent_num_file(data_dir):
    return os.path.join(data_dir, "sent_num.npy")


def offsets_file(data_dir):
    return os.path.join(data_dir, "offsets.npy")


def feature_file(data_dir):
    return os.path.join(data_dir, "features.mmap")


def concat_text_file(data_dir):
    return os.path.join(data_dir, "src.txt")


def img_file(data_dir, group_idx, sent_idx):
    return os.path.join(data_dir, f"img_dir{group_idx}", f"{sent_idx}.jpg")


def warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass
