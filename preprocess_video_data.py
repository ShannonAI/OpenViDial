# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: preprocess_video_data
@time: 2020/11/11 20:06
@desc: tokenize texts and extract image features
"""

import argparse
import json
import os
from typing import List

import numpy as np
from more_itertools import chunked
from sacremoses import MosesTokenizer
from tqdm import tqdm
import logging

from video_dialogue_model.data.utils import (
    sent_num_file,
    offsets_file,
    feature_file,
    src_file,
    img_file,
    object_file,
    object_mask_file
)


TOKENIZER = MosesTokenizer(lang='en')

# os.environ['TORCH_HOME'] = '/userhome/yuxian/torch_models'  # setting the environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = "0,"
RCNN_FEATURE_DIM = 2048
MAX_OBJECTS = 100


def load_origin_texts(data_dir, split="train") -> List[List[str]]:
    """load origin text data"""
    output = []
    input_path = os.path.join(data_dir, f'{split}.src.jsonl')
    logging.info(f"Loading origin data from {input_path}")
    with open(input_path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sents = json.loads(line)
            output.append(sents)
    logging.info(f"Loaded {sum(len(x) for x in output)} sentences from {input_path}")
    return output


def iterate_imgs(img_dir, sent_num: np.array) -> List[str]:
    """get image-paths according to sent-num array"""
    output = []
    for group_idx in range(sent_num.shape[0]):
        for sent_idx in range(sent_num[group_idx]):
            output.append(img_file(img_dir, group_idx, sent_idx))
    return output


def tokenize_text(texts: List[str]):
    return [TOKENIZER.tokenize(t, return_str=True) for t in texts]


def main():
    parser = argparse.ArgumentParser(description='video-data pre-processing.')

    parser.add_argument('--origin-dir', required=True,
                        help='origin data directory.')
    parser.add_argument('--output-dir', required=True,
                        help='output directory.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size when processing image')
    parser.add_argument('--max_sent', type=int, default=5,
                        help='max history sentence number in src')
    parser.add_argument('--split', type=str, default="train",
                        help='split of dataset, train/valid/test')
    parser.add_argument('--rcnn_feature', action="store_true",
                        help='gather rcnn feature memmap')
    parser.add_argument('--cnn_feature', action="store_true",
                        help='gather cnn feature memmap')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load text
    group_texts = load_origin_texts(args.origin_dir, args.split)

    # tokenize text
    with open(src_file(args.output_dir, args.split), "w") as fsrc:
        for group_idx, group_text in enumerate(group_texts):
            tokenized_group = tokenize_text(group_text)
            for src in tokenized_group:
                fsrc.write(src + "\n")

    # compute group offsets/nums
    sent_num = np.array([len(g) for g in group_texts])
    sent_cumsum = np.cumsum(sent_num)
    total_sent = int(sent_cumsum[-1])
    offsets = np.insert(sent_cumsum[: -1], obj=0, values=0)
    np.save(sent_num_file(args.output_dir, args.split), sent_num)
    np.save(offsets_file(args.output_dir, args.split), offsets)

    total_num = sum(len(g) for g in group_texts)

    # compute cnn feature
    if args.cnn_feature:
        from cnn_utils import CNN_FEATURE_DIM, extract_image_feature
        feature_map = np.memmap(feature_file(args.output_dir, args.split), dtype='float32', mode='w+',
                                shape=(total_num, CNN_FEATURE_DIM))
        idx = 0
        for batch_img_files in tqdm(chunked(iterate_imgs(img_dir=os.path.join(args.origin_dir, f"{args.split}_images"),
                                                         sent_num=sent_num),
                                            args.batch_size),
                                    total=(total_sent // args.batch_size),
                                    desc="Computing CNN feature"
                                    ):
            features = extract_image_feature(batch_img_files).cpu().numpy()
            img_num = features.shape[0]
            for img_idx in range(img_num):
                feature_map[idx + img_idx] = features[img_idx]
            idx += img_num

    # gather rcnn feature
    if args.rcnn_feature:
        objects = np.memmap(object_file(args.output_dir, args.split), dtype=np.float32, mode='w+',
                            shape=(total_num, MAX_OBJECTS, RCNN_FEATURE_DIM))
        objects_mask = np.memmap(object_mask_file(args.output_dir, args.split), dtype=np.bool, mode='w+',
                                 shape=(total_num, MAX_OBJECTS))
        rcnn_dir = os.path.join(args.output_dir, "rcnn_feature")
        for img_idx, img_file in tqdm(enumerate(iterate_imgs(origin_dir=args.origin_dir, sent_num=sent_num)),
                                      desc="Gathering Faster-RCNN feature"):
            npy_file = img_file.replace(args.origin_dir, rcnn_dir)[: -3] + "npy"
            rcnn_features = np.load(npy_file, allow_pickle=True)[()]
            objects_features = rcnn_features["features"]
            num_object = objects_features.shape[0]
            objects[img_idx][: num_object] = objects_features
            objects_mask[img_idx][: num_object] = True
            objects_mask[img_idx][num_object:] = False


if __name__ == '__main__':
    main()
