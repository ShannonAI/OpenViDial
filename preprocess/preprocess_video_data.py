# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: preprocess_video_data
@time: 2020/11/11 20:06
@desc: tokenize texts and extract image features todo(shuhe): should have a cleaner way.
"""

import argparse
import json
import logging
import os
from typing import List

import numpy as np
from sacremoses import MosesTokenizer

from video_dialogue_model.data.utils import (
    sent_num_file,
    offsets_file,
    src_file
)


os.environ['CUDA_VISIBLE_DEVICES'] = "0,"
TOKENIZER = MosesTokenizer(lang='en')


def load_origin_texts(data_dir, split="train") -> List[List[str]]:
    """load origin text data"""
    output = []
    ori_sen = []
    input_path = os.path.join(data_dir, f'{split}.origin.txt')
    logging.info(f"Loading origin data from {input_path}")
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line.replace("\u2013", "-")
            ori_sen.append(line)
        f.close()
    
    input_path = os.path.join(data_dir, f'{split}.dialogue.jsonl')
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = json.loads(line)
            t_list = []
            for id_ in ids:
                t_list.append(ori_sen[id_])
            output.append(t_list)
        f.close()
    logging.info(f"Loaded {sum(len(x) for x in output)} sentences from {os.path.join(data_dir, f'{split}.origin.txt')}")
    return output
    

def iterate_imgs(img_dir, split, sent_num: np.array) -> List[str]:
    """get image-paths according to sent-num array"""
    ids = []
    input_path = os.path.join(img_dir, f'{split}.dialogue.jsonl')
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line))
        f.close()
    
    output = []
    for group_idx in range(sent_num.shape[0]):
        for sent_idx in range(sent_num[group_idx]):
            output.append(os.path.join(img_dir, f"{split}_images", f"{ids[group_idx][sent_idx]}.jpg"))
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
    parser.add_argument('--workers', type=int, default=32,
                        help='cpu workers')
    parser.add_argument('--max_sent', type=int, default=5,
                        help='max history sentence number in src')
    parser.add_argument('--split', type=str, default="train",
                        help='split of dataset, train/valid/test')
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
    offsets = np.insert(sent_cumsum[: -1], obj=0, values=0)
    np.save(sent_num_file(args.output_dir, args.split), sent_num)
    np.save(offsets_file(args.output_dir, args.split), offsets)
    print(f"Moses tokenization and offsets computing Finished.")


if __name__ == '__main__':
    main()
