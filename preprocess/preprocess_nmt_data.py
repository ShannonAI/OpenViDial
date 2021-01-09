# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: preprocess_nmt_data
@time: 2020/11/11 20:06
@desc: tokenize texts for nmt task training.
Notes: Since we use it for MMI training, we use sents[i] to predict sents[i-1]
"""

import argparse
import json
import logging
import os
from typing import List

from sacremoses import MosesTokenizer

from video_dialogue_model.data.utils import (
    nmt_src_file,
    nmt_tgt_file
)

TOKENIZER = MosesTokenizer(lang='en')


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
            output.append([x.replace("\u2013", "-") for x in sents])  # todo delete after re-generating data
    logging.info(f"Loaded {sum(len(x) for x in output)} sentences from {input_path}")
    return output


def tokenize_text(texts: List[str]):
    return [TOKENIZER.tokenize(t, return_str=True) for t in texts]


def main():
    parser = argparse.ArgumentParser(description='video-data pre-processing.')

    parser.add_argument('--origin-dir', required=True,
                        help='origin data directory.')
    parser.add_argument('--output-dir', required=True,
                        help='output directory.')
    parser.add_argument('--split', type=str, default="train",
                        help='split of dataset, train/valid/test')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load text
    group_texts = load_origin_texts(args.origin_dir, args.split)

    # tokenize text
    with open(nmt_src_file(args.output_dir, args.split), "w") as fsrc, \
         open(nmt_tgt_file(args.output_dir, args.split), "w") as ftgt:
        for group_idx, group_text in enumerate(group_texts):
            tokenized_group = tokenize_text(group_text)
            for idx in range(1, len(tokenized_group)):
                s = tokenized_group[idx]
                t = tokenized_group[idx-1]
                fsrc.write(s+"\n")
                ftgt.write(t+"\n")


if __name__ == '__main__':
    main()
