# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: preprocess_video_data
@time: 2020/11/11 20:06
@desc: tokenize texts and extract image features
todo: add Faster-RCNN-based features
todo: 其实直接将每一句都按顺序存下来就行，然后再FairSeqDataset类的length中再通过eos拼接的方式把同一个group的拼接到一起。

"""

import argparse
import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from more_itertools import chunked
from sacremoses import MosesTokenizer
from torchvision import transforms
from tqdm import tqdm

from video_dialogue_model.data.utils import sent_num_file, offsets_file, feature_file, src_file, img_file

TOKENIZER = MosesTokenizer(lang='en')

CNN = torch.hub.load('pytorch/vision:v0.7.0', 'resnet50', pretrained=True)
FEATURE_DIM = 1000
DEVICE = "cuda:0"
CNN.to(DEVICE)


def load_origin_texts(data_dir) -> List[List[str]]:
    """load origin text data"""
    output = []
    with open(os.path.join(data_dir, 'src.jsonl')) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sents = json.loads(line)
    output.append(sents)
    return output


def iterate_imgs(origin_dir, sent_num: np.array) -> List[str]:
    """get image-paths according to sent-num array"""
    output = []
    for group_idx in range(sent_num.shape[0]):
        for sent_idx in range(sent_num[group_idx]):
            output.append(img_file(origin_dir, group_idx, sent_idx))
    return output


def tokenize_text(texts: List[str]):
    return [TOKENIZER.tokenize(t, return_str=True) for t in texts]


def extract_image_feature(filenames: List[str]):
    """extract features from image files"""
    input_images = [Image.open(fname) for fname in filenames]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # todo CenterCrop可能会漏掉长屏幕的信息，待优化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensors = [preprocess(img) for img in input_images]
    input_batch = torch.stack(input_tensors)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to(DEVICE)

    with torch.no_grad():
        features = CNN(input_batch)
    return features


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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load text
    group_texts = load_origin_texts(args.origin_dir)

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

    # compute image features
    total_num = sum(len(g) for g in group_texts)
    feature_map = np.memmap(feature_file(args.output_dir, args.split), dtype='float32', mode='w+', shape=(total_num, FEATURE_DIM))
    idx = 0
    for batch_img_files in tqdm(chunked(iterate_imgs(origin_dir=args.origin_dir,
                                                     sent_num=sent_num),
                                        args.batch_size),
                                total=(total_sent // args.batch_size)):
        features = extract_image_feature(batch_img_files).cpu().numpy()
        img_num = features.shape[0]
        for img_idx in range(img_num):
            feature_map[idx + img_idx] = features[img_idx]
        idx += img_num


if __name__ == '__main__':
    main()
