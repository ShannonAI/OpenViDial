# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: preprocess_video_data
@time: 2020/11/11 20:06
@desc: tokenize texts and extract image features
todo: add Faster-RCNN-based features

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

from data.utils import sent_num_file, offsets_file, feature_file, concat_text_file, img_file

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


def iterate_imgs_according_texts(origin_dir, group_texts: List[List[str]]) -> List[str]:
    """get image-paths according to texts"""
    output = []
    for group_idx, texts in enumerate(group_texts):
        for sent_idx in range(len(texts)):
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
                        help='MS-COCO data directory.')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size when processing image')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load text
    group_texts = load_origin_texts(args.origin_dir)

    # tokenize text  todo
    with open(concat_text_file(args.output_dir), "w") as fout:
        for group_text in group_texts:
            tokenized_group = tokenize_text(group_text)
            out_line = " [SEP] ".join(tokenized_group)
            fout.write(out_line + "\n")

    # compute image offsets/nums
    sent_num = np.array([len(g) for g in group_texts])
    sent_cumsum = np.cumsum(sent_num)
    total_sent = int(sent_cumsum[-1])
    offsets = np.insert(sent_cumsum[: -1], obj=0, values=0)
    np.save(sent_num_file(args.output_dir), sent_num)
    np.save(offsets_file(args.output_dir), offsets)

    # compute image features
    total_num = sum(len(g) for g in group_texts)
    feature_map = np.memmap(feature_file(args.output_dir), dtype='float32', mode='w+', shape=(total_num, FEATURE_DIM))
    idx = 0
    for batch_img_files in tqdm(chunked(iterate_imgs_according_texts(origin_dir=args.origin_dir,
                                                                     group_texts=group_texts),
                                        args.batch_size),
                                total=(total_sent // args.batch_size)):
        features = extract_image_feature(batch_img_files).cpu().numpy()
        img_num = features.shape[0]
        for img_idx in range(img_num):
            feature_map[idx + img_idx] = features[img_idx]
        idx += img_num


if __name__ == '__main__':
    main()
