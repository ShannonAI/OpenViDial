# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: cnn_utils
@time: 2020/12/21 11:54
@desc: 

"""

from typing import List

import torch
import torchvision
from PIL import Image
from torchvision import transforms

CNN = torchvision.models.resnet50(pretrained=True)
CNN_FEATURE_DIM = 1000
CNN.cuda()


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
        input_batch = input_batch.cuda()

    with torch.no_grad():
        features = CNN(input_batch)
    return features
