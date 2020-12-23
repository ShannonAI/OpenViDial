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
from torch.utils.data import Dataset, DataLoader
import warnings

CNN = torchvision.models.resnet50(pretrained=True)
CNN.eval()
CNN_FEATURE_DIM = 1000
if torch.cuda.is_available():
    CNN = CNN.cuda()
else:
    warnings.warn("cuda not available")


class ImageDataset(Dataset):
    def __init__(self, file_names: List[str]):
        self.file_names = file_names
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=Image.LINEAR),
            transforms.CenterCrop(224),  # todo CenterCrop可能会漏掉长屏幕的信息，待优化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        return self.preprocess(Image.open(self.file_names[item]))


def get_dataloader(file_names, batch_size, workers):
    dataset = ImageDataset(file_names)
    return DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
