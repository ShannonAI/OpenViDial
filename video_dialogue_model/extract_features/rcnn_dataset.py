# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dataset
@time: 2020/12/22 21:12
@desc: 

"""

from functools import partial
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from maskrcnn_benchmark.structures.image_list import to_image_list
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, file_names: List[str]):
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        image_path = self.file_names[item]
        im, im_scale, im_info = self._image_transform(image_path)
        return im, im_scale, im_info
        # img_tensor.append(im)
        # im_scales.append(im_scale)
        # im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info


def get_dataloader(file_names, batch_size, workers):
    dataset = ImageDataset(file_names)
    return DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(to_image_list, size_divisible=32),
        pin_memory=True,
    )


def yuxian_collate(batch: List[List[torch.Tensor]]):
    return to_image_list(batch), [x[1] for x in batch], [x[2] for x in batch]
