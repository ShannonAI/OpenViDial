# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: text_and_image_dataset
@time: 2020/11/14 15:26
@desc: Combine Text and Image Datasets

"""

import numpy as np
import torch
from fairseq.data.fairseq_dataset import FairseqDataset
from mmi_fairseq.feature.feature_dataset import FeatureDataset
from fairseq.data import data_utils


class MMITextImageDataset(FairseqDataset):
    def __init__(self, image_dataset: FeatureDataset, text_dataset, vocab_dict, span_idxs, shuffle=False):
        self.img_dataset = image_dataset
        self.text_dataset = text_dataset
        self.vocab_dict = vocab_dict
        self.span_idxs = span_idxs
        self.shuffle = shuffle

    def __getitem__(self, index):
        '''
        group_idx, start_idx, end_idx = self.span_idxs[index].tolist()
        source_imgs = np.stack([self.img_dataset[idx] for idx in range(start_idx, end_idx)])  # n * dim
        source_texts = [self.text_dataset[idx] for idx in range(start_idx+1, end_idx+1)]  # n * sent_len
        target = self.text_dataset[end_idx] # will not be computed
        '''
        is_true, start_idx, end_idx = self.span_idxs[index].tolist()
        source_imgs = self.img_dataset[start_idx]  # dim
        source_texts = self.text_dataset[end_idx]  # sent_len
        target = self.text_dataset[end_idx] # will not be computed

        return {
            'id': index,
            'is_true': is_true,
            'source_imgs': source_imgs,
            'source_texts': source_texts,
            'target': torch.LongTensor(target)
        }

    def __len__(self):
        return len(self.span_idxs)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        '''
        group_idx, start_idx, end_idx = self.span_idxs[index].tolist()
        sum_tokens = 0
        for i in range(start_idx+1, end_idx+1):
            sum_tokens += len(self.text_dataset[i])
        '''
        is_true, start_idx, end_idx = self.span_idxs[index].tolist()
        sum_tokens = len(self.text_dataset[start_idx])
        return sum_tokens

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.num_tokens(index)

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # todo 添加bucket
        # # Inspired by LanguagePairDataset.ordered_indices
        # indices = indices[np.argsort(self.cap_ds.sizes[indices], kind='mergesort')]
        # return indices[np.argsort(self.img_ds.sizes[indices], kind='mergesort')]
        return indices

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        indices = []
        source_imgs = []
        source_texts = []
        source_lengths = []
        source_label = []
        targets = []

        target_ntokens = 0
        num_sentences = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)

            source_imgs.append(sample['source_imgs'])
            source_texts.append(torch.LongTensor(sample['source_texts']))
            source_lengths.append(len(sample['source_texts']))
            source_label.append(sample['is_true'])

            targets.append(sample['target'])
            target_ntokens += len(sample["target"])
        num_sentences = len(sample)

        indices = torch.tensor(indices, dtype=torch.long)

        source_label_tensor = torch.tensor(source_label, dtype=torch.float)

        source_lengths_tensor = torch.tensor(source_lengths, dtype=torch.long)

        image_tensor = torch.tensor(source_imgs, dtype=torch.float)

        source_texts_batch = data_utils.collate_tokens(source_texts,
                                                       pad_idx=self.vocab_dict.pad(),
                                                       eos_idx=self.vocab_dict.eos(),
                                                       move_eos_to_beginning=False)

        target_batch = data_utils.collate_tokens(targets,
                                                 pad_idx=self.vocab_dict.pad(),
                                                 eos_idx=self.vocab_dict.eos(),
                                                 move_eos_to_beginning=False)
        prev_target_batch = data_utils.collate_tokens(targets,
                                                      pad_idx=self.vocab_dict.pad(),
                                                      eos_idx=self.vocab_dict.eos(),
                                                      move_eos_to_beginning=True)

        return {
            'id': indices,
            'net_input': {
                'src_tokens': source_texts_batch,
                'src_label': source_label_tensor,
                'src_imgs': image_tensor,
                'src_lengths': source_lengths_tensor,
                'prev_output_tokens': prev_target_batch,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }
