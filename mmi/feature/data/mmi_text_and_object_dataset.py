# encoding: utf-8
"""
@author: Shuhe Wang
@contact: shuhe_wang@shannonai.com

@version: 1.0
@file: text_and_image_dataset
@time: 2021/05/29 12:02
@desc: Combine Text and Object Datasets

"""

import numpy as np
import torch
from fairseq.data.fairseq_dataset import FairseqDataset
from mmi_fairseq.feature.data.object_dataset import ObjectDataset
from fairseq.data import data_utils


class MMITextObjectDataset(FairseqDataset):
    """
    A combine of text dataset and object dataset
    """
    def __init__(self, image_dataset: ObjectDataset, text_dataset, vocab_dict, span_idxs, shuffle=False):
        self.img_dataset = image_dataset
        self.text_dataset = text_dataset
        self.vocab_dict = vocab_dict
        self.span_idxs = span_idxs
        self.shuffle = shuffle
        self.max_obj = image_dataset.max_obj

    def __getitem__(self, index):
        # todo: try to add [bos] at the beginning of text sequence to separate objects/texts
        is_true, start_idx, end_idx = self.span_idxs[index].tolist()
        objects, objects_mask = self.img_dataset[start_idx] # max_obj * dim, max_obj
        source_texts = self.text_dataset[end_idx]  # sent_len
        target = self.text_dataset[end_idx] # will not be computed

        return {
            'id': index,
            'is_true': is_true,
            'objects': objects,
            'objects_mask': objects_mask,
            'source_texts': source_texts,
            'target': torch.LongTensor(target)
        }

    def __len__(self):
        return len(self.span_idxs)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
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
        source_objects = []
        objects_mask = []
        source_texts = []
        source_lengths = []
        source_label = []
        targets = []

        target_ntokens = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)

            source_objects.append(sample["objects"])
            objects_mask.append(sample["objects_mask"])
            source_texts.append(torch.LongTensor(sample['source_texts']))
            source_lengths.append(len(sample['source_texts']))
            source_label.append(sample['is_true'])

            targets.append(sample['target'])
            target_ntokens += len(sample["target"])
        num_sentences = len(samples)

        indices = torch.tensor(indices, dtype=torch.long)

        source_label_tensor = torch.tensor(source_label, dtype=torch.float)

        source_lengths_tensor = torch.tensor(source_lengths, dtype=torch.long)

        image_tensor = torch.tensor(source_objects, dtype=torch.float)

        mask_tensor = torch.tensor(objects_mask, dtype=torch.float)

        

        source_texts_batch = data_utils.collate_tokens(source_texts,
                                                       pad_idx=self.vocab_dict.pad(),
                                                       eos_idx=self.vocab_dict.eos(),
                                                       move_eos_to_beginning=False)

        mask_ones = torch.ones((source_texts_batch.shape[0], source_texts_batch.shape[1]), dtype=torch.float) # B * T

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
                'mask_ones': mask_ones,
                'src_label': source_label_tensor,
                'objs': image_tensor,
                'objs_mask': mask_tensor,
                'src_lengths': source_lengths_tensor,
                'prev_output_tokens': prev_target_batch,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }
