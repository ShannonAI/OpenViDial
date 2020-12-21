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
from video_dialogue_model.data.feature_dataset import FeatureDataset
from fairseq.data import data_utils


class TextImageDataset(FairseqDataset):
    def __init__(self, image_dataset: FeatureDataset, text_dataset, vocab_dict, span_idxs, shuffle=False):
        self.img_dataset = image_dataset
        self.text_dataset = text_dataset
        self.vocab_dict = vocab_dict
        self.span_idxs = span_idxs
        self.shuffle = shuffle

    def __getitem__(self, index):
        group_idx, start_idx, end_idx = self.span_idxs[index].tolist()
        offsets = [self.get_1doffsets(group_idx, sent_idx) for sent_idx in range(start_idx, end_idx+1)]
        source_imgs = np.stack([self.img_dataset[idx] for idx in offsets])  # n * dim
        source_texts = np.concatenate([self.text_dataset[idx] for idx in offsets[:-1]])  # L
        target = self.text_dataset[offsets[-1]]

        return {
            'id': index,
            'source_imgs': torch.FloatTensor(source_imgs),
            'source_texts': torch.LongTensor(source_texts),
            'target': torch.LongTensor(target)
        }

    def __len__(self):
        return len(self.span_idxs)

    def get_1doffsets(self, group_idx, sent_idx):
        group_offset = int(self.img_dataset.offsets[group_idx])
        sent_num = int(self.img_dataset.sent_num[group_idx])
        assert sent_idx < sent_num, f"origin text group {group_idx} has {sent_num} sents, " \
                                    f" sent_idx {sent_idx} should be less than {sent_num}"
        return group_offset + sent_idx

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        group_idx, start_idx, end_idx = self.span_idxs[index].tolist()
        return len(self.text_dataset[self.get_1doffsets(group_idx, end_idx)])  # todo其实应该考虑src

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
        indices = []

        source_imgs = []
        source_texts = []
        source_lengths = []
        targets = []

        target_ntokens = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)

            source_imgs.append(sample['source_imgs'])
            source_texts.append(sample['source_texts'])
            source_lengths.append(len(sample['source_texts']))

            targets.append(sample['target'])
            target_ntokens += len(sample["target"])

        num_sentences = len(samples)

        # # FIXME: workaround for edge case in parallel processing
        # # (framework passes empty samples list
        # # to collater under certain conditions)
        # if num_sentences == 0:
        #     return None

        indices = torch.tensor(indices, dtype=torch.long)

        max_sent = max(x.size(0) for x in source_imgs)
        pad_imgs = torch.zeros([num_sentences, max_sent, self.img_dataset.dim], dtype=torch.float)
        for idx, imgs in enumerate(source_imgs):
            pad_imgs[idx][: imgs.size(0)] = imgs

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
                'src_imgs': pad_imgs,
                'src_lengths': source_lengths,
                'prev_output_tokens': prev_target_batch,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }
