# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: text_and_image_dataset
@time: 2020/11/14 15:26
@desc: Combine Text and Object Datasets

"""

import numpy as np
import torch
from fairseq.data.fairseq_dataset import FairseqDataset
from video_dialogue_model.data.object_dataset import ObjectDataset
from fairseq.data import data_utils


class TextObjectDataset(FairseqDataset):
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
        group_idx, start_idx, end_idx = self.span_idxs[index].tolist()
        offsets = [self.get_1doffsets(group_idx, sent_idx) for sent_idx in range(start_idx, end_idx+1)]
        objects, object_masks = zip(*[self.img_dataset[idx] for idx in offsets])
        objects = np.stack(objects)  # [num_sent, num_objects, dim]
        objects_mask = np.stack(object_masks)  # [num_sent, num_objects]
        source_texts = np.concatenate([self.text_dataset[idx] for idx in offsets[:-1]])  # L
        target = self.text_dataset[offsets[-1]]

        return {
            'id': index,
            'objects': torch.FloatTensor(objects),
            'objects_mask': torch.FloatTensor(objects_mask),
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
        if len(samples) == 0:
            return {}

        indices = []
        source_objects = []
        objects_mask = []
        source_texts = []
        source_lengths = []
        targets = []

        target_ntokens = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)
            sent_num, max_object, rcnn_dim = sample["objects"].shape
            source_objects.append(sample['objects'])  # [sent_num, max_obj, dim]
            objects_mask.append(sample['objects_mask'])  # [sent_num, max_obj]
            source_texts.append(sample['source_texts'])  # [token_num]
            source_lengths.append(len(sample['source_texts']) + sent_num*max_object)

            targets.append(sample['target'])  # [token_num]
            target_ntokens += len(sample["target"])

        num_sentences = len(samples)

        indices = torch.tensor(indices, dtype=torch.long)

        max_sent = max(x.size(0) for x in source_objects)
        pad_objects = torch.zeros([num_sentences, max_sent, self.max_obj, self.img_dataset.dim], dtype=torch.float)
        pad_mask_objs = torch.zeros([num_sentences, max_sent, self.max_obj], dtype=torch.bool)
        for idx, objs in enumerate(source_objects):
            num_sent = objs.size(0)
            pad_objects[idx][: num_sent] = objs
            pad_mask_objs[idx][: num_sent] = objects_mask[idx][: num_sent]

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
                'objs': pad_objects,
                'objs_mask': pad_mask_objs,
                'src_lengths': source_lengths,
                'prev_output_tokens': prev_target_batch,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }
