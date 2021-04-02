import os
import numpy as np

import torch
from fairseq.data import Dictionary, data_utils
from video_dialogue_model.data.utils import text_bin_file
from fairseq.tasks import register_task, FairseqTask
from video_dialogue_model.data.feature_dataset import FeatureDataset
from video_dialogue_model.data.mmi_text_and_feature_dataset import MMITextImageDataset
#from video_dialogue_model.data.text_and_object_dataset import TextObjectDataset
#from video_dialogue_model.data.object_dataset import ObjectDataset


@register_task('mmi-video-dialogue')
class MMIVideoDialogueTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', default='output',
                            help='data directory')
        parser.add_argument('--max-obj', type=int, default=20,
                            help='max objects per sentence')
        parser.add_argument('--img-type', type=str, default="objects", choices=["features", "objects"],
                            help='image feature types')

    @classmethod
    def setup_task(cls, args, **kwargs):
        vocab_dict_file = os.path.join(args.data_dir, 'dict.txt')
        vocab_dict = Dictionary.load(vocab_dict_file)

        return MMIVideoDialogueTask(args, vocab_dict)

    def __init__(self, args, vocab_dict):
        super().__init__(args)
        self.args = args
        self.vocab_dict = vocab_dict

    def load_feature_dataset(self, split, **kwargs):
        features_dataset = FeatureDataset(self.args.data_dir, split)
        span_idxs = self.get_span_info(sent_num=features_dataset.sent_num)

        text_file = text_bin_file(self.args.data_dir, split)  # os.path.join(self.args.data_dir, split)
        text_dataset = data_utils.load_indexed_dataset(text_file, self.vocab_dict)

        self.datasets[split] = MMITextImageDataset(text_dataset=text_dataset,
                                                image_dataset=features_dataset,
                                                vocab_dict=self.vocab_dict,
                                                span_idxs=span_idxs,
                                                shuffle=True if split == "train" else False)
    '''
    def load_text_object_dataset(self, split, **kwargs):
        objects_dataset = ObjectDataset(self.args.data_dir, split, max_obj=self.args.max_obj)
        span_idxs = self.item2span_idxs(sent_num=objects_dataset.sent_num,
                                        max_src_sent=self.args.max_src_sent)

        text_file = text_bin_file(self.args.data_dir, split)  # os.path.join(self.args.data_dir, split)
        text_dataset = data_utils.load_indexed_dataset(text_file, self.vocab_dict)

        self.datasets[split] = TextObjectDataset(text_dataset=text_dataset,
                                                 image_dataset=objects_dataset,
                                                 vocab_dict=self.vocab_dict,
                                                 span_idxs=span_idxs,
                                                 shuffle=True if split == "train" else False)
    '''
    def load_dataset(self, split, **kwargs):
        if self.args.img_type == "features":
            return self.load_feature_dataset(split, **kwargs)
        return self.load_feature_dataset(split, **kwargs)

    @staticmethod
    def get_span_info(sent_num: np.array) -> np.array:
        """
        compute each src/tgt span of dataset.
        For example, if we got [[0,1,2], [3,4]] as source texts,
        then return [[0, 0, 2], [1, 3, 4]]
        """
        span_idxs = []
        start_idx = 0
        #span_value = 10
        for group_idx in range(sent_num.shape[0]):
            num = int(sent_num[group_idx])
            end_ = start_idx + 1
            while (end_ <= start_idx+num-1):
                span_idxs.append((group_idx, end_-1, end_))
                end_ += 1
            '''
            if (num == 1):
                start_idx += num
                continue
            start_ = start_idx
            end_ = min(start_idx+num-1, start_idx+20-1)
            while (end_ <= start_idx+num-1):
                span_idxs.append((group_idx, start_, end_))
                start_ = end_
                end_ += span_value
            '''
            start_idx += num
        return np.array(span_idxs)

    @property
    def source_dictionary(self):
        return self.vocab_dict

    @property
    def target_dictionary(self):
        return self.vocab_dict
    
    def inference_step(
        self, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            for model in models:
                return model(**sample["net_input"])
