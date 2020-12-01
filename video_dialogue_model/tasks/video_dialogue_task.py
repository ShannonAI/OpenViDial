import os
import numpy as np

from fairseq.data import Dictionary, data_utils
from video_dialogue_model.data.utils import text_bin_file
from fairseq.tasks import register_task, FairseqTask
from fairseq.tasks.translation import TranslationTask
from video_dialogue_model.data.feature_dataset import FeatureDataset
from video_dialogue_model.data.text_and_image_dataset import TextImageDataset
from video_dialogue_model.data.text_and_object_dataset import TextObjectDataset
from video_dialogue_model.data.object_dataset import ObjectDataset


# import model.image_transformer


@register_task('video-dialogue')
class VideoDialogueTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', default='output',
                            help='data directory')
        parser.add_argument('--max-src-sent', type=int, default=5,
                            help='max source sentence num')
        parser.add_argument('--img-type', type=str, default="objects", choices=["features", "objects"],
                            help='image feature types')

    @classmethod
    def setup_task(cls, args, **kwargs):
        vocab_dict_file = os.path.join(args.data_dir, f'dict.txt')
        vocab_dict = Dictionary.load(vocab_dict_file)

        return VideoDialogueTask(args, vocab_dict)

    def __init__(self, args, vocab_dict):
        super().__init__(args)
        self.vocab_dict = vocab_dict

    def load_text_image_dataset(self, split, **kwargs):
        features_dataset = FeatureDataset(self.args.data_dir)
        span_idxs = self.item2span_idxs(sent_num=features_dataset.sent_num,
                                        max_src_sent=self.args.max_src_sent)

        text_file = text_bin_file(self.args.data_dir, split)  # os.path.join(self.args.data_dir, split)
        text_dataset = data_utils.load_indexed_dataset(text_file, self.vocab_dict)

        self.datasets[split] = TextImageDataset(text_dataset=text_dataset,
                                                image_dataset=features_dataset,
                                                vocab_dict=self.vocab_dict,
                                                span_idxs=span_idxs,
                                                shuffle=True if split == "train" else False)

    def load_text_object_dataset(self, split, **kwargs):
        objects_dataset = ObjectDataset(self.args.data_dir)
        span_idxs = self.item2span_idxs(sent_num=objects_dataset.sent_num,
                                        max_src_sent=self.args.max_src_sent)

        text_file = text_bin_file(self.args.data_dir, split)  # os.path.join(self.args.data_dir, split)
        text_dataset = data_utils.load_indexed_dataset(text_file, self.vocab_dict)

        self.datasets[split] = TextObjectDataset(text_dataset=text_dataset,
                                                 image_dataset=objects_dataset,
                                                 vocab_dict=self.vocab_dict,
                                                 span_idxs=span_idxs,
                                                 shuffle=True if split == "train" else False)

    def load_dataset(self, split, **kwargs):
        if self.args.img_type == "features":
            return self.load_text_image_dataset(split, **kwargs)
        return self.load_text_object_dataset(split, **kwargs)

    @staticmethod
    def item2span_idxs(sent_num: np.array, max_src_sent: int) -> np.array:
        """
        compute each src/tgt span of dataset.
        For example, if we got [[0,1,2], [3,4]] as source texts,
        sent_num should be [3, 2], and we want to use only one sentence as src.
        the output should be [[0, 0, 1], [0, 1, 2], [1, 0, 1]]
        """
        span_idxs = []
        for group_idx in range(sent_num.shape[0]):
            num = int(sent_num[group_idx])
            for sent_idx in range(1, num):  # predict texts[i] given texts[:i]
                start_idx = max(0, sent_idx - max_src_sent)
                span_idxs.append((group_idx, start_idx, sent_idx))
        return np.array(span_idxs)

    @property
    def source_dictionary(self):
        return self.vocab_dict

    @property
    def target_dictionary(self):
        return self.vocab_dict
