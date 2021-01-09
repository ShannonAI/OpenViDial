# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: stats
@time: 2020/12/26 14:37
@desc: This file includes helper metrics we used in our paper to
       compute statistics of model outputs

"""
from sacremoses import MosesTokenizer
TOKENIZER = MosesTokenizer(lang="en")


class BaseMetric:
    def __init__(self):
        pass

    def update(self, line):
        """update stats values given a line of system output"""
        raise NotImplementedError

    def __repr__(self):
        """print final stats over all lines of system output"""
        raise NotImplementedError


class DiversityMetric(BaseMetric):
    """Compute n-gram diversity"""
    def __init__(self, n):
        super(DiversityMetric, self).__init__()
        self.n = n
        self.s = set()
        self.total = 0

    def update(self, line):
        words = line.split()
        start_idx = self.n - 1
        for end_offset in range(start_idx, len(words)):
            start_offset = end_offset - self.n + 1
            ngram = " ".join(words[start_offset: end_offset + 1])
            self.s.add(ngram)
            self.total += 1

    def __repr__(self):
        return f"Diversity-{self.n}: {len(self.s)/self.total}"


class AvgLengthMetric(BaseMetric):
    def __init__(self):
        super(AvgLengthMetric, self).__init__()
        self.sents = 0
        self.total_length = 0

    def update(self, line):
        self.total_length += len(line.split())
        self.sents += 1

    def __repr__(self):
        return f"AvgLength: {self.total_length/self.sents}"


class StopWordsRatioMetric(BaseMetric):
    def __init__(self):
        super(StopWordsRatioMetric, self).__init__()
        self.sents = 0
        self.total_stop_words = 0
        self.total_words = 0
        self.stop_words = set()
        with open("video_dialogue_model/data/stopwords.txt") as fin:
            stop_words = set(x.strip() for x in fin.readlines() if x.strip())
            for word in stop_words:
                for token in TOKENIZER.tokenize(word):
                    self.stop_words.add(token)

    def update(self, line):
        self.sents += 1
        for word in line.split():
            self.total_words += 1
            if word.lower() in self.stop_words:
                self.total_stop_words += 1

    def __repr__(self):
        return f"StopWords%: {self.total_stop_words/self.total_words}; StopWords/Sent: {self.total_stop_words/self.sents}"


def compute_stats(tgt_file):
    """
    compute diversity of system output
    Args:
        tgt_file: each line is an output line
        n: ngram
    """
    metrics = [DiversityMetric(i) for i in range(1, 5)]
    metrics.append(StopWordsRatioMetric())
    metrics.append(AvgLengthMetric())

    with open(tgt_file) as fin:
        for l in fin:
            l = l.strip()
            if not l:
                continue
            for metric in metrics:
                metric.update(l)

    for metric in metrics:
        print(metric)


if __name__ == '__main__':
    for file in [
        "/data/yuxian/datasets/new-video/models/text/sys.txt",
        "/data/yuxian/datasets/new-video/models/feature_lr3e-4/sys.txt",
        "/data/yuxian/datasets/new-video/models/object_lr2e-4/sys.txt",
    ]:
        print(f"=====Stats of {file}=====")
        compute_stats(file)
