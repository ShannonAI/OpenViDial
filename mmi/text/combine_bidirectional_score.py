# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: combine_bidirectional_score
@time: 2020/12/27 22:54
@desc: weighted average score.forward and score.backward in nbest directories
       to generate final prediction

"""

import os
import numpy as np
from typing import List
import argparse


def find_sub_dirs(base_dir: str) -> List[str]:
    """find all rank-i subdirs under base_dir"""
    sub_dir_str = os.path.join(base_dir, "rank{}")
    i = 0
    sub_dirs = []
    while os.path.exists(sub_dir_str.format(i)):
        sub_dirs.append(sub_dir_str.format(i))
        i += 1
    return sub_dirs


def load_scores(sub_dirs: List[str], split="forward") -> np.array:
    """
    load score from all sub_dirs
    Returns:
        numpy array of [nbest, nsents]
    """
    scores = []
    for sub_dir in sub_dirs:
        with open(os.path.join(sub_dir, f"scores.{split}")) as fin:
            scores.append([float(x.strip()) for x in fin.readlines()])
    return np.array(scores)


def combine_score_only_text(forward_score, backward_score, alpha=1):
    return (1-alpha) * forward_score + alpha * backward_score

def combine_score_feature(forward_score, text_score, feature_score, alpha=0, alpha_2=0, alpha_3=0):
    return alpha*forward_score + alpha_2*text_score + alpha_3*feature_score

def combine_score_object(forward_score, text_score, feature_score, object_score, alpha=0, alpha_2=0, alpha_3=0, alpha_4=0):
    return alpha*forward_score + alpha_2*text_score + alpha_3*feature_score + alpha_4*object_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest-dir", type=str, help="nbest directory, which should contain rank1, .. rankn subdir")
    parser.add_argument("--nbest-dir-feature", type=str, default=None)
    parser.add_argument("--nbest-dir-object", type=str, default=None)
    parser.add_argument("--type", type=str, default="text")
    parser.add_argument("--output-file", type=str, help="selected prediction from nbest list by forward+backward")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="default weight of backward score")
    parser.add_argument("--alpha-2", type=float, default=0,
                        help="default weight of backward score of text")
    parser.add_argument("--alpha-3", type=float, default=0,
                        help="default weight of backward score of feature")
    parser.add_argument("--alpha-4", type=float, default=0,
                        help="default weight of backward score of object")
    args = parser.parse_args()

    base_dir = args.nbest_dir
    sub_dirs = find_sub_dirs(base_dir)

    forward_scores = load_scores(sub_dirs, split="forward")
    backward_scores = load_scores(sub_dirs, split="backward")
    if (args.type == 'text'):
        bidirection_scores = combine_score_only_text(forward_scores, backward_scores, args.alpha)
    elif (args.type == 'feature'):
        feature_dir = find_sub_dirs(args.nbest_dir_feature)
        backward_feature_scores = load_scores(feature_dir, split="backward")
        bidirection_scores = combine_score_feature(forward_scores, backward_scores, backward_feature_scores, args.alpha, args.alpha_2, args.alpha_3)
    elif (args.type == 'object'):
        feature_dir = find_sub_dirs(args.nbest_dir_feature)
        object_dir = find_sub_dirs(args.nbest_dir_object)
        backward_feature_scores = load_scores(feature_dir, split="backward")
        backward_object_scores = load_scores(object_dir, split="backward")
        bidirection_scores = combine_score_object(forward_scores, backward_scores, backward_feature_scores, backward_object_scores, args.alpha, args.alpha_2, args.alpha_3, args.alpha_4)

    best_idx = np.argmax(bidirection_scores, axis=0)
    print(f"compute {best_idx.shape[0]} bidirectional scores")

    pred_files = [open(os.path.join(sub_dir, "src-tgt.src")) for sub_dir in sub_dirs]

    with open(args.output_file, "w") as fout:
        for sent_idx, lines in enumerate(zip(*pred_files)):
            fout.write(lines[best_idx[sent_idx]])
    print(f"Wrote final output to {args.output_file}")


if __name__ == '__main__':
    main()

