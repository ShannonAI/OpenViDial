# encoding: utf-8
import os

def sent_num_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.sent_num.npy")

def offsets_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.offsets.npy")

def feature_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.features.mmap")

def object_file(data_dir, split, truncate=0):
    return os.path.join(data_dir, f"{split}.objects.mmap")+(f".{truncate}" if truncate else "")

def object_mask_file(data_dir, split, truncate=0):
    return os.path.join(data_dir, f"{split}.objects_mask.mmap")+(f".{truncate}" if truncate else "")

def src_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.src.txt")

def nmt_src_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.src-tgt.src")

def nmt_tgt_file(data_dir, split):
    return os.path.join(data_dir, f"{split}.src-tgt.tgt")

def text_bin_file(data_dir, split):
    return os.path.join(data_dir, split)

def img_file(data_dir, group_idx, sent_idx):
    return os.path.join(data_dir, f"img_dir{group_idx}", f"{sent_idx}.jpg")

def warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(10 * 1024 * 1024):
            pass

def padding(sents, pad_word):
    '''
    sents: list[list[int]]
    '''
    max_ = max(len(sen) for sen in sents)
    padding_sents = [[pad_word for j in range(max_)] for i in range(len(sents))]
    for i in range(len(sents)):
        padding_sents[i][0:len(sents[i])] = sents[i][:]
    return padding_sents

def read_sents(path, split):
    output = []
    if (split == 'test'):
        output.append([0])
    with open(os.path.join(path, split+'.mmi'), "r") as f:
        for line in f:
            line = line.strip().split()
            for i in range(len(line)):
                line[i] = int(line[i])
            output.append(line)
        f.close()
    return output

def get_batch(sample):
    batch_text = []
    batch_image = []
    batch_image_mask = []
    text_len = []
    for text_sample, image_sample, image_sample_mask in sample:
        batch_text.append(text_sample)
        batch_image.append(image_sample)
        batch_image_mask.append(image_sample_mask)
        text_len.append(text_sample)
    return batch_text, text_len, batch_image, batch_image_mask