# Video-dialogue-model
This repo contains dataset introduction and three baselines in paper `todo`
## Requirements
pip install -r `requirements.txt`

## Preprocess data
可以参考sample_data中存放的数据
### Origin data
```
├──origin_data
    └── train.src.jsonl  # 每行是一轮完整的对话，list of str
    └── valid.src.jsonl  # 每行是一轮完整的对话，list of str
    └── test.src.jsonl  # 每行是一轮完整的对话，list of str
    └── train_images
             └── img_dir0  # dir_i里存储的是src中第i行对应的每一句话的图片
                     └── 0.jpg
                     └── 0.jpg.npy # faster-rcnn feature
                     └── 1.jpg
                     └── ...
                 ...
              ...
    └── valid_images
    └── test_images
```
### Preprocessed data
```
├──preprocessed_data
    src.txt  # 每行是一轮完整的对话，tokenized后的结果，句子中间用[SEP]隔开
    train.offsets.npy  # offsets[i]存储src中前i-1行一共有多少句话(也即多少个图)
    train.sent_num.npy # sent_num[o]存储src中第i行有多少句话
    train.features.mmap  # 按顺序存储
    train.objects.mmap  # faster-rcnn object feature
    train.objects_mask.mmap  # faster-rcnn object mask
```

### Preprocess pipelines

#### Tokenize and build sentence-offsets
We use Moses Tokenizer to tokenize texts:
`bash scripts/preprocess_video_data.sh`
and followed with byte-pair-encoding and fairseq-preprocess binarization
`bash scripts/preprocess_text_data.sh`

#### Donwload Preprocessed Image Features
##### Download Faster-RCNN features
You can download the preprocessed rcnn directory from [here](todo) and move it as `preprocessed_data_dir/objects.mmap`
and `preprocessed_data_dir/objects_mask.mmap`

##### Download CNN-pooling features
You can download the preprocessed ResNet50 features from [here](todo) and move it as `preprocessed_data_dir/features.mmap`

##### Extract features on your own
please refer to `./data_README.md`

## Baselines for this dataset
We proposed three baselines for this dataset, blabla todo
todo add picture here

We provide scripts to reproduce three baselines for this dataset.
The only thing you should change for your training/generation procedure
is to change the `DATA_DIR`, `MODEL_DIR` and `OUTPUT` variable to your own path.

### binarize data
todo add scripts here

### Text Only Model
`scripts/reproduce_baselines.sh`

### Coarse Visual Model
`scripts/text_and_img_feature.sh`

### Fine Visual Model
todo This model blabla copy from paper
`scripts/text_and_img_objects.sh`


## Generation and Evaluation
1. length/diversity/stopwords% stats `stats.py`
1. (beta) MMI generation `scripts/mmi/mmi_generate.sh`