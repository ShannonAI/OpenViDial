# Video-dialogue-model

## Requirements
pip install -r `requirements.txt`
### Todo:
在预测sentence_t时，允许不使用image_t

## Preprocess data
可以参考sample_data中存放的数据
todo 去除中文字幕
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

todo(yuxian): we should train/apply bpe to corpus

#### Extracting Faster-RCNN features
##### Download
You can download the preprocessed rcnn directory from [here](todo) and move it as `preprocessed_data_dir/objects.mmap`
and `preprocessed_data_dir/objects_mask.mmap`
##### (Optional) Preprocess on your own
install mask-rcnn and use pretrained-model to generate object-detection features in `preprocessed_data_dir/rcnn_feature`.
1. Install [`vqa-maskrcnn-benchmark`](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark) repository and download the model and config. 

```text
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```

2. Extract features for images

Run from root directory

```text
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir <path_to_directory_with_images> --output_folder <path_to_output_extracted_features>
```
3. use `preprocess_video_data.py --rcnn_feature ...` to extract RCNN features in `preprocessed_data_dir/objects.mmap`
and `preprocessed_data_dir/objects_mask.mmap`

#### Extracting CNN-pooling features
##### Download
You can download the preprocessed ResNet50 features from [here](todo) and move it as `preprocessed_data_dir/features.mmap`
##### (Optional) Preprocess on your own
1. use `preprocess_video_data.py --cnn_feature ...` to extract CNN features in `preprocessed_data_dir/features.mmap`


## Generation and Evaluation
1. normal generation `scripts/generate.sh`
2. MMI generation `scripts/mmi/mmi_generate.sh`