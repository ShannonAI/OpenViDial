# Video-dialogue-model

## Requirements
pip install -r `requirements.txt`
### Notes: 
you may encounter a bug when trying to use `fairseq-generate` under torch>1.5.0, please refer to [this github issue](https://github.com/pytorch/fairseq/issues/2460) to resolve it.
### Todo:
upgrade to fairseq==0.10.0
在预测sentence_t时，允许不使用image_t
将图片直接作为input_sequences拼进去

## Preprocess data
可以参考sample_data中存放的数据
### Origin data
```
├──data_dir
    src.jsonl  # 每行是一轮完整的对话，list of str
    img_dir0  # dir_i里存储的是src中第i行对应的每一句话的图片
     └── 0.jpg
     └── 1.jpg
     └── ...
        ...
    ...
```
### Preprocessed data
```
├──preprocessed_data_dir
    src.txt  # 每行是一轮完整的对话，tokenized后的结果，句子中间用[SEP]隔开
    offsets.npy  # offsets[i]存储src中前i-1行一共有多少句话(也即多少个图)
    sent_num.npy # sent_num[o]存储src中第i行有多少句话
    img_features.npy  # 按顺序存储
    rcnn_feature  # mask-rcnn bboxes
      └── feature_dir0
             └── 0.npy
             └── 1.npy
             └── ...
```

### Preprocess pipelines
1. (Optional) install mask-rcnn and use pretrained-model to generate object-detection features in `preprocessed_data_dir/rcnn_feature`.
You can also download the preprocessed rcnn directory from [here](todo) and put it to `preprocessed_data_dir`
1. use `preprocess_video_data.py` to tokenize-text and extract CNN features in `preprocessed_data_dir`
