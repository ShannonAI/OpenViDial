# Video-dialogue-model

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
```
