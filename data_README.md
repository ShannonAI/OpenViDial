# Directory structure
**Note: every `train*` file or directory should have a 'valid' and a 'test' counterpart, we ignore them below for simplicity.**
```
├──origin_dir
      └── train.src.txt // each line has a raw sentence. This file has `num_sents` lines
      └── train.features.mmap  // numpy mmap array file of shape [num_sents, 1024], each row is a 1024-d Resnet-50 feature
      └── train.objects.mmap  // numpy mmap array file of shape [num_sents, 20, 2048],  faster-rcnn object feature file, each row contain 20 objects feature, which is 2048-d 
      └── train.objects_mask.mmap  // numpy mmap array file of shape [num_sents, 20],  faster-rcnn mask file, each row contain 20 objects mask, 1 for valid, 0 for mask.
      └── train.dialogue.jsonl // each line is an episode of dialogue, which contains a list of sentence-id, sentence-id should range from 0 to `num_sents-1`
      └── train_images // train images directory, contains `num_sents` images.
             └── 0.jpg
             └── 1.jpg
             └── ...
```

# Run Preprocessed Feature on Your Own(Optional)

## 1. CNN-pooling feature
`preprocess_video_data.py --cnn_feature ...`

## 2. Faster-RCNN features
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
3. use `preprocess_video_data.py --rcnn_feature ...` to gather R-CNN features from all pictures into `preprocessed_data_dir/objects.mmap`
and `preprocessed_data_dir/objects_mask.mmap`
