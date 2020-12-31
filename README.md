# OpenViDial
This repo contains introduction and downloading instructions of **OpenViDial** dataset, 
which is  proposed in paper [《todo》](todo). 

It also contains codes to reproduce three baselines. (See Section [Baselines](#baselines))

## Dataset

### Introduction
When humans converse, what a speaker will
say next significantly depends on what he sees.
Unfortunately, existing dialogue models generate
dialogue utterances only based on preceding
textual contexts, and visual contexts
are rarely considered. This is due to a lack
of a large-scale multi-module dialogue dataset
with utterances paired with visual contexts.
In this paper, we release OpenViDial, a largescale
multi-module dialogue dataset. The dialogue
turns and visual contexts are extracted
from movies and TV series, where each dialogue
turn is paired with the corresponding
visual context in which it takes place. Open-
ViDial contains a total number of 1.1 million
dialogue turns, and thus 1.1 million visual contexts
stored in images.


### Download Data
todo(shuhe)
1. download link
2. post-process shell (cat * > ...)
3. directory structure.


## Baselines
We proposed three baselines for this dataset:
* Model #1 - NoVisual: use only dialog texts without visual information

<div align="center">
  <img src="demo_data/model1.png"/>
</div>

* Model #2 - CoarseVisual: use texts and a pretrained ResNet50 on ImageNet to compute 1024-d feature from each picture

<div align="center">
  <img src="demo_data/model2.png"/>
</div>

* Model #3 - FineVisual: use texts and a pretrained Faster R-CNN on Genome to compute 2048-d * K objects features from each picture

<div align="center">
  <img src="demo_data/model3.png"/>
</div>

Faster R-CNN is an object detection framework. The detection sample and attention over objects during text decoding is shown below.

<div align="center">
  <img src="demo_data/attention_over_objects.png"/>
</div>

We provide scripts to reproduce three baselines for this dataset.
The only thing you should change for your preprocessing/training/generation procedure
is to change the `DATA_DIR`, `MODEL_DIR` and `OUTPUT` variable to your own path.

### Requirements
* python >= 3.6
* `pip install -r requirements.txt`

### Preprocess text data
todo(shuhe)
We use Moses Tokenizer to tokenize texts:
[todo](todo)
and followed with byte-pair-encoding and fairseq-preprocess binarization
[todo](todo)

### Prepare pre-computed CNN features and Faster-RCNN features
todo(shuhe)
##### Download Faster-RCNN features
You can download the preprocessed rcnn directory from [here](todo) and move it as `preprocessed_data_dir/objects.mmap`
and `preprocessed_data_dir/objects_mask.mmap`

##### Download CNN-pooling features
You can download the preprocessed ResNet50 features from [here](todo) and move it as `preprocessed_data_dir/features.mmap`

##### (Optional) Extract features on your own
See [video_dialogue_model/extract_features/extract_features.md](video_dialogue_model/extract_features/extract_features.md)

### Train and Evaluate Text Only Model
See [scripts/reproduce_baselines/text_only.sh](scripts/reproduce_baselines/text_only.sh)

### Train and Evaluate Coarse Visual Model
See [scripts/reproduce_baselines/text_and_img_feature.sh](scripts/reproduce_baselines/text_and_img_feature.sh)

### Train and Evaluate Fine Visual Model
See [scripts/reproduce_baselines/text_and_img_objects.sh](scripts/reproduce_baselines/text_and_img_objects.sh)

### Other Statistics
1. get length/diversity/stopwords% statistics of system output: `stats.py`
