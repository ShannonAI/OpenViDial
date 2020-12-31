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

### Train and Evaluate Model #1 - NoVisual
`bash scripts/reproduce_baselines/text_only.sh` will train and evaluate model automatically, 
but you should correspondingly change `MODEL_DIR` and `DATA_DIR` for your setup

### Train and Evaluate Model #2 - CoarseVisual
`bash scripts/reproduce_baselines/text_and_img_feature.sh` will train and evaluate model automatically, 
but you should correspondingly change `MODEL_DIR` and `DATA_DIR` for your setup

### Train and Evaluate Model #3 - FineVisual
`bash scripts/reproduce_baselines/text_and_img_objects.sh` will train and evaluate model automatically, 
but you should correspondingly change `MODEL_DIR` and `DATA_DIR` for your setup

### Other Statistics
* get length/diversity/stopwords% statistics of system output: `stats.py`

### Model benchmark
| Model | BLEU-1 | BLEU-2 | BLEU-4 | Stopword% | Dis-1 | Dis-2 | Dis-3 | Dis-4 |
| - | - | - | - | - | - | - | - | - |
| 1-NV | 14.01 | 3.98 | 1.07 | 58.1% | 0.0091 | 0.0355 | 0.0682 | 0.1018 |
| 2-CV | 14.58 | 4.35 | 1.14 | 54.2% | 0.0108 | 0.0448 | 0.0915 | 0.1465 |
| 3-FV | 15.61 | 4.71 | 1.22 | 52.9% | 0.0118 | 0.0502 | 0.1082 | 0.1778 |
