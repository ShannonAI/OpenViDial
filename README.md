# OpenViDial
This repo contains downloading instructions for the two **OpenViDial** datasets in:
* OpenViDial 1.0: **[OpenViDial: A Large-Scale, Open-Domain Dialogue Dataset  with Visual Contexts](https://arxiv.org/pdf/2012.15015.pdf)**
* OpenViDial 2.0: **[OpenViDial 2.0: A Larger-Scale, Open-Domain Dialogue Generation Dataset with Visual Contexts](https://arxiv.org/pdf/2109.12761.pdf)**

and the code to reproduce results based on the two **OpenViDial** datasets in the paper **[Modeling Text-visual Mutual Dependency for Multi-modal dialog Generation](https://arxiv.org/pdf/2105.14445.pdf)**

## Introduction
When humans converse, what a speaker will
say next significantly depends on what he sees. OpenViDial is a largescale
multi-module dialogue dataset for this purpose. Thes dialogue
turns and visual contexts are extracted
from movies and TV series, where each dialogue
turn is paired with the corresponding
visual context in which it takes place. Up to **2022.01.22** OpenViDial has two verseion: **OpenViDial 1.0** and **OpenViDial 2.0**. For **OpenViDial 1.0**, it contains a total number of 1.1 million
dialogue turns, and thus 1.1 million visual contexts
stored in images. For **OpenViDial 2.0**, it is much larger than the previous version OpenViDial 1.0 containing a total number of 5.6 million
dialogue turns along with 5.6 million visual contexts
stored in images.

The following are  two short conversations where visual contexts are crucial.

<div align="center">
  <img src="demo_data/dataset.png"/>
</div>

## Detailed and Downloading Instructions
For the detailed and downloading instructions for the two **OpenViDial** datasets (OpenViDial 1.0, OpenViDial 2.0) can be found [here](datasets/README.md)

#### Noted
If you'd like to take a glance at the a sample of the dataset instead of downloading the full dataset, we provide a data sample [here](https://drive.google.com/drive/folders/17XjJ612wMolkrU-ESW5yv6MnbaclrzoM?usp=sharing). The small size data are sampled from OpenViDial 1.0 dataset, and can be used for debug or any other operations.

## Vanilla Visual Dialog Models
We proposed three models for this dataset. Please refer to the paper for details:
* **Model #1 - NoVisual**: use only dialog texts without visual information

<div align="center">
  <img src="demo_data/model1.png"/>
</div>

* **Model #2 - CoarseVisual**: use texts and a pretrained ResNet50 on ImageNet to compute 1000-d feature from each picture

<div align="center">
  <img src="demo_data/model2.png"/>
</div>

* **Model #3 - FineVisual**: use texts and a pretrained Faster R-CNN on Genome to compute 2048-d * K objects features from each picture

<div align="center">
  <img src="demo_data/model3.png"/>
</div>

### Requirements
* python >= 3.6
* `pip install -r requirements.txt`

### Preprocess directory structure
`preprocessed_data_dir` is a directory that contains all the preprocessed files (text, image feature mmap, offsets, etc.)
generated from [origin_data_dir](#detailed-and-downloading-instructions) and we use them in training models. 
The directory structure is shown below.

**Note: every `train*` file or directory should have a 'valid' and a 'test' counterpart, we ignore them below for simplicity.**
```
├──preprocessed_data_dir
      └── train.features.mmap  // numpy mmap array file of shape [num_sents, 1000], each row is a 1000-d ResNet-50 feature
      └── train.objects.mmap  // numpy mmap array file of shape [num_sents, 20, 2048],  faster-rcnn object feature file, each row contain 20 objects feature, which is 2048-d
      └── train.objects_mask.mmap  // numpy mmap array file of shape [num_sents, 20],  faster-rcnn mask file, each row contain 20 objects mask, 1 for valid, 0 for mask
      └── train.offsets.npy  // numpy array file of shape [num_episodes], each item is the offsets of one episode
      └── train.sent_num.npy // numpy array file of shape [num_episodes], each item is the sentence number of one episode
```

### Preprocess text data
We use Moses Tokenizer to tokenize texts and generate offsets arrays:
`bash ./scripts/preprocess_video_data.sh`
and followed with byte-pair-encoding and fairseq-preprocess binarization:
`bash ./scripts/preprocess_text_data.sh`

**Note: You need to change `DATA_DIR, ORIGIN_DIR, OUTPUT_DIR` to your own path.**

#### Download the pre-computed CNN features and Faster-RCNN features
CNN-pooling features is used for Model #2 - CoarseVisual and Faster R-CNN features is used for Model #3 - FineVisual. You can directly download the pre-computed files for CNN and Faster-RCNN features [here](./datasets/README.md) for either OpenViDial 1.0 dataset or OpenViDial 2.0 dataset.

#### (Optional) Extract features on your own
If you want to extract some feature on your own, or you'd like to know details of extracting visual features, 
see [video_dialogue_model/extract_features/extract_features.md](video_dialogue_model/extract_features/extract_features.md)

**Note: Extracting features will take you too much time.**

### Train and Evaluate Model #1 - NoVisual
`bash scripts/reproduce_baselines/text_only.sh` will train and evaluate NoVisual, 
Remember to change `MODEL_DIR` and `DATA_DIR` for your setup. 

**Note:** `fairseq` may use all gpus on your machine and the actual batch size is times by number of gpus.
Therefore, if you use multiple gpus, batch size should be devided by number of gpus.

### Train and Evaluate Model #2 - CoarseVisual
`bash scripts/reproduce_baselines/text_and_img_feature.sh` will train and evaluate CoarseVisual.
Remember to change `MODEL_DIR` and `DATA_DIR` for your setup. Please make sure you use one single gpu to reproduce our results.

### Train and Evaluate Model #3 - FineVisual
`bash scripts/reproduce_baselines/text_and_img_objects.sh` will train and evaluate FineVisual, 
Remember to change `MODEL_DIR` and `DATA_DIR` for your setup. Please make sure you use one single gpu to reproduce our results.

## MMI
### Prepare training data
For NV seeing [./mmi/text/README.md](./mmi/text/README.md). The structure of training data used in both CV and FV is same as the former part.

### Train and Evaluate Model #4 - MI-NV
`bash ./mmi/text/train.sh && bash ./mmi/text/mmi_generate.sh` will train and evaluate MI-NV. Remember to change all the `MODEL_DIR` and `DATA_DIR` for your setup. Please make sure you use one signle gpu to reproduce our results.

### Train and Evaluate Model #5 - MI-CV
`bash ./mmi/feature/scrtpts/train_image.sh && bash ./mmi/feature/scrtpts/mmi_feature_generate.sh` will train and evaluate MI-CV. Remember to change all the `MODEL_DIR` and `DATA_DIR` for your setup. Please make sure you use one signle gpu to reproduce our results.

### Train and Evaluate Model #6 - MI-NV
`bash ./mmi/feature/scrtpts/train_object.sh && bash ./mmi/feature/scrtpts/mmi_object_generate.sh` will train and evaluate MI-FV. Remember to change all the `MODEL_DIR` and `DATA_DIR` for your setup. Please make sure you use one signle gpu to reproduce our results.

### Other Statistics
* get diversity statistics of system output: `train/stats.py`
* get rouge statistics of system output: `train/rouge.py`

### Model benchmark
#### 1. On OpenViDial 1.0 Dataset
| Model | BLEU-1 | BLEU-2 | BLEU-4 | Dis-1 | Dis-2 | Dis-3 | Dis-4 | ROUGE-1 | ROUGE-2 | ROUGE-4 |
| - | - | - | - | - | - | - | - | - | - | - |
| 1-NV | 14.06 | 3.80 | 0.95 | 0.0006 | 0.0019 | 0.0031 | 0.0043 | 0.06787 | 0.01464 | 0.00224 |
| 2-CV | 14.70 | 4.38 | 1.14 | 0.0023 | 0.0090 | 0.0178 | 0.0272 | 0.08773 | 0.02067 | 0.00347 |
| 3-FV | 14.85 | 4.61 | 1.19 | 0.0026 | 0.0112 | 0.0246 | 0.0406 | 0.09083 | 0.02085 | 0.00329 |
| 4-MI-NV | 14.27 | 3.89 | 0.99 | 0.0006 | 0.0022 | 0.0036 | 0.0043 | 0.06918 | 0.01497 | 0.00238 |
| 5-MI-CV | 14.77 | 4.46 | 1.16 | 0.0023 | 0.0091 | 0.0181 | 0.0272 | 0.08791 | 0.02077 | 0.00350 |
| 6-MI-FV | 14.95 | 4.67 | 1.22 | 0.0027 | 0.0117 | 0.0261 | 0.0433 | 0.09100 | 0.02090 | 0.00338 |

#### 2. On OpenViDial 2.0 Dataset
| Model | BLEU-4 | Dis-1 | Dis-2 | Dis-3 | Dis-4 |
| - | - | - | - | - | - |
| 1-NV | 1.95 | 0.0037 | 0.0302 | 0.0929 | 0.1711 |
| 2-CV | 1.97 | 0.0041 | 0.0353 | 0.0999 | 0.1726 |
| 3-FV | 1.99 | 0.0056 | 0.0431 | 0.1250 | 0.2215 |
| 4-MI-NV | 1.96 | 0.0039 | 0.0311 | 0.0953 | 0.1630 |
| 5-MI-CV | 1.98 | 0.0047 | 0.0392 | 0.1093 | 0.1774 |
| 6-MI-FV | 2.00 | 0.0060 | 0.0460 | 0.1321 | 0.2311 |

#### Noted
The size of OpenViDial 2.0 dataset is too much larger than that of OpenViDial 1.0 dataset. To make the results reproducibility we didn't use the all features for CoarseVisual and FineVisual model (only 5% in this experiments), since the full features will occupy too much memory and may not avaliable for most researchers.