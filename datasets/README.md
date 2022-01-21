# OpenViDial Datasets
This README contains the detailed and downloading instructions for the **OpenViDial** datasets in:

* OpenViDial 1.0: **[OpenViDial: A Large-Scale, Open-Domain Dialogue Dataset  with Visual Contexts](https://arxiv.org/pdf/2012.15015.pdf)** 
* OpenViDial 2.0: **[OpenViDial 2.0: A Larger-Scale, Open-Domain Dialogue Generation Dataset with Visual Contexts](https://arxiv.org/pdf/2109.12761.pdf)** 
  
## OpenViDial 1.0
**\*\*\*\*\* New March 12th, 2021: New cnn/rcnn feature on test/valid dataset \*\*\*\*\***

We fixed the bug of cnn/rcnn features on valid/test dataset and re-run the experiments on the new data.
Evaluation metrics are also updated.

### Detailed statistics for OpenViDial
| Attribute | value |
| - | - |
|Number of turns| 1.1M|
|Number of images |1.1M|
|Vocab size before BPE | 70K |
|Vocab size after BPE | 30K |
|Average length of each episode |14|
|Average length of each turn|7.6 |

### Download the Original Dataset
The main folder `origin_dir` contains training/valid/test sets, each of which is made up by the following files:
```
├──origin_dir
      └── train.dialogue.jsonl // each line is an episode of dialogue, which a list of IDs.    
      └── train.origin.txt // each line corresponds to a dialogue text utterence, with the ID being its line number (staring with 0).
      └── train_images // containing images (visual contexts) in which the text utterence take place, with ID being the image filename (0,1,2, etc)
            └── 0.jpg
            └── 1.jpg
            └── ...
      └── valid.* (i.e., valid.dialogue.jsonl, valid.origin.txt, valid_images)
      └── test.*  (i.e., test.dialogue.jsonl, test.origin.txt, test_images)
```

Data download:
1. Download `[train|valid|test].origin.txt` and `[train|valid|test].dialogue.jsonl` [here](https://drive.google.com/drive/folders/15qznjUWaIJ-TzT4YTdcgR9-fMumOfjFx?usp=sharing) 
2. Download `test_images` (~ 20G)  [here](https://drive.google.com/file/d/1DgZXlGi_x37nQrJYK4tSLXEvVShBKaZY/view?usp=sharing) 
3. Download `valid_images` (~ 20G) [here](https://drive.google.com/file/d/1J6YMq3Zwqdhi93IZFHi1JoS9xvcZcPfM/view?usp=sharing) 
4. Download train_images: Since the size of train_images is too large (~ 170G), we split it to 12 zip files.  Download seperate files `zip_train`  [here](https://drive.google.com/drive/folders/1Aygv6rTWtvDv7-WLzzOSltHnht_dK80g?usp=sharing). Then download and run `cat.sh` [here](https://drive.google.com/file/d/1GUBBAdm8-1O3a5ZJ5JmkwSBFiFoEp09k/view?usp=sharing) to include all files in the same directory.  
5. Move all files to `origin_dir`. 

### Download the pre-computed CNN features and Faster-RCNN features
To save the time of extracting features for CNN and Faster-RCNN, we provide the pre-computed CNN features and Faster-RCNN features. You just need to download them following the steps and re-construct the directory as [here](../README.md/#preprocess-directory-structure).

##### Download CNN-pooling features
The compression file of preprocessed ResNet50 features (`feature_files.tar.gz`) [(~3.7G) can be downloaded from here](https://drive.google.com/drive/folders/1rLREH7GmlNa9uQKcx_KRp3qPwmDeBZup?usp=sharing). You can get preprocessed ResNet50 features (`*.features.mmap`) by command `tar zxvf feature_files.tar.gz`.

##### Download Faster R-CNN features
The compression file of preprocessed Faster R-CNN objects features (`object_files.tar.gz`) [(~50G) can be downloaded from here](https://drive.google.com/drive/folders/1s4-PPGL_mVBQHeNMwfsJpqJA1cAqJAGd?usp=sharing). You can get preprocessed Faster R-CNN objects features (`*objects.mmap`, `*objects_mask.mmap`) by command `tar zxvf object_files.tar.gz`.

### Checkout
Each of files has a hash value by command `md5sum fileName`. You can get it from [here](https://drive.google.com/file/d/1m8l5HfwN88j3NtXRuc3QbyxNC0AoweSl/view?usp=sharing) and we suggest you check each file's hash value before any operations.

## OpenViDial 2.0

### Detailed statistics for OpenViDial
| Attribute | value |
| - | - |
|Number of turns | 5.6M |
|Number of images |5.6M |
|Vocab size before BPE | 278K |
|Vocab size after BPE | 30K |
|Average length of each episode | 48 |
|Average length of each turn| 8.3 |

### Download the Original Dataset
The main folder `origin_dir` contains training/valid/test sets, each of which is made up by the following files:
```
├──origin_dir
      └── train.dialogue.jsonl // each line is an episode of dialogue, which a list of IDs.    
      └── train.origin.txt // each line corresponds to a dialogue text utterence, with the ID being its line number (staring with 0).
      └── train_images // containing images (visual contexts) in which the text utterence take place, with ID being the image filename (0,1,2, etc)
            └── 0.jpg
            └── 1.jpg
            └── ...
      └── valid.* (i.e., valid.dialogue.jsonl, valid.origin.txt, valid_images)
      └── test.*  (i.e., test.dialogue.jsonl, test.origin.txt, test_images)
```

Data download:
1. Download `[train|valid|test].origin.txt` and `[train|valid|test].dialogue.jsonl` [here](https://drive.google.com/drive/folders/1jeTTqSb2ejFmCvu2qS8v9BAh1Yoqecpn?usp=sharing) 
2. Download `test_images` (~ 123G)  [here](https://drive.google.com/drive/folders/1yhqhK5AwBYqbUEGzy90dp-O82_WlKaYa?usp=sharing) 
3. Download `valid_images` (~ 123G) [here](https://drive.google.com/drive/folders/1yhqhK5AwBYqbUEGzy90dp-O82_WlKaYa?usp=sharing) 
4. Download `train_images`: Since the size of `train_images` is too large (~ 1.2T), we split it to 7 zip files.  Download seperate dirctory `train`  [here](https://drive.google.com/drive/folders/1yhqhK5AwBYqbUEGzy90dp-O82_WlKaYa?usp=sharing). Then run the command `cat * > train_images.zip && unzip -d ./train_images train_images.zip` to generate the all images for training set.
5. Move all files to `origin_dir`. 

### Download the pre-computed CNN features and Faster-RCNN features
To save the time of extracting features for CNN and Faster-RCNN, we provide the pre-computed CNN features and Faster-RCNN features. You just need to download them following the steps and re-construct the directory as [here](../README.md/#preprocess-directory-structure).

##### Download CNN-pooling features
The mmap files of preprocessed ResNet50 features for train/valid/test set (`*.features.mmap`) [(~17G, ~2G, ~2G) can be downloaded from here](https://drive.google.com/drive/folders/1SkviBoyBK1cCqz8tiNpCeLgvAkt9KMtU?usp=sharing).

##### Download Faster R-CNN features
The compression file of preprocessed Faster R-CNN objects features (`object_files.tar.gz`) [(~49G) can be downloaded from here](https://drive.google.com/drive/folders/18v3LaJwzylqccdBsL4c1hAX99KDicLze?usp=sharing). You can get preprocessed Faster R-CNN objects features (`*objects.mmap`, `*objects_mask.mmap`) by command `tar zxvf object_files.tar.gz`.

### Checkout
Each of files has a hash value by command `md5sum fileName`. You can get it from [here](https://drive.google.com/file/d/1oRhOiSmd1sOR8_Nkszrrgdm55I_eMfbt/view?usp=sharing) and we suggest you check each file's hash value before any operations.