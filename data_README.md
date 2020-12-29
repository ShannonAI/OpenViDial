# Directory structure
Note: every `train*` file or directory should have a 'valid' and a 'test' counterpart, we ignore them below for simplicity.
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
``'
