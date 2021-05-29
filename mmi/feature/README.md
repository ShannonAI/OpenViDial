# MMI Generation
This directory contains scripts to do MMI-CV and MMI-FV generation.

## 1.Train a Forward(src->tgt) Model
See [../../README.md](../../README.md)

## 2. Prepare Backward(tgt->src) Training Data 
Training data for MMI is just like the training data for forward model, seeing [../../README.md](../../README.md)

## 3. Train backward(tgt->src) model 
`./scrtpts/train_object.sh` and `./scrtpts/train_image.sh`

## 4. MMI Generation
`./scrtpts/mmi_feature_generate.sh` and `./scrtpts/mmi_object_generate.sh`