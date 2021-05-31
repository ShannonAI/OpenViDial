# MMI Generation
This directory contains scripts to do MMI-NV generation.

## 1.Train a Forward(src->tgt) Model
See [../../README.md](../../README.md)

## 2. Prepare Backward(tgt->src) Training Data 
`./preprocess.sh`

## 3. Train backward(tgt->src) model 
`./train.sh`

## 4. MMI Generation
`./mmi_generate.sh`
