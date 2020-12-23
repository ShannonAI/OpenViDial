export CUDA_VISIBLE_DEVICES=0

LR=1e-4
DROPOUT=0.3
LAYER=3

LOG_DIR="/data/yuxian/train_logs/video/layer${LAYER}_lr${LR}_bsz128_drop${DROPOUT}"
DATA_DIR="/userhome/yuxian/data/video/preprocessed_data"
#DATA_DIR="/data/yuxian/datasets/video/preprocessed_data"
#DATA_DIR="/data/yuxian/datasets/new-video/preprocessed_data"

#fairseq-train \
python train.py \
  --save-dir $LOG_DIR \
  --user-dir video_dialogue_model \
  --task video-dialogue \
  --img-type "features" \
  --data-dir $DATA_DIR \
  --arch baseline-img-transformer \
  --encoder-layers $LAYER \
  --decoder-layers $LAYER \
  --encoder-embed-dim 512 \
  --share-decoder-input-output-embed \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 4000 \
  --batch-size 128 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr $LR \
  --weight-decay 0.0001 \
  --max-epoch 100 \
  --keep-last-epochs 5 \
  --ddp-backend=no_c10d


# todo 调整warmup, batchsie, lr, dropout
