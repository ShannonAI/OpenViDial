export CUDA_VISIBLE_DEVICES=1

LR=3e-4
DROPOUT=0.3
LAYER=3
WARMUP=6000


LOG_DIR="/userhome/yuxian/train_logs/video/text_only/layer${LAYER}_lr${LR}_bsz128_drop${DROPOUT}_warmup${WARMUP}"
DATA_DIR="/userhome/yuxian/data/video/preprocessed_data"


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
  --max-tokens 8000 \
  --batch-size 128 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --lr $LR \
  --max-epoch 20 \
  --keep-last-epochs 5 \
  --ddp-backend=no_c10d
