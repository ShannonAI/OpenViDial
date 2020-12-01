export CUDA_VISIBLE_DEVICES=3
#fairseq-train \
python train.py \
  --save-dir /data/yuxian/train_logs/debug \
  --user-dir video_dialogue_model \
  --task video-dialogue \
  --img-type "objects" \
  --data-dir sample_data/preprocessed_data \
  --arch baseline-obj-transformer \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --encoder-embed-dim 512 \
  --dropout 0.0 \
  --optimizer adam \
  --max-sentences 1 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --lr 1e-4 \
  --weight-decay 0.0001 \
  --max-epoch 100 \
  --keep-last-epochs 5 \
  --ddp-backend=no_c10d \
