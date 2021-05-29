# Note that fairseq may use all gpus on your machine and the actual batch-size is times by n_gpus.
# If you use multiple gpus, batch_size should be devided by number of gpus.

# hyper-params
LR=3e-4
DROPOUT=0.1
LAYER=3
WARMUP=6000

# directory to save models
MODEL_DIR="/userhome/shuhe/movie_plus/pre_feature/OpenViDial/mmi_small_object_data_2.0"
# data directory
DATA_DIR="/userhome/shuhe/video-dialogue-model/data/origin_data/21.02.20/preprocess_data"
TYPE="objects"

CUDA_VISIBLE_DEVICES=1 fairseq-train \
  --save-dir $MODEL_DIR \
  --user-dir mmi \
  --task mmi-video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --arch baseline-mmi-obj-transformer \
  --encoder-layers $LAYER \
  --encoder-embed-dim 512 \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 8000 \
  --batch-size 128 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --criterion base-loss \
  --lr $LR \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --max-epoch 20 \
  --keep-last-epochs 20 \
  --ddp-backend=no_c10d \
  --num-workers 40 \
  --fp16