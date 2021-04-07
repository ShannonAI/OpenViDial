# Note that fairseq may use all gpus on your machine and the actual batch-size is times by n_gpus.
# If you use multiple gpus, batch_size should be devided by number of gpus.

# hyper-params
LR=3e-4
DROPOUT=0.1
LAYER=3
WARMUP=6000

# directory to save models
MODEL_DIR="/home/wangshuhe/shuhework/OpenViDial/mmi_test"
# data directory
DATA_DIR="/data/wangshuhe/test_mmi"
TYPE="features"

CUDA_VISIBLE_DEVICES=3 fairseq-train \
  --save-dir $MODEL_DIR \
  --user-dir mmi_fairseq \
  --task mmi-video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --arch baseline-mmi-img-transformer \
  --encoder-layers $LAYER \
  --encoder-embed-dim 512 \
  --dropout $DROPOUT \
  --optimizer adam \
  --max-tokens 100000 \
  --batch-size 150 \
  --adam-betas "(0.9,0.999)" \
  --reset-optimizer \
  --criterion base-loss \
  --lr $LR \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --max-epoch 20 \
  --keep-last-epochs 5 \
  --ddp-backend=no_c10d