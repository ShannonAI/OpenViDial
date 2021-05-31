DATA_DIR="/data/yuxian/datasets/new-video/preprocessed_mmi_data"
SAVE_DIR="/data/yuxian/datasets/new-video/mmi_text"

LAYER=3
DROPOUT=0.3
WARMUP=6000
mkdir -p $SAVE_DIR

# train
CUDA_VISIBLE_DEVICES=3 fairseq-train $DATA_DIR \
    -s src -t tgt \
    --ddp-backend='no_c10d' \
    --arch transformer \
    --encoder-layers $LAYER \
    --decoder-layers $LAYER \
    --encoder-embed-dim 512 \
    --share-decoder-input-output-embed \
    --max-tokens 8000 \
    --batch-size 128 \
    --max-epoch 20 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --lr 3e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates $WARMUP --warmup-init-lr 1e-07 \
    --dropout $DROPOUT --attention-dropout $DROPOUT \
    --keep-last-epochs 5 \
    --save-dir $SAVE_DIR \
    --no-progress-bar
