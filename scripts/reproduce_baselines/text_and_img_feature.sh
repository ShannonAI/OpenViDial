# hyper-params
LR=3e-4
DROPOUT=0.3
LAYER=3
WARMUP=6000

# directory to save models
MODEL_DIR="/data/yuxian/train_logs/video/text_and_image_feature/layer${LAYER}_lr${LR}_bsz128_drop${DROPOUT}_warmup${WARMUP}"
# data directory
DATA_DIR="/data/yuxian/datasets/new-video/preprocessed_data"
TYPE="features"

fairseq-train \
  --save-dir $MODEL_DIR \
  --user-dir video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
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
  --lr $LR \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
  --max-epoch 20 \
  --keep-last-epochs 5 \
  --ddp-backend=no_c10d \
  --use-img \

# generate system predictions to OUTPUT
MODEL_PATH="${MODEL_DIR}/checkpoint_best.th"
OUTPUT="${MODEL_DIR}/gen.out"
python ./train/generate.py \
  --user-dir video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --path $MODEL_PATH \
  --beam 5 \
  --batch-size 32 \
  --remove-bpe \
  --gen-subset "test" >$OUTPUT 2>&1 & tail -f $OUTPUT 2>&1