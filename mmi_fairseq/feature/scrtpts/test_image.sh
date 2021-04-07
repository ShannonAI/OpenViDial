DATA_DIR="/data/wangshuhe/test_mmi"
TYPE="features"
MODEL_PATH="/home/wangshuhe/shuhework/OpenViDial/mmi_test/checkpoint_best.pt"
NBEST=10
BEAM=10
SUBSET="test"


CUDA_VISIBLE_DEVICES=2 python ./mmi_fairseq/feature/generate.py \
  --user-dir mmi_fairseq \
  --task mmi-video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --path $MODEL_PATH \
  --batch-size 5 \
  --remove-bpe \
  --gen-subset $SUBSET