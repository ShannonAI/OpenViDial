ORIGIN_DIR="sample_data/origin_data"
OUTPUT_DIR="sample_data/preprocessed_data"

#ORIGIN_DIR="/data/yuxian/datasets/video/origin_data"
#OUTPUT_DIR="/data/yuxian/datasets/video/preprocessed_data"


python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "train" \
--cnn_feature

python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "dev"

python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "test"
