#ORIGIN_DIR="sample_data/origin_data"
#OUTPUT_DIR="sample_data/preprocessed_data"

#ORIGIN_DIR="/data/yuxian/datasets/video/origin_data"
#OUTPUT_DIR="/data/yuxian/datasets/video/preprocessed_data"
ORIGIN_DIR="/userhome/yuxian/data/video/origin_data"
OUTPUT_DIR="/userhome/yuxian/data/video/preprocessed_data"



python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "train" \
--rcnn_feature \
--cnn_feature

python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "valid" \
--rcnn_feature \
--cnn_feature \

python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "test" \
--rcnn_feature \
--cnn_feature \
