
ORIGIN_DIR="/userhome/yuxian/data/video/origin_data"
OUTPUT_DIR="/userhome/yuxian/data/video/preprocessed_data"


python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "train" 

python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "valid" 


python preprocess_video_data.py \
--origin-dir $ORIGIN_DIR \
--output-dir $OUTPUT_DIR \
--split "test"
