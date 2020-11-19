python preprocess_video_data.py \
--origin-dir "sample_data/origin_data" \
--output-dir "sample_data/preprocessed_data" \
--split "train"

python preprocess_video_data.py \
--origin-dir "sample_data/origin_data" \
--output-dir "sample_data/preprocessed_data" \
--split "dev"

python preprocess_video_data.py \
--origin-dir "sample_data/origin_data" \
--output-dir "sample_data/preprocessed_data" \
--split "test"
