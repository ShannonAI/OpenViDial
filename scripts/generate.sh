#fairseq-generate \
python generate.py \
  --user-dir video_dialogue_model \
  --task video-dialogue \
  --data-dir sample_data/preprocessed_data \
  --path /data/yuxian/train_logs/debug/checkpoint_best.pt \
  --beam 5 \
  --remove-bpe
