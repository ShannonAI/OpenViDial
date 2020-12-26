# 1. extract data from origin jsonl file

#ORIGIN_DIR="/userhome/yuxian/data/video/origin_data"
#MMI_DIR="/userhome/yuxian/data/video/preprocessed_mmi_data"
ORIGIN_DIR="sample_data/origin_data"
MMI_DIR="sample_data/preprocessed_mmi_data"
mkdir -p $MMI_DIR

for split in "valid" "test" "train"; do
  python preprocess_nmt_data.py \
    --origin-dir $ORIGIN_DIR \
    --output-dir $MMI_DIR \
    --split $split
done

# 2. apply bpe
codes_file="/data/yuxian/datasets/new-video/preprocessed_data/codes.30000.bpe"
for split in "train" "valid" "test"; do
  for suffix in "src" "tgt"; do
      fin="${MMI_DIR}/${split}.src-tgt.${suffix}"
      fout="${MMI_DIR}/${split}.bpe.src-tgt.${suffix}"
      echo "apply_bpe.py to ${fin}  ..."
      subword-nmt apply-bpe -c ${codes_file} < $fin > $fout
  done
done


# 3. fairseq binarize
rm $MMI_DIR/dict* remove dict
fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref "${MMI_DIR}/train.bpe.src-tgt" \
    --validpref "${MMI_DIR}/valid.bpe.src-tgt" \
    --testpref "${MMI_DIR}/test.bpe.src-tgt" \
    --workers 8 --destdir $MMI_DIR
