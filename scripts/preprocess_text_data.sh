
DATA_DIR="/userhome/yuxian/data/video/preprocessed_data"

# learn bpe
num_operations=30000
train_file=$DATA_DIR/train.src.txt
codes_file=$DATA_DIR/codes.${num_operations}.bpe
subword-nmt learn-bpe -s ${num_operations} < ${train_file} > ${codes_file}

# apply bpe
for split in "train" "valid" "test"; do
    f="${DATA_DIR}/${split}.src.txt"
    echo "apply_bpe.py to ${f}  ..."
    subword-nmt apply-bpe -c ${codes_file} < $f > $f.bpe
done



# fairseq binarize
# We use '--only-source' here because we want to determine window size later.
echo "Generate vocabulary and train dataset files ..."
fairseq-preprocess \
--only-source  \
--destdir $DATA_DIR \
--trainpref $DATA_DIR/train.src.txt.bpe \
--validpref $DATA_DIR/valid.src.txt.bpe \
--testpref $DATA_DIR/test.src.txt.bpe \
--workers 1
