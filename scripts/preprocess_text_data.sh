# Preprocess text data in preprocessed_data/src.txt


OUT_DIR="sample_data/preprocessed_data"  # output dir

# todo 日后加上bpe
## learn bpe
##TOK_PREF="${OUT_DIR}/text.tok"
#TOK_FILE="${OUT_DIR}/src-tgt.tgt"  # we use tgt to learn bpe because each sentence in origin corpus appears in tgt once. todo except for the first sentence!
#
#BPE_CODE="${OUT_DIR}/codes.bpe"
#
#echo "Learn BPE from $TOK_FILE ..."
#subword-nmt learn-bpe -s 10000 < $TOK_FILE > $BPE_CODE
#
#
## apply bpe
#BPE_CODE="/home/mengyuxian/video-dialogue-model/sample_data/preprocessed_data/codes.bpe"
#DATA_DIR="/home/mengyuxian/video-dialogue-model/sample_data/preprocessed_data/"
#for suffix in "src" "tgt"; do
#    f="${DATA_DIR}/src-tgt.${suffix}"
##    of="${DATA_DIR}/src-tgt.${suffix}.bpe"
#    echo "apply_bpe.py to ${f}  ..."
#    subword-nmt apply-bpe -c $BPE_CODE < $f > $f.bpe
#done



# fairseq binarize

# option 1 nmt


# option 2 lm
DATA_DIR="/home/mengyuxian/video-dialogue-model/sample_data/preprocessed_data/"
fairseq_pref="${DATA_DIR}/src-tgt.src"  # todo 区分train/valid/test
echo "Generate vocabulary and train dataset files ..."

fairseq-preprocess \
--only-source  \
--destdir $OUT_DIR \
--trainpref $fairseq_pref \
--validpref $fairseq_pref \
--testpref $fairseq_pref
