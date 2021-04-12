export CUDA_VISIBLE_DEVICES=0

# 1. normal generation with nbest list

DATA_DIR="/userhome/yuxian/data/video/preprocessed_data"
MODEL_DIR="/userhome/shuhe/movie_plus/pre_feature/OpenViDial/feature_result"
TYPE="features"
MODEL_PATH="${MODEL_DIR}/checkpoint_best.pt"
NBEST=10
BEAM=10
SUBSET="test"
NBEST_FILE="${MODEL_DIR}/${SUBSET}_gen.out.${NBEST}best"


python ./train/generate.py \
  --user-dir video_dialogue_model \
  --task video-dialogue \
  --img-type $TYPE \
  --data-dir $DATA_DIR \
  --path $MODEL_PATH \
  --beam $BEAM \
  --batch-size 16 \
  --remove-bpe \
  --gen-subset $SUBSET \
  --nbest $NBEST \
  --quiet \
  >$NBEST_FILE 2>&1 & tail -f $NBEST_FILE 2>&1
  
# 2. split nbest to different directorys
NBEST_DIR="${MODEL_DIR}/${SUBSET}_best${NBEST}"
python ./scripts/mmi/split_nbest.py \
--nbest-file $NBEST_FILE \
--target-dir $NBEST_DIR \
--nbest $NBEST

echo "copy ..."
cp $DATA_DIR/test.features.mmap $NBEST_DIR/

# 3. score backwardly
codes_file="/data/yuxian/datasets/new-video/preprocessed_data/codes.30000.bpe"
dict_file="/data/yuxian/datasets/new-video/preprocessed_data/dict.txt"
backward_model="/data/yuxian/datasets/new-video/mmi_text/checkpoint_best.pt"

for sub_dir in $(ls ${NBEST_DIR}); do
  sub_dir="${NBEST_DIR}/${sub_dir}"
  echo "compute backward score of ${sub_dir}"
  python ./mmi_fairseq/feature/scrtpts/combine_new_test.py \
  --src-dir $DATA_DIR \
  --nbest-file $sub_dir/src-tgt.src \
  --target-dir $sub_dir/test_feature.src.txt

  subword-nmt apply-bpe -c ${codes_file} < $sub_dir/test_feature.src.txt > $sub_dir/test_feature_bpe.src

  fairseq-preprocess --source-lang src --srcdict $dict_file \
    --only-source \
    --testpref "${sub_dir}/test_feature_bpe.src" \
    --workers 8 --destdir $NBEST_DIR
  
  # backward generation
  out_file="${sub_dir}/gen.out"
  python ./mmi_fairseq/feature/generate.py \
  --user-dir mmi_fairseq \
  --task mmi-video-dialogue \
  --img-type $TYPE \
  --data-dir $NBEST_DIR \
  --path $backward_model \
  --batch-size 32 \
  --gen-subset "test" \
  --score-target-file $sub_dir/scores.backward

done

# 4. weight average score.forward and score.backward for MMI generation
ALPHA1=0.4
ALPHA2=0.3
ALPHA3=0.3
BIRECTION_OUTPUT="${NBEST_DIR}/bidirection${ALPHA}.out"
python scripts/mmi/combine_bidirectional_score.py \
  --nbest-dir=/userhome/shuhe/movie_plus/pre_feature/OpenViDial/mmi_small_text \
  --nbest-dir-feature $NBEST \
  --type feature \
  --output-file=$BIRECTION_OUTPUT \
  --alpha $ALPHA1 \
  --alpha-2 $ALPHA2 \
  --alpha-3 $ALPHA3 

# 5. grep reference from output-file and score
SYS_OUTPUT=${BIRECTION_OUTPUT}
REFERENCE="${MODEL_DIR}/${SUBSET}_gen.ref"
grep ^T $NBEST_FILE | cut -f2- > $REFERENCE

fairseq-score \
-s $SYS_OUTPUT \
-r $REFERENCE