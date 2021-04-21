export CUDA_VISIBLE_DEVICES=0

# 1. normal generation with nbest list

DATA_DIR="/userhome/yuxian/data/video/preprocessed_data"
MODEL_DIR="/userhome/shuhe/movie_plus/pre_feature/OpenViDial/object_result"
TYPE="objects"
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
  >$NBEST_FILE 2>&1 & tail -f $NBEST_FILE 2>&1
  
# 2. split nbest to different directorys
NBEST_DIR="${MODEL_DIR}/${SUBSET}_best${NBEST}"
python ./scripts/mmi/text_only/split_nbest.py \
--nbest-file $NBEST_FILE \
--target-dir $NBEST_DIR \
--nbest $NBEST

echo "copy objects ..."
cp $DATA_DIR/test.objects.mmap.20 $NBEST_DIR/test.objects.mmap
cp $DATA_DIR/test.objects_mask.mmap.20 $NBEST_DIR/test.objects_mask.mmap
cp $DATA_DIR/test.offsets.npy $NBEST_DIR/
cp $DATA_DIR/test.sent_num.npy $NBEST_DIR/

# 3. score backwardly
codes_file="/userhome/yuxian/data/video/preprocessed_data/codes.30000.bpe"
dict_file="/userhome/yuxian/data/video/preprocessed_data/dict.txt"
backward_model="/userhome/shuhe/movie_plus/pre_feature/OpenViDial/mmi_small_object/checkpoint_best.pt"

for sub_dir in $(ls ${NBEST_DIR}); do
  sub_dir="${NBEST_DIR}/${sub_dir}"
  if [ -f "$sub_dir" ]; then
    continue
  fi
  echo "compute backward score of ${sub_dir}"
  python ./mmi_fairseq/feature/scrtpts/combine_new_test.py \
  --src-dir $DATA_DIR \
  --nbest-file $sub_dir/src-tgt.src \
  --target-dir $sub_dir/test_object.src.txt

  subword-nmt apply-bpe -c ${codes_file} < $sub_dir/test_object.src.txt > $sub_dir/test_object_bpe.src

  fairseq-preprocess --source-lang src --srcdict $dict_file \
    --only-source \
    --testpref "${sub_dir}/test_object_bpe" \
    --workers 8 --destdir $NBEST_DIR

  mv $NBEST_DIR/dict.src.txt $NBEST_DIR/dict.txt
  mv $NBEST_DIR/test.src-None.src.bin $NBEST_DIR/test.bin
  mv $NBEST_DIR/test.src-None.src.idx $NBEST_DIR/test.idx
  
  # backward generation
  out_file="${sub_dir}/gen.out"
  python ./mmi_fairseq/feature/scrtpts/generate.py \
  --user-dir mmi_fairseq \
  --task mmi-video-dialogue \
  --img-type $TYPE \
  --data-dir $NBEST_DIR \
  --path $backward_model \
  --batch-size 32 \
  --gen-subset "test" \
  --num-workers 32 \
  --score-target-file $sub_dir/scores.backward

done

FEATURE_NBEST_DIR="${MODEL_DIR}/${SUBSET}_best_feature${NBEST}"
python scripts/mmi/text_only/split_nbest.py \
--nbest-file $NBEST_FILE \
--target-dir $FEATURE_NBEST_DIR \
--nbest $NBEST

echo "copy feature ..."
cp $DATA_DIR/test.features.mmap $FEATURE_NBEST_DIR/
cp $DATA_DIR/test.offsets.npy $FEATURE_NBEST_DIR/
cp $DATA_DIR/test.sent_num.npy $FEATURE_NBEST_DIR/

# 4. feature score backwardly
feature_backward_model="/userhome/shuhe/movie_plus/pre_feature/OpenViDial/mmi_small_feature/checkpoint_best.pt"

for sub_dir in $(ls ${FEATURE_NBEST_DIR}); do
  sub_dir="${FEATURE_NBEST_DIR}/${sub_dir}"
  if [ -f "$sub_dir" ]; then
    continue
  fi
  echo "compute backward score of ${sub_dir}"
  python ./mmi_fairseq/feature/scrtpts/combine_new_test.py \
  --src-dir $DATA_DIR \
  --nbest-file $sub_dir/src-tgt.src \
  --target-dir $sub_dir/test_feature.src.txt

  subword-nmt apply-bpe -c ${codes_file} < $sub_dir/test_feature.src.txt > $sub_dir/test_feature_bpe.src

  fairseq-preprocess --source-lang src --srcdict $dict_file \
    --only-source \
    --testpref "${sub_dir}/test_feature_bpe" \
    --workers 8 --destdir $FEATURE_NBEST_DIR

  mv $FEATURE_NBEST_DIR/dict.src.txt $FEATURE_NBEST_DIR/dict.txt
  mv $FEATURE_NBEST_DIR/test.src-None.src.bin $FEATURE_NBEST_DIR/test.bin
  mv $FEATURE_NBEST_DIR/test.src-None.src.idx $FEATURE_NBEST_DIR/test.idx
  
  # backward generation
  out_file="${sub_dir}/gen.out"
  python ./mmi_fairseq/feature/scrtpts/generate.py \
  --user-dir mmi_fairseq \
  --task mmi-video-dialogue \
  --img-type "features" \
  --data-dir $FEATURE_NBEST_DIR \
  --path $feature_backward_model \
  --batch-size 32 \
  --gen-subset "test" \
  --score-target-file $sub_dir/scores.backward

done

# 5. text backward score
TEXT_NBEST_DIR="${MODEL_DIR}/${SUBSET}_best_text${NBEST}"
python scripts/mmi/text_only/split_nbest.py \
--nbest-file $NBEST_FILE \
--target-dir $TEXT_NBEST_DIR \
--nbest $NBEST

test_backward_model="/userhome/shuhe/movie_plus/pre_feature/OpenViDial/mmi_small_text/checkpoint_best.pt"

for sub_dir in $(ls ${TEXT_NBEST_DIR}); do
  sub_dir="${TEXT_NBEST_DIR}/${sub_dir}"
  echo "compute text backward score of ${sub_dir}"
  # apply bpe
  for suffix in "src" "tgt"; do
    fin="${sub_dir}/src-tgt.${suffix}"
    fout="${sub_dir}/bpe.src-tgt.${suffix}"
    echo "apply_bpe to ${fin}  ..."
    subword-nmt apply-bpe -c ${codes_file} < $fin > $fout
  done

  # binarize
  rm $sub_dir/dict*
  fairseq-preprocess --source-lang src --target-lang tgt --srcdict $dict_file --tgtdict $dict_file \
    --testpref "${sub_dir}/bpe.src-tgt" \
    --workers 8 --destdir $sub_dir

  # backward generation
  out_file="${sub_dir}/gen.out"
  fairseq-generate \
    "${sub_dir}" \
    --score-reference \
    --batch-size 32 \
    --remove-bpe \
    --path $test_backward_model \
    --gen-subset "test" \
    > $out_file

  # extract backward score file
  text_backward_score="${sub_dir}/scores.backward"
  grep ^H "${out_file}" | cut -f 2 >"${text_backward_score}"

done


# 6. weight average score.forward and score.backward for MMI generation
ALPHA1=0.4
ALPHA2=0.1
ALPHA3=0.1
ALPHA4=0.4
BIRECTION_OUTPUT="${NBEST_DIR}/bidirection${ALPHA}.out"
python scripts/mmi/text_only/combine_bidirectional_score.py \
  --nbest-dir=$TEXT_NBEST_DIR \
  --nbest-dir-feature=$FEATURE_NBEST_DIR \
  --nbest-dir-object=$NBEST_DIR \
  --type "object" \
  --output-file=$BIRECTION_OUTPUT \
  --alpha $ALPHA1 \
  --alpha-2 $ALPHA2 \
  --alpha-3 $ALPHA3 \
  --alpha-4 $ALPHA4

# 7. grep reference from output-file and score
SYS_OUTPUT=${BIRECTION_OUTPUT}
REFERENCE="${MODEL_DIR}/${SUBSET}_gen.ref"
grep ^T $NBEST_FILE | cut -f2- > $REFERENCE

fairseq-score \
-s $SYS_OUTPUT \
-r $REFERENCE