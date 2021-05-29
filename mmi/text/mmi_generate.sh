export CUDA_VISIBLE_DEVICES=3

# 1. normal generation with nbest list

DATA_DIR="/data/yuxian/datasets/new-video/preprocessed_data"
MODEL_DIR="/data/yuxian/datasets/new-video/models/object_lr2e-4"
TYPE="objects"
MODEL_PATH="${MODEL_DIR}/checkpoint_best.pt"
NBEST=5
BEAM=5
#SUBSET="valid"
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
python ./mmi/text/split_nbest.py \
--nbest-file $NBEST_FILE \
--target-dir $NBEST_DIR \
--nbest $NBEST

# 3. score backwardly
codes_file="/data/yuxian/datasets/new-video/preprocessed_data/codes.30000.bpe"
dict_file="/data/yuxian/datasets/new-video/preprocessed_data/dict.txt"
backward_model="/data/yuxian/datasets/new-video/mmi_text/checkpoint_best.pt"

for sub_dir in $(ls ${NBEST_DIR}); do
  sub_dir="${NBEST_DIR}/${sub_dir}"
  echo "compute backward score of ${sub_dir}"
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
    --path $backward_model \
    --gen-subset "test" \
    > $out_file

  # extract backward score file
  backward_score="${sub_dir}/scores.backward"
  grep ^H "${out_file}" | cut -f 2 >"${backward_score}"

done


# 4. weight average score.forward and score.backward for MMI generation
ALPHA=0.037
BIRECTION_OUTPUT="${NBEST_DIR}/bidirection${ALPHA}.out"
python ./mmi/text/combine_bidirectional_score.py \
  --nbest-dir=$NBEST_DIR \
  --output-file=$BIRECTION_OUTPUT \
  --alpha $ALPHA


# 5. grep reference from output-file and score
SYS_OUTPUT=${BIRECTION_OUTPUT}
REFERENCE="${MODEL_DIR}/${SUBSET}_gen.ref"
grep ^T $NBEST_FILE | cut -f2- > $REFERENCE

fairseq-score \
-s $SYS_OUTPUT \
-r $REFERENCE
