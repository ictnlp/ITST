mustc_root=PATH_TO_MUSTC_DATA
lang=de
tar -xzvf ${mustc_root}/MUSTC_v1.0_en-${lang}.tar.gz

# prepare raw mustc data
python3 examples/speech_to_text/prep_mustc_data_raw_joint.py \
    --data-root ${mustc_root} --tgt-lang ${lang}

# prepare vocabulary
python3 examples/speech_to_text/prep_vocab.py \
    --data-root ${mustc_root} \
    --vocab-type unigram --vocab-size 10000 --joint \
    --tgt-lang ${lang}

# generate the wav list and reference file for SimulEval
eval_data=PATH_TO_SAVE_EVAL_DATA # such as ${mustc_root}/en-de-eval
for split in dev tst-COMMON tst-HE
do
    python examples/speech_to_text/seg_mustc_data.py \
    --data-root ${mustc_root} --lang ${lang} \
    --split ${split} --task st \
    --output ${eval_data}/${split}
done