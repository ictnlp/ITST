export CUDA_VISIBLE_DEVICES=0
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
last_file=LAST_CHECKPOINT
ref_dir=PATH_TO_REFERENCE
detok_ref_dir=PATH_TO_DETOKENIZED_REFERENCE
mosesdecoder=PATH_TO_MOSESD # https://github.com/moses-smt/mosesdecoder
src_lang=SOURCE_LANGUAGE
tgt_lang=TARGET_LANGUAGE

threshold=TEST_THRESHOLD # test threshold in ITST, such as 0.8

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
    --output ${modelfile}/average-model.pt --last_file ${last_file}
file=${modelfile}/average-model.pt 

# generate translation
python fairseq_cli/sim_generate.py ${data} --path ${file} \
    --batch-size 1 --beam 1 --left-pad-source --fp16 --remove-bpe \
    --itst-decoding --itst-test-threshold ${threshold} > pred.out 2>&1

# latency
echo -e "\nLatency"
tail -n 4 pred.out

# BLEU
echo -e "\nBLEU"
grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref_dir} < pred.translation

# SacreBLEU
echo -e "\nSacreBLEU"
perl ${mosesdecoder}/scripts/tokenizer/detokenizer.perl -l ${tgt_lang} < pred.translation > pred.translation.detok
cat pred.translation.detok | sacrebleu ${detok_ref_dir} --w 2
