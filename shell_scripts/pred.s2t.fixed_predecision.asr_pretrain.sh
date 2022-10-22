export CUDA_VISIBLE_DEVICES=0

mustc_root=PATH_TO_MUSTC_DATA
lang=de
modelfile=PATH_TO_SAVE_MODEL
last_file=LAST_CHECKPOINT

# average last 5 checkpoints
python scripts/average_checkpoints3.py --inputs ${modelfile} --num-update-checkpoints ${n} \
    --output ${modelfile}/average-model.pt --last_file ${last_file}
file=${modelfile}/average-model.pt 

python fairseq_cli/generate.py ${mustc_root}/en-${lang} --config-yaml config_asr.yaml --gen-subset tst-COMMON_asr \
--task speech_to_text --path ${file} --max-tokens 50000 --max-source-position 6000 --beam 1 \
--scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct