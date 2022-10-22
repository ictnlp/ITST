export CUDA_VISIBLE_DEVICES=0

mustc_root=PATH_TO_MUSTC_DATA
lang=de
modelfile=PATH_TO_SAVE_MODEL
last_file=LAST_CHECKPOINT

wav_list=FILE_OF_SRC_AUDIO_LIST
reference=FILE_OF_TGT_REFERENCE
output_dir=DIR_TO_OUTPUT


# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt --last_file ${last_file}
file=${modelfile}/average-model.pt 

simuleval --agent examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py \
    --source ${wav_list} \
    --target ${reference} \
    --data-bin ${MUSTC_ROOT}/en-${LANG} \
    --config config_st.yaml \
    --model-path ${file} \
    --output ${output_dir} \
    --scores --gpu \
    --port 1234