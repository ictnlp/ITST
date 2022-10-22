export CUDA_VISIBLE_DEVICES=0,1,2,3

mustc_root=PATH_TO_MUSTC_DATA
lang=de
asr_modelfile=PATH_TO_SAVE_ASR_MODEL
st_modelfile=PATH_TO_SAVE_ST_MODEL

# latency weight in MMA
latency_weight=0.001

python train.py ${mustc_root}/en-${lang} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --user-dir examples/simultaneous_translation \
    --save-dir ${st_modelfile} --num-workers 8  \
    --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --warmup-updates 4000 --max-update 300000 --max-tokens 20000 --seed 2 \
    --label-smoothing 0.1 \
    --load-pretrained-encoder-from ${asr_modelfile}/best.pt \
    --task speech_to_text  \
    --arch convtransformer_simul_trans_espnet  \
    --simul-type infinite_lookback_fixed_pre_decision  \
    --latency-weight-avg ${latency_weight} \
    --fixed-pre-decision-ratio 7 \
    --update-freq 4 \
    --save-interval-updates 1000 \
    --keep-interval-updates 20 \
    --find-unused-parameters \
    --fp16 \
    --log-interval 10