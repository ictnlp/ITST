export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mustc_root=PATH_TO_MUSTC_DATA
lang=de
modelfile=PATH_TO_SAVE_MODEL
wav2vec_path=PATH_TO_PRETRAINED_WAV2VEC

# unidirectional Wav2Vec2.0 and unidirectional encoder
fairseq-train ${mustc_root}/en-${lang} \
  --config-yaml config_raw_joint.yaml \
  --train-subset train_joint \
  --valid-subset dev_st \
  --save-dir ${modelfile} \
  --max-tokens 800000  \
  --update-freq 2 \
  --task speech_to_text_wav2vec \
  --criterion label_smoothed_cross_entropy_with_itst_s2t_flexible_predecision \
  --report-accuracy \
  --arch convtransformer_espnet_base_wav2vec_itst \
  --w2v2-model-path ${wav2vec_path} \
  --uni-encoder True \
  --uni-wav2vec True \
  --optimizer adam \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --clip-norm 10.0 \
  --seed 1 \
  --ddp-backend=no_c10d \
  --keep-best-checkpoints 10 \
  --best-checkpoint-metric accuracy \
  --maximize-best-checkpoint-metric \
  --save-interval-updates 1000 \
  --keep-interval-updates 30 \
  --max-source-positions 800000 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.1 --activation-dropout 0.1 --attention-dropout 0.1 \
  --empty-cache-freq 1000 \
  --ignore-prefix-size 1 \
  --fp16