export CUDA_VISIBLE_DEVICES=0,1,2,3

mustc_root=PATH_TO_MUSTC_DATA
lang=de
asr_modelfile=PATH_TO_SAVE_ASR_MODEL

python train.py ${mustc_root}/en-${lang} \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${asr_modelfile} --num-workers 4 --max-update 100000 --max-tokens 30000  \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch convtransformer_espnet --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 3 \
  --save-interval-updates 1000 \
  --keep-interval-updates 100 \
  --find-unused-parameters \
  --fp16 \
  --log-interval 10