# ITST for Speech-to-text Simultaneous Translation with Fixed Pre-decision

This is a tutorial of ITST training and inference on speech-to-text simultaneous translation with *fixed pre-decision*.

- For fixed pre-deciison, ITST divides streaming speech input into multiple fixed-length segments (such as 280 ms), and then decides READ/WRITE on these segments. 

## Get Start

### Data Pre-processing

We use the data of MuST-C (download [here](https://nlp.stanford.edu/projects/nmt/)), which is multilingual speech-to-text translation corpus with 8-language translations on English TED talks. Download `MUSTC_v1.0_en-${lang}.tar.gz` to the path `${mustc_root}`， and then preprocess it with:

```bash
mustc_root=PATH_TO_MUSTC_DATA
lang=de

# unzip
tar -xzvf ${mustc_root}/MUSTC_v1.0_en-${lang}.tar.gz

# prepare ASR data
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${mustc_root} --task asr \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global

# prepare ST data
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${mustc_root} --task st \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global

# generate the wav list and reference file for SimulEval
eval_data=PATH_TO_SAVE_EVAL_DATA # such as ${mustc_root}/en-de-eval
for split in dev tst-COMMON tst-HE
do
    python examples/speech_to_text/seg_mustc_data.py \
    --data-root ${mustc_root} --lang ${lang} \
    --split ${split} --task st \
    --output ${eval_data}/${split}
done
```

Finally, the directory `${mustc_root}` should look like:

```
.
├── en-de/
│   ├── config_st.yaml
│   ├── config_asr.yaml
│   ├── spm_unigram10000_st.model
│   ├── spm_unigram10000_st.txt
│   ├── spm_unigram10000_st.vocab
│   ├── train_st.tsv
│   ├── dev_st.tsv
│   ├── tst-COMMON_st.tsv
│   ├── tst-HE_st.tsv
│   ├── spm_unigram10000_asr.model
│   ├── spm_unigram10000_asr.txt
│   ├── spm_unigram10000_asr.vocab
│   ├── train_asr.tsv
│   ├── dev_asr.tsv
│   ├── tst-COMMON_asr.tsv
│   ├── tst-HE_asr.tsv
│   ├── fbank80.zip
│   ├── gcmvn.npz
│   ├── docs/
│   ├── data/
├── en-de-eval/
│   ├── dev/
│   │   ├── dev.de
│   │   ├── dev.wav_list
│   │   ├── ted_****_**.wav
│   │   ├── ...
│   ├── tst-COMMON/
│   │   ├── tst-COMMON.de
│   │   ├── tst-COMMON.wav_list
│   │   ├── ted_****_**.wav
│   │   ├── ...
│   ├── tst-HE/
│   │   ├── tst-HE.de
│   │   ├── tst-HE.wav_list
│   │   ├── ted_****_**.wav
│   │   ├── ...
└── MUSTC_v1.0_en-de.tar.gz
```

The config file `config_st.yaml` should be like this. For `config_asr.yaml`, change all 'st' into 'asr'.

```yaml
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: ABS_PATH_TO_SENTENCEPIECE_MODEL
global_cmvn:
  stats_npz_path: ABS_PATH_TO_GCMVN_FILE
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - global_cmvn
  _train:
  - global_cmvn
  - specaugment
vocab_filename: spm_unigram10000_st.txt
```

### Training

#### 1. ASR Pretraining

First, we pretrain an asr model:

```bash
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
```

Evaluate the ASR task to find the best checkpoint:

```bash
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
```

#### 2. ST Training

Train ITST with fixed pre-decision (280ms):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

mustc_root=PATH_TO_MUSTC_DATA
lang=de
asr_modelfile=PATH_TO_SAVE_ASR_MODEL
st_modelfile=PATH_TO_SAVE_ST_MODEL

python train.py ${mustc_root}/en-${lang} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --user-dir examples/simultaneous_translation \
    --save-dir ${st_modelfile} --num-workers 8  \
    --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
    --criterion label_smoothed_cross_entropy_with_itst_s2t_fixed_predecision \
    --warmup-updates 4000 --max-update 300000 --max-tokens 20000 --seed 2 \
    --label-smoothing 0.1 \
    --load-pretrained-encoder-from ${asr_modelfile}/best.pt \
    --task speech_to_text  \
    --arch convtransformer_simul_trans_itst_espnet  \
    --unidirectional-encoder \
    --simul-type ITST_fixed_pre_decision  \
    --fixed-pre-decision-ratio 7 \
    --update-freq 4 \
    --save-interval-updates 1000 \
    --keep-interval-updates 200 \
    --find-unused-parameters \
    --fp16 \
    --log-interval 10 
```

### Inference

We use [SimulEval](https://github.com/facebookresearch/SimulEval) for evaluation, install it with:

```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .
```

Evaluate ITST with the following command:

```bash
export CUDA_VISIBLE_DEVICES=0

mustc_root=PATH_TO_MUSTC_DATA
lang=de
modelfile=PATH_TO_SAVE_MODEL
last_file=LAST_CHECKPOINT

wav_list=FILE_OF_SRC_AUDIO_LIST # such as tst-COMMONN.wav_list
reference=FILE_OF_TGT_REFERENCE # such as tst-COMMONN.de
output_dir=DIR_TO_OUTPUT

# test threshold in ITST, such as 0.8
threshold=TEST_THRESHOLD

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
    --output ${modelfile}/average-model.pt --last_file ${last_file}
file=${modelfile}/average-model.pt 

simuleval --agent examples/speech_to_text/simultaneous_translation/agents/simul_agent.s2t.moe_waitk.fixed_predecision.py \
    --source ${wav_list} \
    --target ${reference} \
    --data-bin ${MUSTC_ROOT}/en-${LANG} \
    --config config_st.yaml \
    --model-path ${file} \
    --test-threshold ${threshold} \
    --output ${output_dir} \
    --scores --gpu \
    --port 1234
```

## Implementation of Baseline Policy

We also provide the code of baseline policies we implemented. Scripts of training and inference please refer to:

- **Wait-k**: [STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://aclanthology.org/P19-1289)
  - Training: [shell_scripts/run.s2t.waitk.fixed_predecision.sh](shell_scripts/run.s2t.waitk.fixed_predecision.sh)
  - Inference: [shell_scripts/pred.s2t.waitk.fixed_predecision.sh](shell_scripts/pred.s2t.waitk.fixed_predecision.sh)
- **MMA**: [Monotonic Multihead Attention](https://openreview.net/pdf?id=Hyg96gBKPS)
  - Training: [shell_scripts/run.s2t.mma.fixed_predecision.sh](shell_scripts/run.s2t.mma.fixed_predecision.sh)
  - Inference: [shell_scripts/pred.s2t.mma.fixed_predecision.sh](shell_scripts/pred.s2t.mma.fixed_predecision.sh)
- **Multipath Wait-k**: [Efficient Wait-k Models for Simultaneous Machine Translation](http://www.interspeech2020.org/index.php?m=content&c=index&a=show&catid=282&id=611)
  - Training: [shell_scripts/run.s2t.multipath_waitk.fixed_predecision.sh](shell_scripts/run.s2t.multipath_waitk.fixed_predecision.sh)
  - Inference: [shell_scripts/pred.s2t.multipath_waitk.fixed_predecision.sh](shell_scripts/pred.s2t.multipath_waitk.fixed_predecision.sh)
- **MoE Wait-k**: [Universal Simultaneous Machine Translation with Mixture-of-Experts Wait-k Policy](https://aclanthology.org/2021.emnlp-main.581.pdf)
  - Training:  [shell_scripts/run.s2t.moe_waitk.fixed_predecision.sh](shell_scripts/run.s2t.moe_waitk.fixed_predecision.sh)
  - Inference:  [shell_scripts/pred.s2t.moe_waitk.fixed_predecision.sh](shell_scripts/pred.s2t.moe_waitk.fixed_predecision.sh)

## ITST Results

The numerical results on MuST-C English-to-German:

| delta |   CW    |  AP  |   AL    |   DAL   | SacreBLEU |
| :---: | :-----: | :--: | :-----: | :-----: | :-------: |
|  0.2  | 452.35  | 0.71 | 1083.33 | 1476.65 |   14.40   |
|  0.3  | 464.82  | 0.74 | 1207.42 | 1593.02 |   14.81   |
|  0.4  | 500.59  | 0.77 | 1386.12 | 1761.58 |   15.15   |
|  0.5  | 795.99  | 0.80 | 1595.69 | 1964.17 |   15.41   |
|  0.6  | 658.69  | 0.83 | 1911.04 | 2265.77 |   15.68   |
|  0.7  | 929.34  | 0.88 | 2430.46 | 2827.25 |   16.12   |
| 0.75  | 1216.87 | 0.91 | 2797.97 | 3323.25 |   16.17   |
|  0.8  | 1644.32 | 0.94 | 3277.64 | 3999.19 |   16.10   |
| 0.85  | 2394.93 | 0.96 | 3877.90 | 4636.44 |   16.08   |
|  0.9  | 3338.74 | 0.98 | 4494.97 | 5121.08 |   16.17   |

The numerical results on MuST-C English-to-Spanish:

| delta |   CW    |  AP  |   AL    |   DAL   | SacreBLEU  |
| :---: | :-----: | :--: | :-----: | :-----: | :----------: |
|  0.2  | 455.84  | 0.69 | 960.49  | 1452.41 | 17.77 |
|  0.3  | 476.70  | 0.74 | 1152.53 | 1653.25 | 18.38 |
|  0.4  | 510.98  | 0.77 | 1351.47 | 1843.40 | 18.71 |
|  0.5  | 585.07  | 0.81 | 1620.54 | 2112.38 | 19.11 |
|  0.6  | 708.80  | 0.84 | 1964.43 | 2431.00 | 19.77 |
|  0.7  | 889.71  | 0.88 | 2380.75 | 2824.10 | 20.13 |
| 0.75  | 1020.53 | 0.89 | 2642.81 | 3073.15 | 20.46 |
|  0.8  | 1227.30 | 0.91 | 2979.87 | 3453.46 | 20.75 |
| 0.85  | 1583.26 | 0.94 | 3433.96 | 4002.54 | 20.48 |
|  0.9  | 2124.43 | 0.96 | 3982.66 | 4662.57 | 20.64 |

Refer to paper for more numerical results.

## Citation

If you have any questions, feel free to contact me with: `zhangshaolei20z@ict.ac.cn`.

If this repository is useful for you, please cite as:

```
@inproceedings{ITST,
    title = "Information-Transport-based Policy for Simultaneous Translation",
    author = "Zhang, Shaolei  and
      Feng, Yang",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Online and Abu Dhabi",
    publisher = "Association for Computational Linguistics",
    url="https://arxiv.org/pdf/2210.12357.pdf",
}
```
