# ITST for Speech-to-text Simultaneous Translation with Flexible Pre-decision

This is a tutorial of ITST training and inference on speech-to-text simultaneous translation with *flexible pre-decision*.

- For flexible decision, ITST takes raw speech as input and decides READ/WRITE on each speech frame. 

- To utilize the pre-trained acoustic model in smultaneous translation, ITST finetunes the [Wav2Vec2.0](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf) into the unidirectional type for simultaneous decoding. 
- ITST is trained by multi-task learning of ASR and ST tasks.

## Get Start

### Data Pre-processing

We use the data of MuST-C (download [here](https://nlp.stanford.edu/projects/nmt/)), which is multilingual speech-to-text translation corpus with 8-language translations on English TED talks. Download `MUSTC_v1.0_en-${lang}.tar.gz` to the path `${mustc_root}`, and then preprocess it with:

```bash
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
```

Finally, the directory `${mustc_root}` should look like:

```
.
├── en-de/
│   ├── config_raw_joint.yaml
│   ├── spm_unigram10000_raw_joint.model
│   ├── spm_unigram10000_raw_joint.txt
│   ├── spm_unigram10000_raw_joint.vocab
│   ├── dev_raw_st.tsv
│   ├── tst-COMMON_raw_st.tsv
│   ├── train_raw_joint.tsv
│   ├── tst-COMMON_raw_joint.tsv
│   ├── tst-HE_raw_joint.tsv
│   ├── docs/
│   ├── data/
├── en-de-eval/
│   ├── tst-COMMON/
│   │   ├── tst-COMMON.de
│   │   ├── tst-COMMON.wav_list
│   │   ├── ted_****_**.wav
│   │   ├── ...
│   ├── dev/
│   │   ├── dev.de
│   │   ├── dev.wav_list
│   │   ├── ted_****_**.wav
│   │   ├── ...
│   ├── tst-HE/
│   │   ├── tst-HE.de
│   │   ├── tst-HE.wav_list
│   │   ├── ted_****_**.wav
│   │   ├── ...
└── MUSTC_v1.0_en-de.tar.gz
```

The config file `config_raw_joint.yaml` should be like this.

```yaml
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: ABS_PATH_TO_SENTENCEPIECE_MODEL
input_channels: 1
prepend_tgt_lang_tag: true
use_audio_input: true
vocab_filename: spm_unigram10000_raw_joint.txt
```

We train speech-to-text simultaneous translation with flexible pre-decision with multi-task learning, including ASR and ST task. `train_raw_joint.tsv` should be like:

```
id      audio   n_frames        src_text        tgt_text        speaker src_lang        tgt_lang
ted_1_0 ${mustc_root}/en-de/data/train/wav/ted_1.wav:98720:460800       460800  And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful. I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night.    Vielen Dank, Chris. Es ist mir wirklich eine Ehre, zweimal auf dieser Bühne stehen zu dürfen. Tausend Dank dafür. Ich bin wirklich begeistert von dieser Konferenz, und ich danke Ihnen allen für die vielen netten Kommentare zu meiner Rede vorgestern Abend.    spk.1   en      de
ted_1_0 ${mustc_root}/en-de/data/train/wav/ted_1.wav:98720:460800       460800  And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful. I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night.    And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful. I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night.    spk.1   en      en
ted_1_1 /${mustc_root}/en-de/data/train/wav/ted_1.wav:560160:219040      219040  And I say that sincerely, partly because (Mock sob) I need that. (Laughter)     Das meine ich ernst, teilweise deshalb — weil ich es wirklich brauchen kann! (Lachen) Versetzen Sie sich mal in meine Lage! (Lachen) (Applaus) Ich bin bin acht Jahre lang mit der Air Force Two geflogen. spk.1   en      de
ted_1_1 ${mustc_root}/en-de/data/train/wav/ted_1.wav:560160:219040      219040  And I say that sincerely, partly because (Mock sob) I need that. (Laughter)     And I say that sincerely, partly because (Mock sob) I need that. (Laughter)        spk.1   en      en
......
```

### Training

ITST uses a pre-trained Wav2Vec2.0 (download [here](dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)) to capture the speech features. Download `wav2vec_small.pt` to  `wav2vec_path`. To enable simultaneous decoding, we turn Wav2Vec2.0 into unidirectional type with `--uni-encoder` and `--uni-wav2vec`. 

- ***Unidirectional Wav2Vec2.0***: Turning the Transformer blocks in Wav2Vec2.0 into unidirectional (add the causal mask), and freeze the parameters of convolutional layers.

Train ITST with flexible pre-decision:

```bash
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
```

### Inference

We use [SimulEval](https://github.com/facebookresearch/SimulEval) for evaluation, install it with:

```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .
```

Evaluate ITST with the following command.
- Make sure line 273 in [SimulEval/simuleval/scorer/instance.py](https://github.com/facebookresearch/SimulEval/blob/main/simuleval/scorer/instance.py) is: `samples, _ = soundfile.read(wav_path, dtype="float32")`

```bash
export CUDA_VISIBLE_DEVICES=0

mustc_root=PATH_TO_MUSTC_DATA
lang=de
modelfile=PATH_TO_SAVE_MODEL

wav_list=FILE_OF_SRC_AUDIO_LIST
reference=FILE_OF_TGT_REFERENCE
output_dir=DIR_TO_OUTPUT

# test threshold in ITST, such as 0.8
threshold=TEST_THRESHOLD

# average best 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
    --output ${modelfile}/average-model.pt --best True
file=${modelfile}/average-model.pt

simuleval --agent examples/speech_to_text/simultaneous_translation/agents/simul_agent.s2t.itst.flexible_predecision.py \
    --source ${wav_list} \
    --target ${reference} \
    --data-bin ${MUSTC_ROOT}/en-${LANG} \
    --config config_raw_joint.yaml \
    --model-path ${file} \
    --test-threshold ${threshold} \
    --lang ${LANG} \
    --output ${output_dir} \
    --scores --gpu \
    --port 1234
```

## ITST Results

The numerical results on MuST-C English-to-German:

| delta |   CW    |  AP  |   AL    |   DAL   | SacreBLEU |
| :---: | :-----: | :--: | :-----: | :-----: | :-------: |
| 0.75  | 558.30  | 0.73 | 1448.53 | 1720.45 |   17.90   |
| 0.80  | 684.79  | 0.75 | 1588.52 | 2047.05 |   18.47   |
| 0.81  | 773.64  | 0.77 | 1677.98 | 2251.77 |   19.09   |
| 0.82  | 877.89  | 0.79 | 1778.44 | 2499.23 |   19.50   |
| 0.83  | 1042.91 | 0.81 | 1918.86 | 2819.91 |   20.09   |
| 0.84  | 1275.87 | 0.83 | 2136.53 | 3213.04 |   20.64   |
| 0.85  | 1539.91 | 0.86 | 2370.87 | 3594.93 |   21.06   |
| 0.86  | 1842.74 | 0.88 | 2617.66 | 3944.18 |   21.64   |
| 0.87  | 2171.43 | 0.90 | 2892.93 | 4258.03 |   21.80   |
| 0.88  | 2559.36 | 0.92 | 3192.52 | 4544.17 |   22.02   |
| 0.89  | 2971.17 | 0.94 | 3501.27 | 4786.36 |   22.27   |
| 0.90  | 3430.62 | 0.95 | 3875.92 | 5006.06 |   22.51   |
| 0.92  | 4296.22 | 0.98 | 4556.58 | 5317.75 |   22.62   |
| 0.95  | 5114.86 | 0.99 | 5206.45 | 5543.74 |   22.71   |

Refer to paper for more numerical results.

## Citation

If you have any questions, feel free to contact me with: `zhangshaolei20z@ict.ac.cn`.

In this repository is useful for you, please cite as:

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
}
```
