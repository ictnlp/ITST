# ITST for Text-to-text Simultaneous Translation

This is a tutorial of ITST training and inference on text-to-text simultaneous translation.

## Get Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)) and WMT15 German-English (download [here](https://www.statmt.org/wmt15/)).

For WMT15 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 32K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt). Follow [preprocess scripts](https://github.com/Vily1998/wmt16-scripts) to perform tokenization and BPE.

Then, we process the data into the fairseq format, adding `--joined-dictionary` for WMT15 German-English:

```bash
src_lang=SOURCE_LANGUAGE
tgt_lang=TARGET_LANGUAGE
train_data=PATH_TO_TRAIN_DATA
vaild_data=PATH_TO_VALID_DATA
test_data=PATH_TO_TEST_DATA
data=PATH_TO_DATA

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${src_lang} --target-lang ${tgt_lang} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20
```

### Training

Train the ITST with the following command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL

python train.py --ddp-backend=no_c10d ${data} --arch transformer_itst --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --criterion label_smoothed_cross_entropy_with_itst_t2t \
 --label-smoothing 0.1 \
 --left-pad-source \
 --uni-encoder True \
 --fp16 \
 --save-dir ${modelfile} \
 --max-tokens 8192 --update-freq 1 \
 --save-interval-updates 1000 \
 --keep-interval-updates 200 \
 --log-interval 10
```

### Inference

Evaluate the model with the following command:

- Note that simultaneous machine translation require `--batch-size=1`and `--beam=1` .

```bash
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

```

## Our Results

The numerical results on IWSLT15 English-to-Vietnamese with Transformer-Small:

| delta |  CW  |  AP  |  AL   |  DAL  | BLEU  |
| :---: | :--: | :--: | :---: | :---: | :---: |
|  0.1  | 1.18 | 0.68 | 3.95  | 5.04  | 28.56 |
|  0.2  | 2.08 | 0.72 | 4.55  | 8.59  | 28.68 |
|  0.3  | 4.24 | 0.80 | 6.10  | 13.26 | 28.81 |
|  0.4  | 6.61 | 0.88 | 8.31  | 16.61 | 28.82 |
|  0.5  | 9.01 | 0.92 | 10.75 | 18.73 | 28.89 |

The numerical results on WMT15 German-to-English with Transformer-Base:

| delta |  CW  |  AP  |  AL   |  DAL  | BLEU  | SacreBLEU |
| :---: | :--: | :--: | :---: | :---: | :---: | :-------: |
|  0.2  | 1.43 | 0.59 | 2.27  | 3.87  | 26.44 |   25.17   |
|  0.3  | 1.70 | 0.61 | 2.85  | 4.86  | 28.22 |   26.94   |
|  0.4  | 2.16 | 0.65 | 3.83  | 6.61  | 29.65 |   28.58   |
|  0.5  | 3.18 | 0.71 | 5.47  | 10.16 | 30.63 |   29.51   |
|  0.6  | 4.63 | 0.78 | 7.60  | 14.24 | 31.58 |   30.46   |
|  0.7  | 7.04 | 0.86 | 10.17 | 19.17 | 31.92 |   30.74   |
|  0.8  | 9.78 | 0.91 | 12.72 | 22.52 | 32.00 |   30.84   |

The numerical results on WMT15 German-to-English with Transformer-Big:

| delta |  CW  |  AP  |  AL   |  DAL  | BLEU  | sacreBLEU |
| :---: | :--: | :--: | :---: | :---: | :---: | :-------: |
|  0.2  | 1.33 | 0.58 | 1.89  | 3.62  | 25.90 |   24.73   |
|  0.3  | 1.48 | 0.60 | 2.44  | 4.21  | 27.51 |   26.75   |
|  0.4  | 1.70 | 0.62 | 2.99  | 4.91  | 29.35 |   28.52   |
|  0.5  | 2.04 | 0.66 | 4.09  | 6.42  | 30.83 |   29.99   |
|  0.6  | 2.98 | 0.72 | 6.07  | 9.95  | 31.90 |   31.05   |
|  0.7  | 4.59 | 0.81 | 8.60  | 15.03 | 32.85 |   32.02   |
|  0.8  | 7.23 | 0.89 | 11.37 | 20.05 | 32.90 |   32.09   |

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
