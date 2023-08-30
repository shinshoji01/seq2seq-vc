# Parallel one-to-one non-autoregreseive (non-AR) seq2seq VC recipe using the CMU ARCTIC dataset

## AAS-VC: automatic alignment search based model

2023.8.14

### MUST READ notes before training

The default config is `conf/conformer_fastspeech_flowdp_forwardsum.v1_melmelmel_durloss0_er1_per4.yaml`. What's worthwhile noting is the `per4` part. This refers to the following hyperparameter:

```
model_params:
    ...
    post_encoder_reduction_factor: 4
    ...
```

Ignore the "post", this hyperparameter controls the "compresses rate" of the encoder output feature sequence before we send it into the attention learning module. It is ESSENTIAL to properly set this number: the automatic alignment search method requires that in EVERY parallel pair, the length of the source utterance needs to be reduced such that it is shorter than that of the target utterance. It is recommended to calculate the duration ratio of EVERY training pair prior to training, and properly set this number. (Note that if this number is set to be too large, attention learning will fail.)

If OOM happens (which is quite possible when using PPG or simply caused by a lengthy dataset), change `gradient_accumulate_steps: 1` to some number larger than 1. Note that the effective batch size will be `batch_size * gradient_accumulate_steps`.

### Training

First, extract the features, calculate the statistics and normalize all the features. This is basically the same as FS2-VC.

```
./run.sh --stage 0 --stop_stage 2 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ
```

It is always recommended to explicitly set `--src_feat`, `--trg_feat` and `--dp_feat`.

Then, train the model.

```
./run.sh --stage 3 --stop_stage 3 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ \
  --tag <tag_name>
```

Training results will be saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>`.

### Decoding and evaluation

Then, decoding (conversion) and evaluation can be done by executing the following (this part is also the same as FS2-VC):

```
./run.sh --stage 4 --stop_stage 5 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ \
  --tag <tag_name>
```


## FS2-VC: modified FastSpeech based model

2023.7.12

This recipe attepts to implement the paper: Non-autoregressive sequence-to-sequence voice conversion. https://arxiv.org/abs/2104.06793.

### Difference from the original paper

- I did not include the pitch and energy predictors, so this is essentially just FastSpeech 1.
- The input of the duration predictor is not the encoder outputs (left of figure). Rather, I use a separate, small CNN to process the feature for the duration prediction (right of figure). This unlocks the possibility to use different features.
![](https://file.notion.so/f/s/35fa5499-aaa2-41e1-a627-20228ba0832d/%E6%88%AA%E5%9C%96_2023-05-25_%E4%B8%8B%E5%8D%883.11.01.png?id=088dc130-faca-462e-b993-58029e49f455&table=block&spaceId=77565380-b940-4852-95c0-10905d8aaf4a&expirationTimestamp=1689264000000&signature=3w2LyokT4BTXxhw4BXwYbXzgNjB61Pkh3SlgUB07L2w&downloadName=%E6%88%AA%E5%9C%96+2023-05-25+%E4%B8%8B%E5%8D%883.11.01.png)

### Prepare the target durations for training

First please train an AR seq2seq VC model in `arctic/vc1`. I recommend using a TTS pre-trained AR seq2seq VC model, as it will greatly improve the intelligibility (in terms of DDUR, CER and WER). Note that the input and output features, as long as the decoder reduction factor need to be the same. For example, if in the AR model, the input is PPG, output is mel, and the decoder reduction factor is 4, then the non-AR model has to have the same setting.

After the training of the AR model is done, generate the durations using teacher forcing. Example:

```
./run.sh --stage 6 --stop_stage 6 --norm_name ljspeech --conf conf/vtn.tts_pt.v1.yaml --tag tts_pt_r4 --pretrained_model_checkpoint /data/group1/z44476r/Experiments/seq2seq-vc/egs/ljspeech/tts1/exp/tts_aept_phn_tacotron_r4_checkpoint-100000steps/checkpoint-50000steps.pkl
```

The generated durations will be in `exp/clb_slt_932_<tag>/results/checkpoint-50000steps/clb_train/out.1/durations/*.txt`. Example duration file:

```
1 0 1 1 1 1 2 1 1 1 1 2 1 1 1 1 0 1 1 1 1 0 1 1 2 0 1 1 1 1 1 0 1 1 0 0 1 1 1 2 1 0 2 0 1 1 1 0 1 2
```


### Training the non-AR model

Currently, following the original paper, no pre-training is used.

The file name of the config files usually has three parts, referring to the features of the source, target and duration preeictor input. For example, `conf/conformer_fastspeech.v1_melmelppg_r4teacher.yaml` means the input and output features are mel, and PPG is the input to the duration predictor.

First, extract the features, calculate the statistics and normalize all the features.

```
./run.sh --stage 0 --stop_stage 2 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ
```

It is always recommended to explicitly set `--src_feat`, `--trg_feat` and `--dp_feat`.

Then, train the model.

```
./run.sh --stage 3 --stop_stage 3 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ \
  --tag <tag_name> \
  --train_duration_dir <train_duration_dir> \
  --dev_duration_dir <dev_duration_dir>
```

Please see the `run.sh` file for examples of the `<train_duration_dir>` and `<dev_duration_dir>`.

Training results will be saved in `exp/<srcspk>_<trgspk>_<num_train_utterances>_<tag>`.

### Decoding and evaluation

Then, decoding (conversion) and evaluation can be done by executing the following:

```
./run.sh --stage 4 --stop_stage 5 \
  --norm_name self --conf <conf_file> \
  --src_feat XXX --trg_feat YYY --dp_feat ZZZ \
  --tag <tag_name> \
  --train_duration_dir <train_duration_dir> \
  --dev_duration_dir <dev_duration_dir>
```