# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-do-train">How To Do Train</a> •
  <a href="#how-to-evaluate">How To Evaluate</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n project_env python=3.10

   # activate env
   conda activate project_env

   # support youtokentome ://
   conda install -c conda-forge gcc=12.1.0
   pip install Cython
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```
If you have issues with installing `youtokentome` library please write me or `youtokentome` authors.

## How To Do Train
You should have single A100-80gb GPU to exactly reproduce training, otherwise please implement and use gradient accumulation

To train a model, run the following commands and register in WandB:

Two-steps training:
```bash
python3 train.py -cn=deepspeech_config.yaml writer.run_name=full_deepspeech_beam_search_torch_zero_to_hero 
```
Stop this training at 23 epoch (due to coverage) and resume it with second config
```bash
python3 train.py -cn=deepspeech_config_part2.yaml writer.run_name=full_deepspeech_beam_search_torch_zero_to_hero_part_2_more_augs_wo_limits trainer.resume_from=<ABSOLUTE_PATH_TO_DIRECTORY>/saved/full_deepspeech_beam_search_torch_zero_to_hero/model_best.pth
```
Stop second training at 45 epoches (in total with first step). Two-steps will take about 24 hours to train

> Pay attention that in all configs base model is DeepSpeech2-repack-by-teasgen, but you may use Conformer additionally (take a look at conformer_config.yaml)

Also you may use BPE tokenizer instead of dummy chars, to do it firstly train BPE on full LibriSpeech dataset:
```bash
python3 src/utils/train_bpe.py --ls-indices-dir <ABSOLUTE_PATH_TO_DIRECTORY>/data/datasets/librispeech --dir-to-save-model <ABSOLUTE_PATH_TO_DIRECTORY>/data/bpe
```
And then run training as same as char tokenizer. For more hydra details take a look at deepspeech_config_bpe.yaml. Unfortunately in this mode you mustn't use beam search with LM. You may download my trained BPE from `https://disk.yandex.ru/d/6KNUINjFGn9ofQ`

Moreover, the training report and other logs are available in WandB https://api.wandb.ai/links/teasgen/dx35cnsu

## How To Evaluate
All generated texts will be saved into `data/saved/evals` directory with corresponing names. Download pretrained model from `https://disk.yandex.ru/d/6KNUINjFGn9ofQ` and put it in `saved/full_deepspeech_beam_search_torch_zero_to_hero_part_2_more_augs_wo_limits/model_best.pth`

0. To run inference provide custom dataset (possibly without transcriptions) as same as `data/test_data` format and run
   Optionally you can use LibriSpeech dataset format (set datasets=libri_speech_eval in hydra_config)

   ```bash
   python3 inference.py -cn=inference.yaml dataloader.batch_size=5
   ```
   Set dataloader.batch_size not more than len(dataset)

   Optionally set `text_encoder.decoder_type` to preferred evalution algorithm. Possible values are:
   - argmax
   - beam_search (my slow implementation)
   - beam_search_torch (fast batched algorithm)
   - beam_search_lm (slow single-sample beam search with open source kenlm)

   To speed up or raise score you may change `beam_size` value in hydra config. For getting reported values I used beam_size=50
   When you get transcriptions, run following command to calculate WER and CER metrics
   ```bash
   export PYTHONPATH=./ && python3 src/utils/calculate_cer_wer.py --predicts-dir data/saved/evals/test --gt-dir data/test_data/transcriptions
   ```

1. If you want to calculate metrics on dataset provide a it as same as `data/test_data` format and run
   
   ```bash
   python3 inference.py -cn=inference_and_metrics.yaml dataloader.batch_size=5
   ```

   Optionally you can use LibriSpeech dataset format (set datasets=libri_speech_eval in hydra_config)

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
