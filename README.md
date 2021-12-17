<div align="center">

# The importance of fillers for text representations of speech transcripts.

[![Paper](http://img.shields.io/badge/paper-arxiv.2009.11340-B31B1B.svg)](https://arxiv.org/abs/2009.11340)
[![Conference](http://img.shields.io/badge/EMNLP-2020-4b44ce.svg)](https://2020.emnlp.org/schedule#s1024)
<!--
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!--
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--
Conference
-->
</div>

## Description
While being an essential component of spoken language, fillers (e.g."um" or "uh") often remain overlooked in Spoken Language Understanding (SLU) tasks. We explore the possibility of representing them with deep contextualised embeddings, showing improvements on modelling spoken language and two downstream tasks - predicting a speaker's stance and expressed confidence.

## How to run
First, install dependencies.
```bash
# clone project
git clone https://github.com/eusip/SLU-Fillers

# create a conda environment from the export file
cd SLU-Fillers
conda create -n myenv --file package-list.txt
 ```
An overview of the primary arguments can be accessed using the `--help` option.
 ```
$ python main.py --help

usage: trainer.py [--output_dir OUTPUT_DIR] [--data_dir DATA_DIR]
                  [--log_dir LOG_DIR] --dataset_name DATASET_NAME
                  [--max_grad_norm GRADIENT_CLIP_VAL] [--do_train DO_TRAIN]
                  [--do_validate DO_VALIDATE]
                  [--gradient_accumulation_steps ACCUMULATE_GRAD_BATCHES]
                  [--seed SEED] [--overwrite_cache] --experiment EXPERIMENT
                  [--gpus GPUS] [--precision PRECISION]
                  [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
                  [--fast_dev_run]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written.
  --log_dir LOG_DIR     The Tensorboard log directory.
  --dataset_name DATASET_NAME
                        The name of the dataset to use.
  --max_grad_norm GRADIENT_CLIP_VAL
                        Max gradient norm.                    
  --do_train DO_TRAIN   Whether to run full training.
                        Whether to run one evaluation epoch over the validation set.
  --do_validate DO_VALIDATE
                        Whether to run one evaluation epoch over the validation set.
  --gradient_accumulation_steps ACCUMULATE_GRAD_BATCHES
                        Number of updates steps to accumulate before performing a backward/update pass.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Evaluate the model every n training epochs.
  --seed SEED           Random seed for initialization.
  --overwrite_cache     Overwrite the cached training and evaluation sets.
  --experiment EXPERIMENT
                        Set the model for this analysis - 'LM', 'MLM', 'ConfPred', 'ConfPredFT'.
  --gpus GPUS
                        The number of GPUs allocated for this, it is by default 0 meaning none.
  --precision PRECISION
                        Whether to use full precision or native amp 16-bit half precision.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Evaluate the model every n training epochs.
  --fast_dev_run        Runs 1 batch of train, test and val to find any bugs.
 ```
 The following commands allow the 4 experiments of the paper to be reproduced.
 ```bash
# Perplexity experiment
python trainer.py --gpus 1 --experiment MLM --dataset_name no_fillers --do_train False --do_validate True

# Fine-tuned Perplexity experiment (Masked Language Model)
python trainer.py --gpus 1 --num_train_epochs 10 --experiment MLM --dataset_name no_fillers

# Confidence Prediction experiment
python trainer.py --gpus 1 --num_train_epochs 10 --experiment ConfPred --dataset_name no_fillers

# Fine-tuned Confidence Predition experiment
python trainer.py --gpus 1 --num_train_epochs 10 --experiment ConfPredFT --dataset_name no_fillers
```
The --dataset_name for the case of no filler is `no_filler`, the case of a single filler is `unique_filler`, and the case of distinct fillers for "um" and "uh" is `distinct_fillers`.

In order to make use of the masked language model for running the fine-tuned confidence prediction experiment, the fine-tuned perplexity must be run immediately prior so that the masked language model can be available to load prior to training. The masked language model is temporarily saved in the folder `best_tfmr`. If you want to save this model for future use move it to another location. In order to use it again as part of the fine-tuned confidence prediction experiment add the option `--mlm_path` as well the folder path to where the contents of `best_tfmr` are now located.

## View the output
The output directory that is produced by this program contains folders named by experiment and dataset name. They contain the text files with the final perplexity and loss values during training and validation.

<!--
. <br>
+-- best_tfmr <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--config.json <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--pytorch_model.bin <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--special_tokens_map.json <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--tokenizer_config.json <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--vocab.txt <br>
+-- checkpoints <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- checkpoint-perplexity.ckpt <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- checkpoint-perplexity_ft.ckpt <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- checkpoint-prediction.ckpt <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- checkpoint-prediction_ft.ckpt <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- 'experiment'_'dataset_name' <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- version_*n* <br>
-->

## Run Tensorboard
The metrics which are logged as part of the training loop (loss, perplexity, MSE) can be viewed using Tensorboard.
```bash
$ cd SLU-fillers/output
$ tensorboard --logdir lightning_logs
```

<!--
## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer
-->
<!--
# model
model = LitClassifier()
-->
<!--
# data
train, val, test = mnist()
-->
<!--
# train
trainer = Trainer()
trainer.fit(model, train, val)
-->
<!--
# test using the best model!
trainer.test(test_dataloaders=test)
```
-->

### Contact
Please contact tanvi dot dinkar at telecom-paris.fr for queries.

### Citation
```
@article{dinkar2020importance,
  title={The importance of fillers for text representations of speech transcripts},
  author={Dinkar, Tanvi and Colombo, Pierre and Labeau, Matthieu and Clavel, Chlo{\'e}},
  journal={arXiv preprint arXiv:2009.11340},
  year={2020}
}
```
