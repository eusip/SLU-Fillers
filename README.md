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

usage: main.py [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--overwrite_cache] [--gpus GPUS] [--max_grad_norm GRADIENT_CLIP_VAL] [--gradient_accumulation_steps ACCUMULATE_GRAD_BATCHES] [--seed SEED] [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--do_train] [--do_test] [--do_perplexity] [--do_predict_confidence] [--use_mlm] [--fast_dev_run]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   The input data dir. Should contain the training files based on the POM (2014) dataset.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written.
  --overwrite_cache     Overwrite the cached training and evaluation sets.
  --gpus GPUS           The number of GPUs allocated for this, it is by default 0 meaning none.
  --max_grad_norm GRADIENT_CLIP_VAL
                        Value used to clip the global gradient norm for the optimizer.
  --gradient_accumulation_steps ACCUMULATE_GRAD_BATCHES
                        Number of updates steps to accumulate before performing a backward/update pass.
  --seed SEED           Random seed for initialization.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Evaluate the model every n training epochs.
  --do_train            Run the training.
  --do_test             Run the testing.
  --do_perplexity       Whether to compute perplexity.
  --do_predict_confidence
                        Whether to compute a confidence prediction.
  --use_mlm             Modelling using MLM model during training. Inference using the MLM model during testing.
  --fast_dev_run        Runs 1 batch of train, test and val to find any bugs.
 ```
 The following commands allow the 4 experiments of the paper to be reproduced.
 ```bash
# Perplexity experiment (Left-to-right Language Model)
python main.py --filler_case none --do_train --do_perplexity

# Fine-tuned Perplexity experiment (Masked Language Model)
python main.py --filler_case none --do_train --do_perplexity --use_mlm

# Confidence Prediction experiment
python main.py --filler_case none --do_train --do_predict_confidence

# Fine-tuned Confidence Predition experiment
python main.py --filler_case none --do_train --do_predict_confidence --use_mlm
```
The default filler_case is `none`, the case of a single filler is `unique`, and the case of distinct fillers for "um" and "uh" is `distinct`.

In order to make use of the masked language model for running the fine-tuned confidence prediction experiment, the fine-tuned perplexity must be run immediately prior so that the masked language model can be available to load prior to training. The masked language model is temporarily saved in the folder `best_tfmr`. If you want to save this model for future use move it to another location. In order to use it again as part of the fine-tuned confidence prediction experiment add the option `--mlm_path` as well the folder path to where the contents of `best_tfmr` are now located.

## View the output
The output directory that is produced by this program has the following structure:

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
+-- runs <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- perplexity <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- version_*n* <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- perplexity_ft <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- version_*n* <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- prediction <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- version_*n* <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- prediction_ft <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- version_*n* <br>
+-- results <br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- test_results.txt

## Run Tensorboard
The metrics which are logged as part of the training loop (perplexity, loss, val_loss) can be viewed using Tensorboard.
```bash
$ cd SLU-fillers/output
$ tensorboard --logdir runs/perplexity_ft
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


### Citation
```
@article{Ebenge Usip,
  title={Research Engineer},
  author={Affective Computing group, Institute Mines-Telecom, Telecom ParisTech},
  journal={Saclay, France},
  year={2020}
}
```
