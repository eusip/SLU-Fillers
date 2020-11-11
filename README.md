<div align="center">    
 
# The importance of fillers for text representations of speech transcripts.

[![Paper](http://img.shields.io/badge/paper-arxiv.2009.11340-B31B1B.svg)](https://www.nature.com/articles/nature14539)
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
First, install dependencies   
```bash
# clone project   
git clone https://github.com/eusip/SLU-Fillers

# install project  
cd SLU-Fillers
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, excute main.py in order to replicate any of 4 experiments.   
 ```bash
# Perplexity experiment (Left-to-right Language Model)
python main.py --filler_case no_filler --do_train --do_perplexity

# Fine-tuned Perplexity experiment (Masked Language Model)
python main.py --filler_case no_filler --do_train --do_perplexity --use_mlm

# Confidence Prediction experiment
python main.py --filler_case no_filler --do_train --do_predict_confidence

# Fine-tuned Confidence Predition experiment
python main.py --filler_case no_filler --do_train --do_predict_confidence --use_mlm
```
In order to make use of the masked language model for running the fine-tuned confidence prediction experiment the fine-tuned perplexity must be run immediately prior so that the masked language model can be available to load prior to training.

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
 
