import argparse
import glob
import logging
import os
import re
from pathlib import Path
import warnings

import torch 

from pytorch_lightning.loggers import TensorBoardLogger

from lightning_base import (
    BaseTransformer, 
    add_generic_args, 
    add_trainer_args, 
    generic_train,
    )
from models import (
    BertLM, 
    BertMLM, 
    BertSentPrediction, 
    BertSentPredictionFT,
)
from data import NoFillers, UniqueFiller, DistinctFillers

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

logger = logging.getLogger(__name__)

DATASETS = {
    "no_fillers": NoFillers,
    "unique_filler": UniqueFiller,
    "distinct_fillers": DistinctFillers,
}

MODELS = {
    "LM": BertLM,
    "MLM": BertMLM,
    "SentPred": BertSentPrediction,
    "SentPredFT": BertSentPredictionFT,
}

def main():
    # instantiate generic-trainer parser
    generic_trainer_parser = argparse.ArgumentParser(add_help=False)
    add_generic_args(generic_trainer_parser, os.getcwd())
    add_trainer_args(generic_trainer_parser, os.getcwd())

    # parse generic and trainer args
    generic_trainer_args = generic_trainer_parser.parse_known_args()[0]
    
    # general parser 
    parser = argparse.ArgumentParser(parents=[generic_trainer_parser])

    # determine model specific args for experiment
    MODELS[generic_trainer_args.experiment].add_model_specific_args(parser, os.getcwd())
    
    # parse args
    args = parser.parse_args()

    # instantiate model for experiment
    model = MODELS[args.experiment](args)

    # instantiate dataset for experiment
    data = DATASETS[model.hparams.dataset_name](model.hparams)

    # instantiate Tensorboard
    tb_logs = os.path.join(os.getcwd(), "lightning_logs")
    rdir = Path(tb_logs)
    # rdir.mkdirs(parents=True, exist_ok=True)
    log_name = model.hparams.experiment + "_" + model.hparams.dataset_name
    tb_logger = TensorBoardLogger(save_dir=rdir, name=log_name)

    # instantiate trainer
    trainer = generic_train(model, args, logger=tb_logger)  # logger=True
    trainer.fit(model, data)
    if args.do_predict:
        trainer.test(model, data)


if __name__ == "__main__":
    main()