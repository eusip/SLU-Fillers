import argparse
import glob
import logging
import os
import time
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import glue_output_modes
from transformers import glue_tasks_num_labels
from lightning_base import (
    BaseTransformer,
    add_generic_args,
    generic_train
)
from utils import (
    load_tsv,
    max_length,
    convert_examples_to_features,
    compute_metrics
)
from models import (
    BertLM,
    Mlm,
    Prediction,
    PredictionFT
)

def main():
    # init generic args
    generic_parser = argparse.ArgumentParser(add_help=False)
    add_generic_args(generic_parser, os.getcwd())
    generic_args = generic_parser.parse_known_args()[0]
    
    # confirm data and output folders exist
    ddir = Path(generic_args.data_dir)
    ddir.mkdir(0o755, parents=True, exist_ok=True)

    odir = Path(generic_args.output_dir)
    odir.mkdir(0o755, parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(parents=[generic_parser])

    # init model-specific args
    if generic_args.do_perplexity and generic_args.use_mlm:
        parser = Mlm.add_model_specific_args(parser, os.getcwd())
    elif generic_args.do_perplexity and not generic_args.use_mlm:
        parser = BertLM.add_model_specific_args(parser, os.getcwd())
    elif generic_args.do_predict_confidence and generic_args.use_mlm:
        parser = PredictionFT.add_model_specific_args(parser, os.getcwd())
    elif generic_args.do_predict_confidence and not generic_args.use_mlm:
        parser = Prediction.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    # init model and model path
    if generic_args.do_perplexity and generic_args.use_mlm:
        model = Mlm(args)
        model_path = "perplexity_ft"
    elif generic_args.do_perplexity and not generic_args.use_mlm:
        model = BertLM(args)
        model_path = "perplexity"
    elif generic_args.do_predict_confidence and generic_args.use_mlm:
        model = PredictionFT(args)
        model_path = "prediction_ft"
    elif generic_args.do_predict_confidence and not generic_args.use_mlm:
        model = Prediction(args)
        model_path = "prediction"

    # model training
    if generic_args.do_train or generic_args.fast_dev_run:
        trainer = generic_train(model, args, model_path)
        trainer.fit(model)

    # model inference
    if generic_args.do_test:
        # init test results
        rdir = Path(os.path.join(generic_args.output_dir, "/results"))
        rdir.mkdir(0o755, parents=True, exist_ok=True)

        trainer = generic_train(model, args, model_path)
        # checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpoints", 
        # model_path, "checkpointepoch=*.ckpt"), recursive=True))) 
        # model = model.load_from_checkpoint(checkpoints[-1])
        model = model.load_from_checkpoint(os.path.join(args.output_dir, "checkpoints", model_path, 
        ".ckpt"))
        trainer.test(model)

if __name__ == "__main__":
    main()