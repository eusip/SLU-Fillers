import argparse
import glob
import logging
import os
import time
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
# from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import glue_output_modes
from transformers import glue_tasks_num_labels
from lightning_base import (BaseTransformer,
                            add_generic_args,
                            generic_train
)

from utils import (load_tsv,
                    max_length,
                    convert_examples_to_features,
                    compute_metrics
)
from models import (BertLM,
                    Mlm,
                    Prediction,
                    PredictionFT
)

def main():
    # init generic args
    generic_parser = argparse.ArgumentParser(add_help=False)
    add_generic_args(generic_parser, os.getcwd())
    generic_args = generic_parser.parse_known_args()[0]
    
    parser = argparse.ArgumentParser(parents=[generic_parser])

    # init model specific args
    if generic_args.do_perplexity and generic_args.use_mlm:
        parser = Mlm.add_model_specific_args(parser, os.getcwd())
    elif generic_args.do_perplexity and not generic_args.use_mlm:
        parser = BertLM.add_model_specific_args(parser, os.getcwd())
    elif generic_args.do_predict_confidence and generic_args.use_mlm:
        parser = PredictionFT.add_model_specific_args(parser, os.getcwd())
    elif generic_args.do_predict_confidence and not generic_args.use_mlm:
        parser = Prediction.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    # # If output_dir not provided, a folder will be generated in pwd
    # if args.output_dir is None:
    #     args.output_dir = os.path.join(
    #         "./results",
    #         f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
    #     )
    #     os.makedirs(args.output_dir)

    if generic_args.do_perplexity and generic_args.use_mlm:
        model = Mlm(args)
    elif generic_args.do_perplexity and not generic_args.use_mlm:
        model = BertLM(args)
    elif generic_args.do_predict_confidence and generic_args.use_mlm:
        model = PredictionFT(args)
    elif generic_args.do_predict_confidence and not generic_args.use_mlm:
        model = Prediction(args)

    if generic_args.do_train or generic_args.fast_dev_run:
        trainer = generic_train(model, args)
        trainer.fit(model)

    if generic_args.do_test:
        trainer = generic_train(model, args)
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        print('checkpoints', checkpoints)
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)

if __name__ == "__main__":
    main()
