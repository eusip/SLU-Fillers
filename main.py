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
from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import glue_output_modes
from transformers import glue_tasks_num_labels

from utils import *

# MODEL_NAME = "bert-base-cased"

# torch.set_grad_enabled(False)

# "sts-b"


logger = logging.getLogger(__name__)


class BertForSequenceClassification(BaseTransformer):

    mode = "sequence-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        loss = outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features."
        args = self.hparams

        for type_path in ["train", "dev", "test"]:
            data = load_tsv(args.filler_case, args.data_dir, type_path)
            max_len = max_length(data, self.tokenizer)
            cached_features_file = self._feature_file(type_path)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                features = convert_examples_to_features(data, self.tokenizer, max_len)
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."

        cached_features_file = self._feature_file(type_path)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_input_ids = torch.stack([f.input_id for f in features]).squeeze()
        all_attention_mask = torch.stack([f.attention_mask for f in features]).squeeze()
        all_labels = torch.stack([f.label_id for f in features]).squeeze()

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_labels),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"avg_val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            "--filler_case",
            default="distinct",
            type=str,
            help="Set the filler case for this analysis - 'distinct', 'unique', 'none'.",
        )
        parser.add_argument(
            "--task",
            default="sts-b",
            type=str,
            help="The GLUE task to run",
        )
        parser.add_argument(
            "--gpus",
            default=0,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )
        parser.add_argument(
            "--auto_select_gpus",
            action="store_true",
            help="Pick available GPUs automatically.",
        )
        parser.add_argument(
            "--benchmark",
            action="store_true",
            help="Enables cudnn.deterministic for the Trainer instance.",
        )
        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument(
            "--check_val_every_n_epoch",
            action='store_true',
            help="Evaluate the model every n training epochs."
        )
        parser.add_argument(
            "--fast_dev_run",
            action='store_true',
            help="Runs 1 batch of train, test and val to find any bugs."
        )

        return parser


def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = BertForSequenceClassification.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    model = BertForSequenceClassification(args)
    # trainer = generic_train(model, args)

    if args.do_train:
        trainer = generic_train(model,
                                args,
                                train_dataloader=model.train_dataloader,
                                val_dataloaders=model.val_dataloader
        )
        trainer.fit(model)

    if args.do_predict_confidence:
        trainer = generic_train(model,
                                args,
                                test_dataloader=model.test_dataloader
        )
        trainer.test(model)

    if args.do_predict_sentiment:
        pass

    if args.fast_dev_run:
        trainer = generic_train(model,
                                args,
                                train_dataloader=model.train_dataloader,
                                val_dataloaders=model.val_dataloader,
                                test_dataloader=model.test_dataloader
        )
        trainer.fit(model)

    # Predict on test set and write to output_dir
    # if args.do_predict_confidence:
    #     checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
    #     model = model.load_from_checkpoint(checkpoints[-1])
    #     return trainer.test(model)

# def main():
#     parser = argparse.ArgumentParser()

#     ## Required parameters
#     parser.add_argument("--data_dir", default='.', type=str, required=False,
#                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
#     parser.add_argument("--model_name", default="bert-base-cased", type=str, required=False,
#                         help="Path to pre-trained model or shortcut name.")
#     parser.add_argument("--task_name", default='foak', type=str, required=False,
#                         help="The name of the task to train selected in the list: " + ", ")
#     parser.add_argument("--output_dir", default=None, type=str, required=True,
#                         help="The output directory where the model predictions and checkpoints will be written.")


#     parser.add_argument("--filler_case", default='distinct', type=str, required=True,
#                         help="Set the filler case for this analysis - 'distinct', 'unique', 'none'.")
#     parser.add_argument("--local_rank", type=int, default=-1,
#                         help="For distributed training: local_rank")

#     args = parser.parse_args()

#     # Setup logging
#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                         datefmt='%m/%d/%Y %H:%M:%S',
#                         level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
#     # logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#     #                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

#     model = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=True, return_dict=True, num_labels=1)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)

#     load_and_cache_examples(args, tokenizer, evaluate=False)


if __name__ == "__main__":
    main()