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

from utils import load_tsv, max_length, convert_examples_to_features, compute_metrics
from lightning_base import BaseTransformer, add_generic_args, generic_train

logger = logging.getLogger(__name__)


class Prediction(BaseTransformer):

    mode = "sequence-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    @property
    def total_steps(self) -> int:
        return super().total_steps

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
        args = self.hparams
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
            num_workers=args.num_workers
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs: dict) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
        return {"avg_val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            "--filler_case",
            default="no_filler",
            type=str,
            help="Set the filler case for this analysis - 'distinct', 'unique', 'none'.",
        )
        parser.add_argument(
            "--use_mlm",
            action="store_true",
            help="The trained MLM model ('fine-tuned') is loaded instead of the trained pretrained BERT model ('not fine-tuned')"
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
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument(
            "--check_val_every_n_epoch",
            default=1,
            type=int,
            help="Evaluate the model every n training epochs."
        )
        parser.add_argument(
            "--fast_dev_run",
            action='store_true',
            help="Runs 1 batch of train, test and val to find any bugs."
        )

        return parser


class PredictionFT(BaseTransformer):  # BertForMaskedLM + MLP

    mode = "language-modeling"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        super().__init__(hparams, num_labels, self.mode)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # see line 179

    def forward(self, **inputs):  # labels need to be passed through forward()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(**inputs, return_dict=return_dict)

        pooled_output = outputs[2]  # hidden_states ?

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        #  We are doing regression
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def total_steps(self) -> int:
        return super().total_steps

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
        args = self.hparams
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
            num_workers=args.num_workers
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs: dict) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
        return {"avg_val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            "--filler_case",
            default="no_filler",
            type=str,
            help="Set the filler case for this analysis - 'distinct', 'unique', 'none'.",
        )
        parser.add_argument(
            "--use_mlm",
            action="store_true",
            help="The trained MLM model ('fine-tuned') is loaded instead of the trained pretrained BERT model ('not fine-tuned')"
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
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument(
            "--check_val_every_n_epoch",
            default=1,
            type=int,
            help="Evaluate the model every n training epochs."
        )
        parser.add_argument(
            "--fast_dev_run",
            action='store_true',
            help="Runs 1 batch of train, test and val to find any bugs."
        )

        return parser