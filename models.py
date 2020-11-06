import argparse
import glob
import logging
import os
from pathlib import Path
import time
from math import exp
from argparse import Namespace

import numpy as np
import torch
from torch.nn import Dropout, Linear, MSELoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoTokenizer,
                        AutoConfig, 
                        AutoModelForSequenceClassification,
                        DataCollatorForLanguageModeling
)
from transformers.modeling_bert import BertPooler, BertLMHeadModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import glue_output_modes
from transformers import glue_tasks_num_labels

from utils import (
    load_tsv,
    max_length,
    load_and_cache_examples_lm,
    convert_examples_to_features,
    compute_metrics,
    BertForMaskedLM
)
from lightning_base import BaseTransformer, add_generic_args, generic_train

logger = logging.getLogger(__name__)

class BertLM(BaseTransformer):
    """An instance of the `BertLMHeadModel` model."""

    mode = "language-modeling"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        config = AutoConfig.from_pretrained(
                    hparams.config_name,
                    cache_dir=hparams.cache_dir if hparams.cache_dir else None,
                    is_decoder=True
        )

        super().__init__(hparams, num_labels, self.mode, config=config)
        self.data_collator = DataCollatorForLanguageModeling(
                                                            tokenizer=self.tokenizer,
                                                            mlm=True,
                                                            mlm_probability=0.15
        )
        self.collator_fn = self.data_collator._tensorize_batch
        
    def forward(self, **inputs):
        return self.model(**inputs)

    @property
    def total_steps(self) -> int:
        return super().total_steps

    def training_step(self, batch, batch_idx):
        labels = batch.clone().detach()
        if self.tokenizer.pad_token_id is not None:
           labels[labels == self.tokenizer.pad_token_id] = -100
        inputs = {"input_ids": batch, "labels": labels}

        outputs = self(**inputs)
        loss = outputs[0]
        
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        # tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        self.log("loss", loss)
        self.log("rate", lr_scheduler.get_last_lr()[-1])
        return {"loss": loss}

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."
        args = self.hparams

        if type_path == "train":
            dataset = load_and_cache_examples_lm(args,
                                                self.tokenizer
            )
        if type_path == "dev":
            dataset = load_and_cache_examples_lm(args,
                                                self.tokenizer,
                                                evaluate=True
            )
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            collate_fn=self.collator_fn,
                                            shuffle=shuffle,
                                            num_workers=args.num_workers
        )

        return loader
        
    def validation_step(self, batch, batch_idx):
        labels = batch.clone().detach()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        inputs = {"input_ids": batch, "labels": labels}

        outputs = self(**inputs)
        loss = outputs[0]
        
        self.log("val_loss", loss)

        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()

        results = {**{"val_loss": val_loss_mean},  **{"perplexity": torch.exp(val_loss_mean.clone().detach()).detach().cpu()}}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("perplexity", logs["perplexity"], on_step=False)
        # return {"avg_val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("perplexity", logs["perplexity"], on_step=False)
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        # return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

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
            "--block_size",
            default=512,
            type=int,
            help="The number of tokens to trained on during a single forward pass",
        ) 
        parser.add_argument(
            "--task",
            default="sts-b",
            type=str,
            help="The GLUE task to run",
        )

        return parser


class Mlm(BaseTransformer):
    """An instance of the `BertForMaskedLM` model."""

    mode = "masked-language-modeling"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]

        config = AutoConfig.from_pretrained(
                    hparams.config_name,
                    cache_dir=hparams.cache_dir if hparams.cache_dir else None,
                    is_decoder=False
        )

        super().__init__(hparams, num_labels, self.mode, config=config)
        self.data_collator = DataCollatorForLanguageModeling(
                                                            tokenizer=self.tokenizer,
                                                            mlm=True,
                                                            mlm_probability=0.15
        )
        self.collator_fn = self.data_collator._tensorize_batch
        self.mask_tokens = self.data_collator.mask_tokens

    def forward(self, **inputs):
        return self.model(**inputs)

    @property
    def total_steps(self) -> int:
        return super().total_steps

    def training_step(self, batch, batch_idx):
        inputs, labels = self.mask_tokens(batch)
        inputs = {"input_ids": inputs, "labels": labels}

        outputs = self(**inputs)
        loss = outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        # tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}  # "perplexity": perplexity,
        self.log("loss", loss)
        self.log("rate", lr_scheduler.get_last_lr()[-1])
        return {"loss": loss}

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."
        args = self.hparams

        if type_path == "train":
            dataset = load_and_cache_examples_lm(args,
                                                self.tokenizer
            )
        if type_path == "dev":
            dataset = load_and_cache_examples_lm(args,
                                                self.tokenizer,
                                                evaluate=True
            )
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            collate_fn=self.collator_fn,
                                            shuffle=shuffle,
                                            num_workers=args.num_workers
        )

        return loader


    def validation_step(self, batch, batch_idx):
        inputs, labels = self.mask_tokens(batch)
        inputs = {"input_ids": inputs, "labels": labels}

        outputs = self(**inputs)
        loss = outputs[0]
        
        self.log("val_loss", loss)

        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()

        results = {**{"val_loss": val_loss_mean},  **{"perplexity": torch.exp(val_loss_mean.clone().detach()).detach().cpu()}}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("perplexity", logs["perplexity"], on_step=False)
        # return {"avg_val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("perplexity", logs["perplexity"], on_step=False)
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        # return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

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
            "--block_size",
            default=512,
            type=int,
            help="The number of tokens to trained on during a single forward pass",
        ) 
        parser.add_argument(
            "--task",
            default="sts-b",
            type=str,
            help="The GLUE task to run",
        )

        return parser


class Prediction(BaseTransformer):
    """An instance of the `BertForSequenceClassification` model."""

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
        # tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        self.log("loss", loss)
        self.log("rate", lr_scheduler.get_last_lr()[-1])
        return {"loss": loss}

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

        return DataLoader(TensorDataset(all_input_ids, 
                                        all_attention_mask, 
                                        all_labels),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=args.num_workers
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        self.log("val_loss", loss)

        return {"val_loss": loss, "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs: dict) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
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

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("mse", logs["mse"], on_step=False)
        # return {"avg_val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("mse", logs["mse"], on_step=False)
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        # return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

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
            "--task",
            default="sts-b",
            type=str,
            help="The GLUE task to run",
        )

        return parser


class PredictionFT(BaseTransformer):
    """ A model consisting of `BertForMaskedLM` and an MLP classifcation layer."""

    mode = "sequence-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]
        
        output_dir = Path(hparams.output_dir)

        config = AutoConfig.from_pretrained(
                hparams.config_name,
                **({"num_labels": num_labels}),
                cache_dir=hparams.cache_dir if hparams.cache_dir else None,
        ) 

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #         output_dir.joinpath("best_tfmr"),
        # )

        # approach 1 - load model from TF checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(output_dir.joinpath("best_tfmr"),
                        config=config,
        )
        
        super().__init__(hparams, num_labels, self.mode, config=config, model=model)
        # approach 2 - load model from explicit save of PyTorch model
        # bert_path = self.output_dir.joinpath("best_tfmr")    
        # self.bert = torch.load(bert_path)

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
        # tensorboard_logs = {"loss": loss, "rate": lr_scheduler.get_last_lr()[-1]}
        self.log("loss", loss)
        self.log("rate", lr_scheduler.get_last_lr()[-1])
        return {"loss": loss}

    def _feature_file(self, type_path):
        return os.path.join(
            self.hparams.data_dir,
            "cached_lm_{}_{}_{}".format(
                type_path,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                self.hparams.filler_case
            ),
        )

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features."
        args = self.hparams
        
        for type_path in ["train", "dev", "test"]:
            data = load_tsv(args.filler_case, args.data_dir, type_path)
            max_len = max_length(data, self.tokenizer)
            cached_features_file = self._feature_file(type_path)
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

        return DataLoader(TensorDataset(all_input_ids, 
                                        all_attention_mask, 
                                        all_labels),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=args.num_workers
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        outputs = self(**inputs)
        loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        self.log("val_loss", loss)

        return {"val_loss": loss, "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs: dict) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
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

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("mse", logs["mse"], on_step=False)
        # return {"avg_val_loss": logs["val_loss"], "progress_bar": logs}

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.log("avg_val_loss", logs["val_loss"], on_step=False)
        self.log("mse", logs["mse"], on_step=False)        
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `mask_lm_loss`
        # return {"avg_test_loss": logs["val_loss"], "progress_bar": logs}


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
            "--task",
            default="sts-b",
            type=str,
            help="The GLUE task to run",
        )

        return parser