import logging
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

import torch 
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertLMHeadModel, 
    BertForMaskedLM,
    BertForSequenceClassification,
    AdamW,
)

from torch import nn

from lightning_base import BaseTransformer

logger = logging.getLogger(__name__)


class BertLM(BaseTransformer):
    """An instance of the `BertLMHeadModel` model."""

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        model = BertLMHeadModel(config)

        super().__init__(hparams, config=config, model=model)

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    # @property
    # def total_steps(self) -> int:
    #     return super().total_steps

    def training_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        perplexity = torch.exp(loss.clone().detach()).cpu()

        # lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        self.logger.experiment.add_scalar("Loss/Train",
                                            loss.clone().detach().cpu(),
                                            self.global_step,
        )  
        self.logger.experiment.add_scalar("Perplexity/Train",
                                            perplexity,
                                            self.global_step,
        )
        # self.log("Loss/Train", loss.clone().detach().cpu(), logger=True)
        # self.log("Perplexity/Train", perplexity, logger=True)
        # self.log("Rate", lr_scheduler.get_last_lr()[-1])
        
        return {"loss": loss}

    def training_epoch_end(self,outputs):
        # the mean training loss for all the examples in the batch
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        results = {
            **{"loss": loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(loss_mean.clone().detach()).cpu()}
        }

        # self.logger.experiment.add_scalar("Loss/Train",
        #                                     results["loss"],
        #                                     self.current_epoch,
        # )
        # self.logger.experiment.add_scalar("Perplexity/Train",
        #                                     results["perplexity"],
        #                                     self.current_epoch,
        # )
        # self.log("Loss/Train", results["loss"])
        # self.log("Perplexity/Train", results["perplexity"])

    def validation_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        # perplexity = torch.exp(loss.clone().detach()).cpu()
        
        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        # the mean validation loss for all the examples in the batch.
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        results = {
            **{"val_loss": val_loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(val_loss_mean.clone().detach()).cpu()}
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )
        # self.log("Loss/Validation", logs["val_loss"], logger=True)
        # self.log("Perplexity/Validation", logs["perplexity"], logger=True)

        val_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        val_results = os.path.join(val_folder, "val_results.txt")
        Path(val_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(val_results, "a") as writer:
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`       
        self.logger.experiment.add_scalar("Loss/Test",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Test",
                                            logs["perplexity"],
                                            self.current_epoch,
        )
        # self.log("Loss/Test", logs["val_loss"])
        # self.log("Perplexity/Test", logs["perplexity"])

        test_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        test_results = os.path.join(test_folder, "test_results.txt")
        Path(test_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(test_results, "a") as writer:
                writer.write("***** Seed - {} *****".format(self.hparams.seed))
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--line_by_line', 
            action='store_true', 
            default=True,
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
        )
        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            '--mlm', 
            action='store_true', 
            default=False,
            # help= ,
        )
        parser.add_argument(
            '--mlm_probability', 
            type=float, 
            default=0.15,
            # help= ,
        )
        return parser


class BertMLM(BaseTransformer):
    """An instance of the `BertForMaskedLM` model."""

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        model = BertForMaskedLM(config)

        super().__init__(hparams, config=config, model=model)

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    # @property
    # def total_steps(self) -> int:
    #     return super().total_steps

    def training_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        perplexity = torch.exp(loss.clone().detach()).cpu()

        # lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        self.logger.experiment.add_scalar("Loss/Train",
                                            loss.clone().detach().cpu(),
                                            self.global_step,
        )  
        self.logger.experiment.add_scalar("Perplexity/Train",
                                            perplexity,
                                            self.global_step,
        )
        # self.log("Loss/Train", loss.clone().detach().cpu(), logger=True)
        # self.log("Perplexity/Train", perplexity, logger=True)
        # self.log("Rate", lr_scheduler.get_last_lr()[-1])
        
        return {"loss": loss}

    def training_epoch_end(self,outputs):
        # the mean training loss for all the examples in the batch
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()

        results = {
            **{"loss": loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(loss_mean.clone().detach()).cpu()}
        }

        # self.logger.experiment.add_scalar("Loss/Train",
        #                                     results["loss"],
        #                                     self.current_epoch,
        # )
        # self.logger.experiment.add_scalar("Perplexity/Train",
        #                                     results["perplexity"],
        #                                     self.current_epoch,
        # )
        # self.log("Loss/Train", results["loss"])
        # self.log("Perplexity/Train", results["perplexity"])

    def validation_step(self, batch, batch_idx):
        # forward pass
        outputs = self(**batch)
        loss = outputs[0]
        # perplexity = torch.exp(loss.clone().detach()).cpu()
        
        return {"val_loss": loss}

    def _eval_end(self, outputs: dict) -> tuple:
        # the mean validation loss for all the examples in the batch
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        results = {
            **{"val_loss": val_loss_mean.detach().cpu()},  
            **{"perplexity": torch.exp(val_loss_mean.clone().detach()).cpu()}
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("Loss/Validation",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Validation",
                                            logs["perplexity"],
                                            self.current_epoch,
        )
        # self.log("Loss/Validation", logs["val_loss"], logger=True)
        # self.log("Perplexity/Validation", logs["perplexity"], logger=True)

        val_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        val_results = os.path.join(val_folder, "val_results.txt")
        Path(val_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(val_results, "a") as writer:
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`       
        self.logger.experiment.add_scalar("Loss/Test",
                                            logs["val_loss"],
                                            self.current_epoch,
        )
        self.logger.experiment.add_scalar("Perplexity/Test",
                                            logs["perplexity"],
                                            self.current_epoch,
        )
        # self.log("Loss/Test", logs["val_loss"])
        # self.log("Perplexity/Test", logs["perplexity"])

        test_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        test_results = os.path.join(test_folder, "test_results.txt")
        Path(test_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(test_results, "a") as writer:
                writer.write("***** Seed - {} *****".format(self.hparams.seed))
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--line_by_line', 
            action='store_true', 
            default=True,
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
        )
        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            '--mlm', 
            action='store_true', 
            default=True,
            # help= ,
        )
        parser.add_argument(
            '--mlm_probability', 
            type=float, 
            default=0.15,
            # help= ,
        )
        return parser


class BertSentPrediction(BaseTransformer):
    """An instance of the `BertForSequenceClassification` model with an MLP classifcation layer."""

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path, num_labels=1)
        self.config = config
        model = BertForSequenceClassification(config=config)

        super().__init__(hparams, config=config, model=model)

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    # @property
    # def total_steps(self) -> int:
    #     return super().total_steps

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"]
        }

        # forward pass
        outputs = self(**inputs)
        
        return None

    def training_epoch_end(self,outputs):
        return None

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"]
        }

        # forward pass
        outputs = self(**inputs)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        target = batch["labels"].detach().cpu().numpy()

        return {"pred": preds, "target": target}

    def _eval_end(self, outputs: dict) -> tuple:
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.squeeze(preds)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_ids = out_label_ids[:, 0]
        results = {**compute_metrics(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("MSE/Validation",
                                            logs["mse"],
                                            self.current_epoch,
        )
        # self.log("MSE/Validation", logs["mse"], logger=True)
        
        val_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        val_results = os.path.join(val_folder, "val_results.txt")
        Path(val_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(val_results, "a") as writer:
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.logger.experiment.add_scalar("MSE/Test",
                                            logs["mse"],
                                            self.current_epoch,
        )
        # self.log("MSE/Test", logs["mse"])

        test_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        test_results = os.path.join(test_folder, "test_results.txt")
        Path(test_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(test_results, "a") as writer:
                writer.write("***** Seed - {} *****".format(self.hparams.seed))
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--line_by_line', 
            action='store_true', 
            default=True,
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
        )
        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            '--mlm', 
            action='store_true', 
            default=False,
            # help= ,
        )
        parser.add_argument(
            '--mlm_probability', 
            type=float, 
            default=0.15,
            # help= ,
        )

        return parser


class BertSentPredictionFT(BaseTransformer):
    """An instance of the `BertForSequenceClassification` model with an MLP classifcation layer."""

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        config = AutoConfig.from_pretrained(hparams.model_name_or_path, num_labels=1)
        # load HF `pretrained` save of the MLM
        model = AutoModelForSequenceClassification.from_pretrained(hparams.mlm_path, config=config)

        super().__init__(hparams, config=config, model=model)

        # TODO: finish researching distributed loss accumulation
        # if hparams.gpus > 1:
        #     def training_step_end(self, training_step_outputs):
        #         gpu_0_pred = training_step_outputs[0]['pred']
        #         gpu_1_pred = training_step_outputs[1]['pred']
        #         # gpu_n_pred = training_step_outputs[n]['pred']

        #         # this softmax now uses the full batch
        #         # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        #         # return loss

        #     def test_step_end(self, attention_results):
        #         # this out is now the full size of the batch
        #         all_test_step_outs = attention_results.out
        #         # loss = nce_loss(all_test_step_outs)
        #         # self.log('test_loss', loss)

    def forward(self, **inputs):
        return self.model(**inputs)

    # @property
    # def total_steps(self) -> int:
    #     return super().total_steps

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"]
        }

        # forward pass
        outputs = self(**inputs)
        
        return None

    def training_epoch_end(self,outputs):
        return None

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"]
        }

        # forward pass
        outputs = self(**inputs)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        target = batch["labels"].detach().cpu().numpy()

        return {"pred": preds, "target": target}


    def _eval_end(self, outputs: dict) -> tuple:
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.squeeze(preds)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_ids = out_label_ids[:, 0]
        results = {**compute_metrics(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def validation_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]
       
        self.logger.experiment.add_scalar("MSE/Validation",
                                            logs["mse"],
                                            self.current_epoch,
        )
        # self.log("MSE/Validation", logs["mse"], logger=True)

        val_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        val_results = os.path.join(val_folder, "val_results.txt")
        Path(val_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(val_results, "a") as writer:
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def test_epoch_end(self, outputs: dict) -> dict:
        ret = self._eval_end(outputs)
        logs = ret["log"]

        self.logger.experiment.add_scalar("MSE/Test",
                                            logs["mse"],
                                            self.current_epoch,
        )
        # self.log("MSE/Test", logs["mse"])

        test_folder = os.path.join(self.hparams.output_dir, self.hparams.experiment + "_" + self.hparams.dataset_name)
        test_results = os.path.join(test_folder, "test_results.txt")
        Path(test_folder).mkdir(parents=True, exist_ok=True)

        if self.current_epoch == (self.hparams.max_epochs - 1):
            with open(test_results, "a") as writer:
                writer.write("***** Seed - {} *****".format(self.hparams.seed))
                for key in sorted(logs.keys()):
                    writer.write("%s = %s\n" % (key, str(logs[key]. numpy())))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        parser.add_argument(
            '--line_by_line', 
            action='store_true', 
            default=True,
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
        )
        parser.add_argument(
            '--pad_to_max_length', 
            action='store_true', 
            default=True,
            help="Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        )
        parser.add_argument(
            '--max_seq_length', 
            type=int, 
            default=512,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        )
        parser.add_argument(
            '--mlm', 
            action='store_true', 
            default=False,
            # help= ,
        )
        parser.add_argument(
            '--mlm_probability', 
            type=float, 
            default=0.15,
            # help= ,
        )
        parser.add_argument(
            "--mlm_path",
            default="./output/best_tfmr",
            type=str,
            help="Path to the Hf `pretrained` save of the MLM",
        )

        return parser


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    mse = mean_squared_error(labels, preds)
    return {"mse": mse}