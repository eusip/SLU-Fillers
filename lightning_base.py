import argparse
import logging
import os
import datetime
import socket
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)


# logger = logging.getLogger(__name__)


# this dict provides the various HF transformer models that can be reference in this codebase
MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelForCausalLM,
    "masked-language-modeling": AutoModelForMaskedLM,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}

# this dict provides the various schedulers that can be referenced in this codebase
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Instantiate a model, tokenizer and config."""
        super().__init__()
        self.save_hyperparameters(hparams)
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        # retrieve model config
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        # set some model-specific parameters
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        # retrieve model tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer

        # accomodate various filler cases
        if hparams.filler_case == "distinct":
            self.tokenizer.add_tokens(['(umm)', '(uhh)'])
        if hparams.filler_case == "unique":
            self.tokenizer.add_tokens(['[FILLER_WORD]'])

        # retrieve model
        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def get_lr_scheduler(self):
        """Instantiate a learning rate scheduler."""
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and scheduler (linear warmup and decay)."""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        """Function for completing a step of the testing loop."""
        return self.validation_step(batch, batch_nb)

    # def test_epoch_end(self, outputs):
    #     """Function for computing metrics at the end of the test loop."""
    #     return self.validation_epoch_end(outputs)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)
        return int((dataset_size / effective_batch_size) * self.hparams.max_epochs)

    def setup(self, stage):
        """Data preparation and dataloader initialization."""
        self.prepare_data() 
        self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)
        if stage == "fit":
            self.val_loader = self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)
        elif stage == "test":
            self.test_loader = self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        """This function is defined in each model subclass."""
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def _feature_file(self, type_path):
        """Returns the path to the cached dataset being referenced."""
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                type_path,
                self.hparams.filler_case,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
            ),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """This function saves the model configuration to config.json as well as the model itself 
        and the tokenizer."""
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.config.total_steps = self.total_steps
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default="bert-base-cased",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--config_name",
            default="bert-base-cased",
            type=str,
            help="Pretrained config name or path if not the same as model_name."
        )
        parser.add_argument(
            "--tokenizer_name",
            default="bert-base-cased",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name.",
        )
        parser.add_argument(
            "--cache_dir",
            default="./.cache",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3.",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config.",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config.",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config.",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config.",
        )
        parser.add_argument("--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam."
        )
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler.",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some."
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer."
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps."
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="The num of workers passed to the DataLoader."
        )
        parser.add_argument(
            "--num_train_epochs",
            dest="max_epochs",
            default=50,
            type=int,
            help="The num of workers passed to the DataLoader."
        )
        parser.add_argument(
            "--train_batch_size",
            default=32,
            type=int,
        )
        parser.add_argument(
            "--eval_batch_size",
            default=32,
            type=int,
        )
        parser.add_argument(
            "--adafactor",
            action="store_true",
        )


class LoggingCallback(pl.Callback):
    """This class provides hooks for logging values at the end of each batch, each validation and 
    each test."""
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        # Log lr
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("\n***** Validation results *****\n")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("\n***** Test results *****\n")
        metrics = trainer.callback_metrics  # rename val_loss to test loss
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "results/test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser, root_dir) -> None:
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the training files based on the POM (2014) dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets."
        )
    parser.add_argument(
        "--gpus",
        default=0,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none.",
    )
    parser.add_argument(
        "--max_grad_norm",
        dest="gradient_clip_val",
        default=1.0,
        type=float,
        help="Value used to clip the global gradient norm for the optimizer."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        default=1,
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        default=1,
        type=int,
        help="Evaluate the model every n training epochs."
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Run the training."
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Run the testing."
    )
    parser.add_argument(
        "--do_perplexity",
        action="store_true",
        help="Whether to compute perplexity."
    )
    parser.add_argument(
        "--do_predict_confidence",
        action="store_true",
        help="Whether to compute a confidence prediction."
    )
    parser.add_argument(
        "--use_mlm",
        action="store_true",
        help="Modelling using MLM model during training. Inference using the MLM model during testing."
    )
    parser.add_argument(
        "--fast_dev_run",
        action='store_true',
        help="Runs 1 batch of train, test and val to find any bugs."
    )

def generic_train(
        model: BaseTransformer,
        args: argparse.Namespace,
        model_path: str,
        early_stopping_callback=False,
        logger=None,
        extra_callbacks=[],
        checkpoint_callback=None,
        logging_callback=None,
    ):
    """This function accepts a model and parsed arguments in order to configure the logging and 
    checkpoint callbacks."""
    output_dir = Path(args.output_dir)
    logger_path = output_dir.joinpath("runs")
    cp_path = output_dir.joinpath("checkpoints")

    pl.seed_everything(args.seed)

    # init dataloaders
    if args.do_train or args.fast_dev_run:
        model.setup(stage="fit")

    if args.do_test:
        model.setup(stage="test")

    # init Tensorboard logger
    tb_logger = TensorBoardLogger(logger_path, name=model_path)

    # set checkpoint paths
    filepath = os.path.join(cp_path, model_path)
    print('filepath: ', filepath)

    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=filepath,
            # filename=,
            prefix="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    # GPU training parameters
    if args.gpus:
        train_params["gpus"] = args.gpus

    if args.gpus >= 1:
        train_params["benchmark"] = True
        train_params["precision"] = 16

    if args.gpus > 1:
        train_params["accelerator"] = "ddp"
        train_params["auto_select_gpus"] = True

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        # comment line below if no logging_callback
        callbacks=[logging_callback] + extra_callbacks,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        # early_stop_callback=early_stopping_callback,
        **train_params
    )

    return trainer