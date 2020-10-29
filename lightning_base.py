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
)


# logger = logging.getLogger(__name__)


MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelForMaskedLM,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
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
        """Initialize a model, tokenizer and config."""
        super().__init__()
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading

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

        # accomodate filler case
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

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
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
        return self.validation_step(batch, batch_nb)

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)
        return int((dataset_size / effective_batch_size) * self.hparams.max_epochs)

    def setup(self, stage):
        # if self.hparams.use_mlm:
        #     if stage == "fit":
        #         self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)
        #     elif stage == "test":
        #         self.val_loader = self.get_dataloader("dev", self.hparams.train_batch_size, shuffle=False)

        # self.prepare_data() # should automatically be called by trainer prior to setup
        self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)
        if stage == "fit":
            self.val_loader = self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)
        elif stage == "test":
            self.test_loader = self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def _feature_file(self, type_path):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                type_path,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                self.hparams.filler_case
            ),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count  # check if/where this is being updated
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default="bert-base-cased",
            type=str,
            # required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name",
            default="bert-base-cased",
            type=str,
            help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="bert-base-cased",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
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
            help="Learning rate scheduler",
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
            help="kwarg passed to DataLoader"
        )
        parser.add_argument(
            "--num_train_epochs",
            dest="max_epochs",
            default=50,
            type=int
        )
        parser.add_argument(
            "--train_batch_size",
            default=32,
            type=int
        )
        parser.add_argument(
            "--eval_batch_size",
            default=32,
            type=int
        )
        parser.add_argument(
            "--adafactor",
            action="store_true"
        )


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics  # rename val_loss to test loss
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "results/test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser, root_dir) -> None:

    # parser = pl.Trainer.add_argparse_args(parser)  # bug
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
    # parser.add_argument(
    #     "--precision",
    #     default=16,
    #     type=int,
    #     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    # )
    # parser.add_argument(
    #     "--amp_backend",
    #     default="apex",
    #     type=str,
    #     help="The PyTorch AMP `native` or NVIDIA `apex`.",
    # )
    # parser.add_argument(
    #     "--amp_level",
    #     type=str,
    #     default="O2",
    #     help="Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #     "See details at https://nvidia.github.io/apex/amp.html",
    # )
    parser.add_argument(
        "--max_grad_norm",
        dest="gradient_clip_val",
        default=1.0,
        type=float,
        help="Max gradient norm"
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
        help="random seed for initialization"
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Run the training."
    )
    parser.add_argument(
        "--use_mlm",
        action="store_false",
        help="Model training using MLM modelling"
    )
    parser.add_argument(
        "--do_perplexity",
        action="store_true",
        help="Whether to compute perplexity on the test set."
    )
    parser.add_argument(
        "--do_predict_confidence",
        action="store_true",
        help="Whether to compute a confidence prediction on the test set."
    )

def generic_train(
        model: BaseTransformer,
        args: argparse.Namespace,
        early_stopping_callback=False,
        logger=None,
        extra_callbacks=[],
        checkpoint_callback=None,
        logging_callback=None
        # **extra_train_kwargs
    ):
    pl.seed_everything(args.seed)

    # init dataloaders
    if args.do_train or args.fast_dev_run:
        model.setup(stage="fit")

    if args.do_predict_confidence:
        model.setup(stage="test")

    # init Tensorboard logger
    # tb_logger = TensorBoardLogger(os.path.join(args.output_dir, "runs",
    #                               "_".join(socket.gethostname(),
    #                               datetime.now().strftime('%Y-%m-%d_%H-%M'))),
    #                               name="confidence_prediction")
    tb_logger = TensorBoardLogger(os.path.join(args.output_dir, "output/runs"), name="confidence_prediction")

    # # init test results
    # ldir = Path(os.path.join(model.hparams.output_dir, "/results"))
    # ldir.mkdir(0o755,exist_ok=True)

    # # init model
    # odir = Path(model.hparams.output_dir)
    # odir.mkdir(0o755, exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir,
            prefix="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    if args.gpus >= 1:
        train_params["benchmark"] = True
        train_params["precision"] = 32
        train_params["amp_backend"] = "native"
        # train_params["amp_level"] = "O2"

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"
        train_params["auto_select_gpus"] = True

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        **train_params
    )

    return trainer