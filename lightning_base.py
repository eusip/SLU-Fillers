# Adapted from: https://github.com/huggingface/transformers/blob/master/examples/research_projects/seq2seq-distillation/lightning_base.py

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from transformers import (
    AdamW,
    AutoConfig,
    # AutoModel,  # NOTE: issue with pickling AutoModel
    # AutoModelForPreTraining,
    # AutoModelForQuestionAnswering,
    # AutoModelForSeq2SeqLM,
    # AutoModelForSequenceClassification,
    # AutoModelForTokenClassification,
    # AutoModelWithLMHead,
    # AutoModelForMaskedLM,
    AutoTokenizer,
    # PretrainedConfig,
    # PreTrainedTokenizer,
    # # NOTE: for future reference
    # CONFIG_MAPPING, 
    # MODEL_FOR_MASKED_LM_MAPPING,
)
from transformers.optimization import (
    Adafactor,
#     get_cosine_schedule_with_warmup,
#     get_cosine_with_hard_restarts_schedule_with_warmup,
#     get_linear_schedule_with_warmup,
#     get_polynomial_decay_schedule_with_warmup,
)

# from transformers.utils.versions import require_version_examples

logger = logging.getLogger(__name__)

# require_version_examples("pytorch_lightning>=1.0.4")

# MODEL_MODES = {
#     "base": AutoModel,
#     "sequence-classification": AutoModelForSequenceClassification,
#     "question-answering": AutoModelForQuestionAnswering,
#     "pretraining": AutoModelForPreTraining,
#     "token-classification": AutoModelForTokenClassification,
#     "language-modeling": AutoModelWithLMHead,
#     "summarization": AutoModelForSeq2SeqLM,
#     "translation": AutoModelForSeq2SeqLM,
#     "mlm": AutoModelForMaskedLM,
# }


# # update this and the import above to support new schedulers from transformers.optimization
# arg_to_scheduler = {
#     "linear": get_linear_schedule_with_warmup,
#     "cosine": get_cosine_schedule_with_warmup,
#     "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
#     "polynomial": get_polynomial_decay_schedule_with_warmup,
#     # '': get_constant_schedule,             # not supported for now
#     # '': get_constant_schedule_with_warmup, # not supported for now
# }
# arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
# arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


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

        # NOTE: tokenization takes place in LightningDataModule but tokenizer needed for resizing the embedding in the model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )

        # accomodate various filler cases
        if hparams.dataset_name == "distinct_fillers":
            self.tokenizer.add_tokens(['(umm)', '(uhh)'])
        if hparams.dataset_name == "unique_filler":
            self.tokenizer.add_tokens(['[FILLER_WORD]'])

        # else:
        #     self.tokenizer: PreTrainedTokenizer = tokenizer
        # self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

        # env_cp = os.environ.copy()
        # self.node_rank, self.local_rank, self.world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']

        # self.is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
        # self.pl_trainer_gpus = enc_cp['PL_TRAINER_GPUS']

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    # NOTE: PyTorch Lightning cannot pickle scheduler.lr_lambda object. Scheduler has been disabled until bug report.
    # def get_lr_scheduler(self):
    #     get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
    #     scheduler = get_schedule_func(
    #         self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     return scheduler

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

        # scheduler = self.get_lr_scheduler()

        return [optimizer]  #, [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs: dict) -> dict:
        raise NotImplementedError("You must implement this for your task")

    # def total_steps(self) -> int:
    #     """The number of total training steps that will be run. Used for lr scheduler purposes."""
    #     num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
    #     effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
    #     return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    # NOTE: not needed when using Hf Datasets library
    # def _feature_file(self, mode):
    #     return os.path.join(
    #         self.hparams.data_dir,
    #         "cached_{}_{}_{}".format(
    #             mode,
    #             list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
    #             str(self.hparams.max_seq_length),
    #         ),
    #     )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        # self.tokenizer.save_pretrained(save_path)

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
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default=".cache",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from huggingface.co",
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
        parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        # parser.add_argument(
        #     "--lr_scheduler",
        #     default="linear",
        #     choices=arg_to_scheduler_choices,
        #     metavar=arg_to_scheduler_metavar,
        #     type=str,
        #     help="Learning rate scheduler",
        # )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--num_workers",
            default=8,
            type=int,
            help="kwarg passed to DataLoader",
        )
        parser.add_argument(
            '--preprocessing_num_workers', 
            default=8,
            type=int,
            help="The number of processes to use for the preprocessing.",
        )
        parser.add_argument(
            "--num_train_epochs",
            dest="max_epochs",
            default=3,
            type=int,
            # help = ,
        )
        parser.add_argument(
            "--train_batch_size",
            default=8,
            type=int,
            # help = ,
        )
        parser.add_argument(
            "--eval_batch_size",
            default=8,
            type=int,
            # help = ,
        )
        parser.add_argument(
            "--adafactor",
            action="store_true",
            # help = ,
        )
        # parser.add_argument(
        #     "--filler_case",
        #     default="unique",
        #     type=str,
        #     help="Set the filler case for this analysis - 'distinct', 'unique', 'none'.",
        # )


# class LoggingCallback(pl.Callback):
    # def on_batch_end(self, trainer, pl_module):
    #     lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
    #     lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
    #     pl_module.logger.log_metrics(lrs)

    # def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    #     rank_zero_info("***** Validation results *****")
    #     metrics = trainer.callback_metrics
    #     # Log results
    #     for key in sorted(metrics):
    #         if key not in ["log", "progress_bar"]:
    #             rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    # def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    #     rank_zero_info("***** Test results *****")
    #     metrics = trainer.callback_metrics
    #     # Log and save results to file
    #     output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
    #     with open(output_test_results_file, "w") as writer:
    #         for key in sorted(metrics):
    #             if key not in ["log", "progress_bar"]:
    #                 rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
    #                 writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser, root_dir) -> None:
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        type=str,
        required=False,
        help="The input data dir. Should contain the modeling data.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        help="The name of the dataset to use: 'distinct', 'unique', 'none'."
    )
    parser.add_argument(
        "--max_grad_norm",
        dest="gradient_clip_val",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    # parser.add_argument(
    #     "--do_train",
    #     action="store_true",
    #     help="Whether to run training.",
    # )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization.",
    )
    parser.add_argument(
        '--overwrite_cache', 
        action='store_true', 
        default=False,
        help="Overwrite the cached training and evaluation sets.",
    )
    parser.add_argument(
        '--experiment',
        default=None,
        type=str,
        required=True,
        help="Set the model for this analysis - 'LM', 'MLM', 'SentPred', 'SentPredFT'.",
    )

def add_trainer_args(parser, root_dir) -> None:
    # NOTE: To allow all pl args uncomment the following line:
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--gpus",
        default=0,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Whether to use full precision or native amp 16-bit half precision."
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

def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    early_stopping_callback=None,
    logger=None,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    pl.seed_everything(args.seed)

    # init project folders
    cdir = Path(args.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    # NOTE: only needed if _feature_file() is implemented
    # ddir = Path(args.data_dir)
    # ddir.mkdir(parents=True, exist_ok=True)
    odir = Path(args.output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    # add custom checkpoints
    # if checkpoint_callback is None:
    #     checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #         filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
    #     )
    # if early_stopping_callback:
    #     extra_callbacks.append(early_stopping_callback)
    # if logging_callback is None:
    #     logging_callback = LoggingCallback()

    train_params = {}

    if args.gpus > 1:
        train_params["accelerator"] = "ddp"
    elif args.gpus > 0:
        train_params["benchmark"] = True
        train_params["precision"] = 16
    else:
        train_params["accelerator"] = extra_train_kwargs.get("accelerator", None)

    train_params["log_every_n_steps"] = 50

    # train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    # train_params["profiler"] = extra_train_kwargs.get("profiler", None)

    trainer = pl.Trainer.from_argparse_args(
        args,
        # weights_summary=None,
        # callbacks=[logging_callback] + extra_callbacks,
        logger=logger,
        # checkpoint_callback=checkpoint_callback,
        **train_params,
    )

    return trainer