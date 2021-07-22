# Adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py

import logging
import os
import re
import warnings
from argparse import Namespace

import pandas as pd

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from pytorch_lightning import LightningDataModule

from transformers import (
    BertTokenizer, 
    BertTokenizerFast, 
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
)
from transformers.data import data_collator
from datasets import load_dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

hparams = {
    "cache_dir": '.cache',
    "config_name": 'bert-base-cased',
    "data_dir": 'data',
    "eval_batch_size": 8,
    "train_batch_size": 8,
    "model_name_or_path": 'bert-base-cased',
    "preprocessing_num_workers": 4,
    "num_workers": 4,
    "output_dir": 'output',
    "subset_name": 'swda',
    "tokenizer_name": 'bert-base-cased',
    "line_by_line": True,
    "pad_to_max_length": True,
    "overwrite_cache": True,
    "max_seq_length": 512,
    "mlm_probability": 0.15,
    "task": 'sts-b',
    "filler_case": 'unique',
    "num_train_epochs": '10',
    "do_train": False,
    "do_perplexity": False,
    "do_predict_confidence": False,
    "mlm": False,
    "fast_dev_run": False,
    "experiment": 'SentPred',
}

class NoFillers(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        self.hparams = Namespace(**hparams)

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

    def setup(self, stage):
        dataset = load_dataset(
                    path='csv', 
                    column_names=['text', 'labels'],             
                    delimiter='\t',           
                    data_files={'train': 'data/train_classif_without_fillers.tsv',
                                'validation': 'data/val_classif_without_fillers.tsv',
                                'test': 'data/test_classif_without_fillers.tsv',
                                }
                    )
        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=False,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=False,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        def label_to_list(example):
            value = example["labels"]
            label = [float(value)]
            example["labels"] = label
            return example

        tokenize = tokenize.map(
                label_to_list,
                batched=False,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        if self.hparams.experiment in ("LM", "MLM"):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=self.hparams.mlm, 
                mlm_probability=self.hparams.mlm_probability,
            )
        else:
            data_collator = DataCollatorForTokenClassification(
                tokenizer=tokenizer,
                max_length=self.hparams.max_seq_length, 
            )
            
        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.test = tokenize["test"]
        self.data_collator = data_collator

    # def _get_train_sampler(self):
    #     return RandomSampler(self.train)

    def train_dataloader(self):
        # train_sampler = self._get_train_sampler()

        return DataLoader(self.train,
                          batch_size = self.hparams.train_batch_size,
                        #   sampler = train_sampler,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    # def _get_val_sampler(self):
    #     return SequentialSampler(self.val)

    def val_dataloader(self):
        # val_sampler = self._get_val_sampler()

        return DataLoader(self.val,
                          batch_size = self.hparams.eval_batch_size,
                        #   sampler = val_sampler,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size = self.hparams.eval_batch_size,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class UniqueFiller(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        self.hparams = Namespace(**hparams)

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

    def setup(self, stage):
        dataset = load_dataset(
                    path='csv',
                    column_names=['text', 'labels'],                
                    delimiter='\t',           
                    data_files={'train': 'data/train_classif_unique.tsv',
                                'validation': 'data/val_classif_unique.tsv',
                                'test': 'data/test_classif_unique.tsv',
                                }
                    )
        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=False,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=False,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        def label_to_list(example):
            value = example["labels"]
            label = [float(value)]
            example["labels"] = label
            return example

        tokenize = tokenize.map(
                label_to_list,
                batched=False,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        if self.hparams.experiment in ("LM", "MLM"):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=self.hparams.mlm, 
                mlm_probability=self.hparams.mlm_probability,
            )
        else:
            data_collator = DataCollatorForTokenClassification(
                tokenizer=tokenizer,
                max_length=self.hparams.max_seq_length, 
            )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.test = tokenize["test"]
        self.data_collator = data_collator

    # def _get_train_sampler(self):
    #     return RandomSampler(self.train)

    def train_dataloader(self):
        # train_sampler = self._get_train_sampler()

        return DataLoader(self.train,
                          batch_size = self.hparams.train_batch_size,
                        #   sampler = train_sampler,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    # def _get_val_sampler(self):
    #     return SequentialSampler(self.val)

    def val_dataloader(self):
        # val_sampler = self._get_val_sampler()

        return DataLoader(self.val,
                          batch_size = self.hparams.eval_batch_size,
                        #   sampler = val_sampler,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size = self.hparams.eval_batch_size,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


class DistinctFillers(LightningDataModule):
    def __init__(self, hparams):
        """Initialize Dataloader parameters."""
        self.hparams = Namespace(**hparams)

        super().__init__()

        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

    def setup(self, stage):
        dataset = load_dataset(
                    path='csv',
                    column_names=['text', 'labels'],                
                    delimiter='\t',           
                    data_files={'train': 'data/train_classif_distinct.tsv',
                                'validation': 'data/val_classif_distinct.tsv',
                                'test': 'data/test_classif_distinct.tsv',
                                }
                    )
        tokenizer =  BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            return_tensors="pt",
        )

        column_names = dataset["train"].column_names
        text = "text"

        if self.hparams.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.hparams.pad_to_max_length else False

            def encode(examples):
                # Remove empty lines
                # examples[text] = [line for line in examples[text]
                #                     if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples[text],
                    padding=padding,
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_special_tokens_mask=False,
                )

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.hparams.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            def encode(examples):
                return tokenizer(examples[text], return_special_tokens_mask=False,)

            tokenize = dataset.map(
                encode,
                batched=True,
                num_proc=self.hparams.num_workers,
                remove_columns=["text"],
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

            if self.hparams.max_seq_length is None:
                self.hparams.max_seq_length = tokenizer.model_max_length
            else:
                if self.hparams.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.hparams.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.hparams.max_seq_length = min(self.hparams.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.hparams.max_seq_length) * self.hparams.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.hparams.max_seq_length]
                        for i in range(0, total_length, self.hparams.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenize = tokenize.map(
                group_texts,
                batched=True,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        def label_to_list(example):
            value = example["labels"]
            label = [float(value)]
            example["labels"] = label
            return example

        tokenize = tokenize.map(
                label_to_list,
                batched=False,
                num_proc=self.hparams.num_workers,
                load_from_cache_file=not self.hparams.overwrite_cache,
            )

        if self.hparams.experiment in ("LM", "MLM"):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=self.hparams.mlm, 
                mlm_probability=self.hparams.mlm_probability,
            )
        else:
            data_collator = DataCollatorForTokenClassification(
                tokenizer=tokenizer,
                max_length=self.hparams.max_seq_length, 
            )

        self.train = tokenize["train"]
        self.val = tokenize["validation"]
        self.test = tokenize["test"]
        self.data_collator = data_collator

    # def _get_train_sampler(self):
    #     return RandomSampler(self.train)

    def train_dataloader(self):
        # train_sampler = self._get_train_sampler()

        return DataLoader(self.train,
                          batch_size = self.hparams.train_batch_size,
                        #   sampler = train_sampler,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    # def _get_val_sampler(self):
    #     return SequentialSampler(self.val)

    def val_dataloader(self):
        # val_sampler = self._get_val_sampler()

        return DataLoader(self.val,
                          batch_size = self.hparams.eval_batch_size,
                        #   sampler = val_sampler,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size = self.hparams.eval_batch_size,
                          collate_fn=self.data_collator,
                          num_workers = self.hparams.num_workers,
                          drop_last= True,
        )


if __name__ == "__main__":
    nf = NoFillers(hparams)
    nf.setup(None)
    print(nf.train[0])
    print(nf.val[0])
    loader = nf.val_dataloader()
    batch = next(iter(loader))