import re
import logging
import pandas as pd
import os

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from sklearn.metrics import mean_squared_error


logger = logging.getLogger(__name__)


CONFIDENCE_DATASETS = {"distinct": (
                            "train_classif.tsv",
                            "val_classif.tsv",
                            "test_classif.tsv"
                        ),
                        "unique": (
                            "train_classif_unique.tsv",
                            "val_classif_unique.tsv",
                            "test_classif_unique.tsv"
                        ),
                        "no_filler": (
                            "train_classif_without_fillers.tsv",
                            "val_classif_without_fillers.tsv",
                            "test_classif_without_fillers.tsv"
                        )
}


def load_tsv(filler_case, dir, type_path):
    """Load the dataset into a pandas dataframe.

    Args:
            filler_case: "distinct", "unique", "no_filler"
            dir: File path string
            type_path: "train", "dev", "test"
    """
    if type_path == "train":
        df = pd.read_csv(os.path.join(dir, CONFIDENCE_DATASETS[filler_case][0]), delimiter='\t',
                        header=None, names=["example", "label"])
    elif type_path == "dev":
        df = pd.read_csv(os.path.join(dir, CONFIDENCE_DATASETS[filler_case][1]), delimiter='\t',
                        header=None, names=["example", "label"])
    elif type_path == "test":
        df = pd.read_csv(os.path.join(dir, CONFIDENCE_DATASETS[filler_case][2]), delimiter='\t',
                        header=None, names=["example", "label"])

    return df

def max_length(df, tokenizer):
    """Determine the max length of an example.

    Args:
            df: Panda dataframe containing examples.
            tokenizer: An instance of the relevant tokenizer.
    """
    max_len = 0
    examples = df.example.values

    for example in examples:
        tokenized = tokenizer(re.split("\.\s+", example.rstrip(".").strip()), add_special_tokens=True)
        segments  = tokenized['input_ids']
        for segment in segments:
            max_len = max(max_len, len(segment))

    return max_len

def convert_examples_to_features(data, tokenizer, max_length):
    """ Convert each review into a tensor of length 512, the maximum for BERT base.

    Note: https://huggingface.co/transformers/v2.4.0/_modules/transformers/data/processors/glue.html

    Args:
            data: Panda dataframe containing examples and labels.
    """
    input_ids = []
    attention_masks = []
    labels = []
    features = []

    examples = data.example.values

    for idx, example in enumerate(examples):
        encoded_dict = tokenizer.encode_plus(example,
                            add_special_tokens = True,
                            truncation = True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(torch.tensor(data.label[idx], dtype=torch.float))

    for idx,_ in enumerate(input_ids):
        features.append(InputFeatures(input_id=input_ids[idx].squeeze(), attention_mask=attention_masks[idx].squeeze(), label_id=labels[idx]))

    return features

def convert_examples_to_features_multi(data, tokenizer, max_length):
    """ Convert each sentence of an example into a tensor uniform in the dimension `max_length`.

    Note: https://huggingface.co/transformers/v2.4.0/_modules/transformers/data/processors/glue.html

    Args:
            data: Panda dataframe containing examples and labels.
    """
    input_ids = []
    attention_masks = []
    labels = []
    features = []

    examples = data.example.values

    for idx, example in enumerate(examples):
        sentences = re.split("\.\s+", example.rstrip(".").strip())
        for sentence in sentences:
            encoded_dict = tokenizer.encode_plus(sentence,
                                add_special_tokens = True,
                                max_length = max_length,
                                truncation = True,
                                padding = 'max_length',
                                return_attention_mask = True,
                                return_tensors = 'pt',
                            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(torch.tensor(data.label[idx], dtype=torch.float))

    for idx,_ in enumerate(input_ids):
        features.append(InputFeatures(input_id=input_ids[idx].squeeze(), attention_mask=attention_masks[idx].squeeze(), label_id=labels[idx]))

    return features

def convert_examples_to_features_batch(data, tokenizer, max_length):
    """ Convert each review into a tensor uniform in the dimensions number of sentences and
        `max_length`.

    Note: https://huggingface.co/transformers/v2.4.0/_modules/transformers/data/processors/glue.html

    Args:
            data: Panda dataframe containing examples and labels.
    """
    input_ids = []
    attention_masks = []
    features = []

    examples = data.example.values

    for example in examples:
        encoded_dict = tokenizer.batch_encode_plus(re.split("\.\s+", example.rstrip(".").strip()),
                            add_special_tokens = True,
                            max_length = max_length,
                            truncation = True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    label_ids = torch.tensor(data.label.values, dtype=torch.float)

    for idx,_ in enumerate(examples):
        input_id = input_ids.narrow(0, idx, 1).squeeze().permute(1, 0)  # B x F x S --> B x S x F
        attention_mask = attention_masks.narrow(0, idx, 1).squeeze().permute(1, 0)  # B x F x S --> B x S x F
        label_id = label_ids.narrow(0, idx, 1)
        features.append(InputFeatures(input_id=input_id, attention_mask=attention_mask, label_id=label_id))

    return features

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_id,
               attention_mask,
               label_id):
    self.input_id = input_id
    self.attention_mask = attention_mask
    self.label_id = label_id

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    mse = mean_squared_error(labels, preds)
    return {"mse": mse}

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    data = load_tsv('no_filler', "./data", "test")
    max_length = max_length(data, tokenizer)
    convert_examples_to_features(data, tokenizer, max_length)
