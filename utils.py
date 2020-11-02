import re
import logging
import os
import warnings 

import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, TextDataset
from transformers.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel
from transformers.file_utils import ModelOutput
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


def load_and_cache_examples_lm(args, tokenizer, evaluate=False):
    """Load and cache raw data.

    Args:
            filler_case: "distinct", "unique", "no_filler"
            dir: File path string.
            args: An argparser Namespace.
            tokenizer: An instance of the relevant tokenizer.
    """
    if evaluate:
        file_path = os.path.join(args.data_dir, CONFIDENCE_DATASETS[args.filler_case][1])
    else:
        file_path = os.path.join(args.data_dir, CONFIDENCE_DATASETS[args.filler_case][0])

    dataset = TextDataset(tokenizer,
                        file_path,
                        block_size=args.block_size,
                        overwrite_cache=args.overwrite_cache,
    )

    return dataset

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
    Update: Instead of padding for number of sequences tensors can be collated when batched by dataloader

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


class MaskedLMOutput(ModelOutput):
    loss: [torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None


class BertForMaskedLM(BertPreTrainedModel):
    """ A modified version of BertForMaskedLM that trains on the MLM task and also """
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-a;;;;;;;;; ttention."
            )

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        pooled_output = outputs[1]

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + ((pooled_output,))
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            pooled_output=pooled_output,
        )