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
from transformers.data.processors.utils import DataProcessor  # InputExample, InputFeatures
# from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.metrics import mean_squared_error


logger = logging.getLogger(__name__)


CONFIDENCE_DATASETS = {"distinct": ("train_classif.tsv",
                            "val_classif.tsv",
                            "test_classif.tsv"
                        ),
                        "unique": ("train_classif_unique.tsv",
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
            filler_case: 'distinct','unique','no_filler'.
            dir: File path string.
            evaluate: Whether to evaluate val dataset.
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
    """Determine the max length of an example.

    Note: https://huggingface.co/transformers/v2.4.0/_modules/transformers/data/processors/glue.html

    Args:
            data: Panda dataframe containing examples and labels.
    """
    input_ids = []
    attention_masks = []
    features = []

    examples = data.example.values
    labels = data.label.values

    for example in examples:
        encoded_dict = tokenizer.batch_encode_plus(
                            re.split("\.\s+", example.rstrip(".").strip()),
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

    for i in range(len(examples)):
        input_id = input_ids.narrow(0, i, 1).squeeze()
        attention_mask = attention_masks.narrow(0, i, 1).squeeze()
        label_id = label_ids.narrow(0, i, 1)
        features.append(InputFeatures(input_id=input_id, attention_mask=attention_mask, label_id=label_id))

    return features


class InputFeatures():
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

def load_and_cache_examples(args, tokenizer, evaluate=False):
    data = load_tsv(args.filler_case, args.data_dir, 'train')
    max_len = max_length(data, tokenizer)

    logger.info("Creating features from dataset file at %s", args.data_dir)
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train', args.model_name, args.task_name, args.filler_case))

    # logger.info("Max length of tokenized segment is %s", max_len)
    label_list = data.label.values.tolist()
    # print('label_list: ', type(label_list))
    features = convert_examples_to_features(data, tokenizer, max_len)
    print(features[0].input_id.size())
    print(features[0].label_id.size())
    # input_ids, attention_masks = convert_examples_to_features(data.example.values, tokenizer, max_len)
    # print('input_ids: ', type(input_ids), 'attention_masks: ', type(attention_masks))
    # input_ids = pad_sequence(input_ids, batch_first=True)
    # attention_mask = pad_sequence(attention_masks, batch_first=True)
    # labels = torch.tensor(data.label.values, dtype=torch.float)

    # print('\n input_ids dimensions: {}'.format(input_ids.size()))
    # print('\n attention_mask dimensions: {}'.format(attention_mask.size()))
    # print('\n labels dimensions: {}'.format(labels.size()))

    print('So far so good!')

    # if args.local_rank in [-1, 0]:
    #         logger.info("Saving features into cached file %s", cached_features_file)
    #         torch.save([input_ids, attention_mask, labels], cached_features_file)

    # dataset = TensorDataset(input_ids, attention_mask, labels)

    # return dataset


class POM(DataProcessor):
    """Variant of transformers.data.processors.glue.Sst2Processor"""
    def __init__(self, filler_case, args):
        """
        Creation of the various datasets for confidence regression analysis
        and sentiment analysis.

        Args:
            filler_case: 'distinct','unique','no_filler'
            args: parser object
            task (string): 'confidence','sentiment'
        """
        self.args = args
        self.train = confidence_datasets[filler_case][0]
        self.dev = confidence_datasets[filler_case][1]
        #     if filler_case == 'distinct':
        #         self.train = confidence_datasets['distinct'][0]
        #         self.dev = confidence_datasets['distinct'][1]
        #     elif filler_case == 'unique':
        #         self.train = confidence_datasets['unique'][0]
        #         self.dev = confidence_datasets['unique'][1]
        #     elif filler_case == 'no_filler':
        #         self.train = confidence_datasets['no_filler'][0]
        #         self.dev = confidence_datasets['no_filler'][1]

    # def get_example_from_tensor_dict(self, tensor_dict):  # not sure this function is needed
    #     """Gets an example from a dict with tensorflow tensors.

    #     Args:
    #         tensor_dict: Keys and values should match the corresponding Glue
    #             tensorflow_dataset examples.
    #     """
    #     return InputExample(tensor_dict['idx'].numpy(),
    #                         tensor_dict['sentence'].numpy().decode('utf-8'),
    #                         None,
    #                         int(tensor_dict['label'].numpy()))

    def get_train_examples(self):
        """Gets a collection of :class:`InputExample` for the train dataset."""
        examples = self._create_examples(_read_tsv(self.train), 'train')

        return examples

    def get_dev_examples(self):
        """Gets a collection of :class:`InputExample` for the val dataset."""
        examples = self._create_examples(self._read_tsv(self.dev), 'dev')

        return examples

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets.

        Args:
            lines: A list containing each row of raw data as a list.
            set_type: a string denoting the set type. Either 'train', 'dev', or 'test'.
        """
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    data = load_tsv('distinct', evaluate=False)
    print(max_length(data, tokenizer))
