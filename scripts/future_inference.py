import argparse
from tabnanny import verbose

from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.active_learner import PoolBasedActiveLearner
from transformers import AutoTokenizer
import pandas as pd


def preprocess_data(tokenizer, data, max_length=500):
    """
    Converts a list of string sentence into an TransformersDataset as model input
    :param tokenizer: AutoTokenizer
        containing the tokenizer of the model
    :param data: List[str]
        the text data
    :param max_len: int
        Maximum sequence length to encode 
    :return TransformerDataset
        the dataset to input into the model
    """
    data_out = []
    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )
        data_out.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], 0))  # Dummy label, ingore!
    return TransformersDataset(data_out)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--import_data', nargs="?", default="data/classification_set.pkl", action='store',
                            help='Import data from a given path. Tries to use an existing dataset if not provided.')
    args_parser.add_argument("-v", "--verbose", action="store_true", 
                            help="Print additional information on the console.")
    args_parser.add_argument('--save_data', nargs="?", default="data/future_classification.pkl", action='store',
                            help='save data to a given path. Tries to use an existing name if not provided.')
    args = args_parser.parse_args()

    model = PoolBasedActiveLearner.load("models/20220824_76futacc.pkl").classifier
    ds_path = args.import_data

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    input_data = pd.read_pickle(ds_path)
    if verbose:
        print(input_data)
    input_data_str = input_data["candidate"].values
    encoded_inputs = preprocess_data(tokenizer, input_data_str)

    targets = model.predict(encoded_inputs)
    input_data["future_statement"] = targets
    if verbose:
        print("label_result")
        print(input_data)
    input_data.to_pickle(args.save_data)
