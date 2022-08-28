from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.active_learner import PoolBasedActiveLearner
from transformers import AutoTokenizer
import pandas as pd


def preprocess_data(tokenizer, data, max_length=500):
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
    model = PoolBasedActiveLearner.load("models/20220824_76futacc.pkl").classifier
    ds_path = "data/classification_set.pkl"

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    input_data = pd.read_pickle(ds_path)
    print(input_data)
    input_data_str = input_data["candidate"].values
    encoded_inputs = preprocess_data(tokenizer, input_data_str)

    targets = model.predict(encoded_inputs)
    input_data["future_statement"] = targets
    print("label_result")
    print(input_data)
    input_data.to_pickle("data/future_classification.pkl")
