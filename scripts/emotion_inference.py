import argparse
from tabnanny import verbose

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import torch.nn.functional as F


def perform_emotion_analysis(model_name, ds_path, save_data, target_names, verbose=False):
    """
    Performs emotion analysis and saves the newly labeled data in ds_path
    :param model_name: str
        model name on the HuggingFace hub
    :param ds_path: str
        file path of the pd.DataFrame 
    :param save_data: str
        file path of to save the pd.DataFrame 
    :param target_names: List[str]
        list of the specific model's class names
    :param verbose: Boolean
        switch to print additional information on the console
    """

    # batch the data because classifying everything at once takes too long
    inference_batch_size = 4
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_data = pd.read_pickle(ds_path)
    # we only want to label future statements
    input_data = input_data.loc[input_data["future_statement"] == 0].reset_index(drop=True)
    batched_targets = []
    for i in range(input_data.shape[0] // inference_batch_size + 1):
        #break if we already at the end of the data
        if i * inference_batch_size == input_data.shape[0]:
            break
        if verbose:
            print("inference on batch", i + 1)

        input_data_str = input_data["candidate"].values.tolist()[
                         i * inference_batch_size:min(i * inference_batch_size + inference_batch_size,
                                                      input_data.shape[0])]
        encoded_inputs = tokenizer(input_data_str, padding=True, truncation="longest_first", return_tensors="pt")
        targets = model(**encoded_inputs)
        #target contains logits, softmax to get probabilty distribution over all classes
        batched_targets.append(F.softmax(targets[0].detach(), dim=1))
    
    #concatinate the batches and add the labels to the DataFrame 
    targets = torch.cat(batched_targets, dim=0).numpy()
    for i, target_name in enumerate(target_names):
        input_data[target_name] = targets[:, i]
    if verbose:
        print(input_data)
    input_data.to_pickle(save_data)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--import_data', nargs="?", default="data/classification_future.pkl", action='store',
                            help='Import data from a given path. Tries to use an existing dataset if not provided.')
    args_parser.add_argument('--save_data', nargs="?", default="data/emotion_classification.pkl", action='store',
                            help='save data to a given path. Tries to use an existing name if not provided.')                    
    args_parser.add_argument("-v", "--verbose", action="store_true", 
                            help="Print additional information on the console.")
    args = args_parser.parse_args()


    model_name = "j-hartmann/emotion-english-distilroberta-base"
    ds_path = args.import_data
    save_data = args.save_data
    target_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    verbose = args.verbose

    perform_emotion_analysis(model_name, ds_path, save_data, target_names, verbose)


