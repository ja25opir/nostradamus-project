
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import pandas as pd 
import torch 
import torch.nn.functional as F 
import numpy as np

if __name__ == "__main__": 
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    ds_path = "data/classification_future.pkl" 
    target_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    inference_batch_size = 4 # because classifying everything at once takes too long
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    input_data = pd.read_pickle(ds_path)
    #input_data = input_data.head(200)
    input_data = input_data.loc[input_data["future_statement"] == 0] # we only want to label future statements 
    batched_targets = []
    for i in range(input_data.shape[0]//inference_batch_size + 1):
        if i*inference_batch_size == input_data.shape[0]:
            break
        print("inference on batch", i+1)
        input_data_str = input_data["candidate"].values.tolist()[i*inference_batch_size:min(i*inference_batch_size + inference_batch_size, input_data.shape[0])]
        #input_data = ["In 10 years everything goes to hell!", "AI will become dangerous."]
        encoded_inputs = tokenizer(input_data_str, padding=True, truncation="longest_first", return_tensors="pt")
        targets = model(**encoded_inputs)
        batched_targets.append(F.softmax(targets[0].detach(), dim=1))
    targets = torch.cat(batched_targets, dim=0).numpy()
    for i, target_name in enumerate(target_names):
        input_data[target_name] = targets[:, i]
    print(input_data)
    input_data.to_pickle("data/classification_fully_labeled.pkl")
