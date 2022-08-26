import logging 
import datetime
from functools import partial


import numpy as np

import pandas as pd

from transformers import AutoTokenizer

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from small_text.active_learner import PoolBasedActiveLearner
from small_text.initialization import random_initialization_balanced
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling, LeastConfidence, PredictionEntropy
from small_text.integrations.transformers.datasets import TransformersDataset

#CONFIG
TRANSFORMER_MODEL = TransformerModelArguments('roberta-base')
QUERY_STRATEGY = PredictionEntropy()
NUM_ITERATIONS = 10
NUM_QUERY_SAMPLES = 20
LABELS = ["future_statement", "none"]
SAVE_SAMPLE = False 
REPORT = True 
SAVE_BEST = False 


BEST_MODEL = None 

def main():
    # Active learning parameters
    clf_factory = TransformerBasedClassificationFactory(TRANSFORMER_MODEL,
                                                        len(SPEECH_ACTS),
                                                        kwargs=dict({'device': 'cuda'}))


    # Prepare some data
    train_labeled, train, test = load_dataset()
    
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='.cache/')
    x_train = preprocess_data(tokenizer, train["data"].values, train["target"].values)
    y_train = train_labeled["target"].values
    x_test = preprocess_data(tokenizer, test["data"].values, test["target"].values)


    #Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, QUERY_STRATEGY, x_train)
    labeled_indices = initialize_active_learner(active_learner, y_train)
    print(classification_report(x_test.y, active_learner.classifier.predict(x_test), target_names=SPEECH_ACTS))
    save_model(active_learner)

    try:
        perform_active_learning(active_learner, x_train, labeled_indices, x_test, train.values)
    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')
    
    print(classification_report(x_test.y, active_learner.classifier.predict(x_test), target_names=SPEECH_ACTS))



def load_dataset():

    labeled_data = {class_name.lower(): value for class_name, value in zip(SPEECH_ACTS, range(len(SPEECH_ACTS)))}

    # load labeled dataset
    dfs = []
    for label in labeled_data.keys():
        with open(f"{label}s.txt") as f:
            data = f.readlines()
        data = [line.replace("\n", " ") for line in data]
        df = pd.DataFrame(columns=["data", "target"])
        df["data"] = data
        df["target"] = labeled_data[label]
        dfs.append(df)
    dataset = pd.concat(dfs, axis=0, ignore_index=True)
    train_labeled, test_labeled = train_test_split(dataset, test_size=0.2, stratify=dataset["target"], random_state=42)

    # load unlabeled
    with open("unlabeled.txt") as f:
        data = f.readlines()
        data = [line.replace("\n", " ") for line in data]
        train_unlabeled = pd.DataFrame(columns=["data", "target"])
        train_unlabeled["data"] = data
        train_unlabeled["target"] = len(SPEECH_ACTS) #dummy label since TransformersDataset expects it  

    print(len(train_labeled))
    return train_labeled, pd.concat([train_labeled, train_unlabeled], axis=0, ignore_index=True), test_labeled 

def save_sample(oracle, sentence):
    files = {"c":"commissives.txt", "d":"directives.txt", "n": "nones.txt"}
    with open(files[oracle], mode="a") as f:
        f.write(sentence + "\n")


def preprocess_data(tokenizer, data, labels, max_length=500):

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
        data_out.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i]))
    
    return TransformersDataset(data_out)

def perform_active_learning(active_learner, train, labeled_indices, test, text):
    # Perform [NUM_ITERATIONS] iterations of active learning...
    for i in range(NUM_ITERATIONS):
        # ...where each iteration consists of labelling [QUERY_SAMPLES] samples
        q_indices = active_learner.query(num_samples=QUERY_SAMPLES)
        
        y = []
        annotations = {"fs":0, "n":1}
        for q_index in q_indices:
            print(text[q_index][0])
            while True:
                oracle = input("[f]uture statement, [n]one, [i]gnore")
                if oracle in list(annotations.keys()):
                    print("Label als", annotations[oracle])
                    if SAVE_SAMPLE:
                        save_sample(oracle, text[q_index][0])

                    y.append(annotations[oracle])
                    break 
                if oracle == "i":
                    y.append(None)
                    print("ignored")
                    break
                else:
                    print("improper input. Try again!")
                

        # Return the labels for the current query to the active learner.

        active_learner.update(y)

        labeled_indices = np.concatenate([q_indices, labeled_indices])

        print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
        evaluate(active_learner, train[labeled_indices], test)
        save_model(active_learner)


def initialize_active_learner(active_learner, y_train):
    x_indices_initial = random_initialization_balanced(y_train, n_samples=len(y_train))
    y_initial = np.array([y_train[i] for i in x_indices_initial])

    active_learner.initialize_data(x_indices_initial, y_initial)

    return x_indices_initial

def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)
    f1_score_test = f1_score(test.y, y_pred_test, average='macro')

    print('Train accuracy: {:.2f}'.format(
        f1_score(train.y, y_pred, average='macro')))
    print('Test accuracy: {:.2f}'.format(f1_score_test))
    
    if REPORT:
        print(classification_report(test.y, y_pred_test, target_names=SPEECH_ACTS))

    print('---')

    # preserve only best
    if SAVE_BEST:
      global BEST_MODEL 
      if BEST_MODEL is not None:
        best_y_pred_test = BEST_MODEL.classifier.predict(test) 
        if f1_score_test > f1_score(test.y, best_y_pred_test, average='macro'): 
          print("save new best model")
          BEST_MODEL = active_learner 
      else: 
        BEST_MODEL = active_learner 



def save_model(model):
    date = datetime.today().strftime("%Y%m%d")
    name = input("Name of to be saved model[Enter to skip, no save]: ")
    if name != "":
        model.save(F"../models/{date}_{name}.pkl")
    
    #always save best model
    if BEST_MODEL is not None:
      BEST_MODEL.save(f"../models/{date}_best.pkl")


if __name__ == '__main__':
    logging.getLogger('small_text').setLevel(logging.INFO)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

    main()