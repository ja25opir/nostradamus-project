import sys

import pandas as pd
import argparse


def import_data(path):
    print('importing text file...')
    f = open(path, 'r', encoding='utf8')
    lines = f.readlines()
    f.close()

    data = {'candidate': [], 'label': []}

    print('creating df...')
    for index, line in enumerate(lines):
        if len(line) > 1:
            data['candidate'].append(line.strip())
            data['label'].append(0)
    df = pd.DataFrame(data=data, columns=['candidate', 'label'])

    print('saving df as pickle...')
    df.to_pickle('../data/candidates.pkl')
    return df


def split_data(size, data):
    shuffled = data.sample(frac=1).reset_index(drop=True)
    data_slice = shuffled.loc[:size - 1, :]
    data_slice_unlabeled = shuffled.loc[size:, :]
    data_slice_unlabeled.to_pickle('../data/candidates_unlabeled.pkl')
    return data_slice


def label_data(data):
    for index, row in data.iterrows():
        print(row['candidate'])
        label = ''
        while not label.isnumeric():
            label = input('label: ')
        data.at[index, 'label'] = label
        print('------')
    data.to_pickle('../data/candidates_labeled.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--import_data', nargs=1, action='store',
                        help='Import data from a given path. Tries to use an existing dataset if not provided.')
    parser.add_argument('--start_labeling', nargs=1, action='store',
                        help='Slice a set of a given size out of the original dataset and present every sentence for '
                             'a manual labeling process.')
    parser.add_argument('--show_data', nargs=1, action='store', help='Prints the head of a selected DataFrame.')
    args = parser.parse_args()

    if args.import_data:
        candidates = import_data(args.import_data[0])
    else:
        try:
            candidates = pd.read_pickle('../data/candidates.pkl')
        except FileNotFoundError:
            print("No pickled data available. Use --import argument.")
            sys.exit(1)

    if args.start_labeling:
        candidates_split = split_data(int(args.start_labeling[0]), candidates)
        label_data(candidates_split)

    if args.show_data:
        try:
            print(pd.read_pickle(args.show_data[0]).head)
        except FileNotFoundError:
            print("File not found.")
            sys.exit(1)
