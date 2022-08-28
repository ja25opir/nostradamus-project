import sys
import numpy as np
import pandas as pd
import argparse


def import_data(path, output_name):
    """
    Reads sentences seperated with a newline from a given textfile and writes them into a DataFrame that gets saved.
    :param path: str
        path to textfile with sentences
    :param output_name: str
        filename of output DataFrame
    :return: DataFrame
        with columns "candidates" (str) containing the sentences and "labels" (int16) containing 0s.
    """
    print('importing text file...')
    f = open(path, 'r', encoding='utf8')
    lines = f.readlines()
    f.close()

    data = {'candidate': [], 'label': []}

    print('creating df...')
    for index, line in enumerate(lines):
        if len(line) > 1:
            data['candidate'].append(line.strip())
            data['label'].append(1)
    df = pd.DataFrame(data=data, columns=['candidate', 'label']).astype({'candidate': str, 'label': np.dtype('int16')})

    print('saving df as pickle...')
    df.to_pickle('data/{}.pkl'.format(output_name))
    return df


def split_data(size, data):
    """
    Shuffles a given DataFrame and splits it into two sets. The first one gets returned for the labeling process and
    the second one saved as data/candidates_unlabeled.pkl.
    :param size: int
        size of the first set
    :param data: DataFrame
        data to be split
    :return: DataFrame
        first DF resulting of the split
    """
    shuffled = data.sample(frac=1).reset_index(drop=True)
    data_slice = shuffled.loc[:size - 1, :]
    data_slice_unlabeled = shuffled.loc[size:, :]
    data_slice_unlabeled.to_pickle('data/candidates_unlabeled.pkl')
    return data_slice


def label_data(data):
    """
    Prints each sentence of a given DataFrame to the CLI and saves the user input as associated label.
    :param data: DataFrame
        containing candidates
    """
    for index, row in data.iterrows():
        print('---Candidate {}---'.format(index))
        print(row['candidate'])
        label = ''
        while not label.isnumeric():
            label = input('label: ')
        data.at[index, 'label'] = label
    data.to_pickle('data/candidates_labeled.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--import_data', nargs=2, action='store',
                        help='Import data from a given path. Tries to use an existing dataset if not provided.')
    parser.add_argument('--start_labeling', nargs=1, action='store',
                        help='Slice a set of a given size out of the original dataset and present every sentence for '
                             'a manual labeling process.')
    parser.add_argument('--show_data', nargs=1, action='store', help='Prints the head of a given DataFrame.')
    parser.add_argument('--count_classes', nargs=1, action='store',
                        help='Counts all candidates belonging to the classes 1 and 0 of a given DataFrame.')
    args = parser.parse_args()

    if args.import_data:
        candidates = import_data(args.import_data[0], args.import_data[1])
    else:
        try:
            candidates = pd.read_pickle('data/candidates.pkl')
        except FileNotFoundError:
            print('No pickled data available. Use --import argument.')
            sys.exit(1)

    if args.start_labeling:
        candidates_split = split_data(int(args.start_labeling[0]), candidates)
        label_data(candidates_split)

    if args.show_data:
        try:
            print(pd.read_pickle(args.show_data[0]).head)
        except FileNotFoundError:
            print('File not found.')
            sys.exit(1)

    if args.count_classes:
        count_data = pd.read_pickle(args.count_classes[0])
        print(count_data['label'].value_counts())
