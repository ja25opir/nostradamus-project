import sys

import pandas as pd
import argparse


def import_data(path):
    f = open(path, 'r', encoding='utf8')
    lines = f.readlines()
    f.close()

    data = {'candidates': [], 'label': []}

    for index, line in enumerate(lines):
        data['candidates'].append(line.strip())
        data['label'].append(0)
        if index == 10:
            break

    df = pd.DataFrame(data=data, columns=['candidates', 'label'])
    df.to_pickle('../data/candidates.pkl')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--import_data', nargs='?', action='store')
    args = parser.parse_args()
    if args.import_data:
        candidates = import_data(args.import_data)
    else:
        try:
            candidates = pd.read_pickle('../data/candidates.pkl')
        except FileNotFoundError:
            print("No pickled data available. Use --import argument.")
            sys.exit(1)

    print(candidates)
