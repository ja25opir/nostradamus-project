import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', nargs=1, action='store',
                        help='Analyze data from a given path.')
    args = parser.parse_args()

    data = pd.read_pickle(args.input_file[0]).iloc[:, 2:]
    max_col = pd.Series(data.idxmax(axis="columns"), name='emotion class')
    max_vals = data.max(axis=1)
    low_confidence_idx = max_vals.index[max_vals <= 0.5].tolist()
    max_col.iloc[low_confidence_idx] = 'undecided'

    max_col.value_counts().plot(kind="pie")
    plt.show()
