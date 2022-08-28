import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_emotions_distribution(plt_data, drop_neutral=False):
    emotions_count = plt_data.value_counts(normalize=True, ascending=True).drop(
        'neutral') if drop_neutral else plt_data.value_counts(normalize=True, ascending=True)
    labels = emotions_count.index.tolist()
    y_pos = np.arange(len(labels))
    plt.style.use('bmh')
    plt.title('Distribution after emotion detection exluding "neutral"') if drop_neutral else plt.title(
        'Distribution after emotion detection')
    plt.barh(y_pos, emotions_count)
    plt.yticks(y_pos, labels)
    plt.xlabel('Percentage')
    plt.savefig('documentation/figures/emotions_distribution_drop_neutral.png') if drop_neutral else plt.savefig(
        'documentation/figures/emotions_distribution.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', nargs=1, action='store',
                        help='Analyze data from a given path.')
    args = parser.parse_args()

    data = pd.read_pickle(args.input_file[0])
    emotions = data.iloc[:, 2:]
    sentences = data.iloc[:, 0]
    assigned_emotions = pd.Series(emotions.idxmax(axis='columns'), name='emotion class')
    max_vals = emotions.max(axis=1)
    low_confidence_idx = max_vals.index[max_vals <= 0.5].tolist()
    assigned_emotions.iloc[low_confidence_idx] = 'undecided'

    plot_emotions_distribution(assigned_emotions, drop_neutral=True)
    plot_emotions_distribution(assigned_emotions, drop_neutral=False)
