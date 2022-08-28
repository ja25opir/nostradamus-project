import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_emotions_distribution(plt_data, drop_neutral=False):
    """
    Method that creates a horizontal bar plot displaying the distribution of assigned emotion classes.
    Can be done with or without the "neutral"-class.
    :param plt_data: pd.Series
        with classes and distributions
    :param drop_neutral: Boolean
        whether to drop the "neutral"-class or not
    """
    emotions_count = plt_data.value_counts(normalize=True, ascending=True).drop(
        'neutral') if drop_neutral else plt_data.value_counts(normalize=True, ascending=True)
    labels = emotions_count.index.tolist()
    y_pos = np.arange(len(labels))
    plt.style.use('bmh')
    plt.title('Distribution after emotion detection excluding "neutral"') if drop_neutral else plt.title(
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

    # calculate distribution of classes
    assigned_emotions = pd.Series(emotions.idxmax(axis='columns'), name='emotion class')
    max_vals = emotions.max(axis=1)
    low_confidence_idx = max_vals.index[max_vals <= 0.5].tolist()
    assigned_emotions.iloc[low_confidence_idx] = 'undecided'

    plot_emotions_distribution(assigned_emotions, drop_neutral=True)
    plot_emotions_distribution(assigned_emotions, drop_neutral=False)
