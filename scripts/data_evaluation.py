import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_emotions_distribution(plt_data, drop_neutral=False):
    """
    Creates a horizontal bar plot displaying the distribution of assigned emotion classes.
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


def plot_emotion_confidence(plt_data, plt_data_undecided):
    """
        Creates a bar plot displaying the mean confidence for every assigned emotion class. Every class gets two bars
        showing the mean confidence with and without using a threshold for pruning low confidence values to the
        'undecided' class.
        :param plt_data: DataFrame
            containing the assigned class and its confidence for a classified sentence each row
        :param plt_data_undecided: DataFrame
            same as plt_data but now containing a class 'undecided' for low confidence values
        """
    emotion_classes = ['neutral', 'joy', 'fear', 'sadness', 'surprise', 'disgust', 'anger']
    emotion_mean = []
    for idx, emotion in enumerate(emotion_classes):
        emotion_mean.append(plt_data.loc[plt_data['emotion class'] == emotion][0].mean())
    emotion_mean_undecided = []
    for idx, emotion in enumerate(emotion_classes):
        emotion_mean_undecided.append(
            plt_data_undecided.loc[plt_data_undecided['emotion class'] == emotion][0].mean())

    plt.style.use('bmh')
    r1 = np.arange(len(emotion_mean))
    barWidth = 0.4
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, emotion_mean, width=barWidth, label='without undecided')
    plt.bar(r2, emotion_mean_undecided, width=barWidth, label='with undecided')
    plt.xticks([r + barWidth / 2 for r in range(len(emotion_mean))], emotion_classes)
    plt.ylabel('confidence')
    plt.legend(bbox_to_anchor=(0.7, 0.9))
    plt.savefig('documentation/figures/emotions_confidence.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', nargs=1, action='store',
                        help='Analyze data from a given path.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--distribution', action='store_true',
                       help='Plot distribution of emotion classes.')
    group.add_argument('--confidence', action='store_true',
                       help='Plot confidence of emotion classes.')
    args = parser.parse_args()

    data = pd.read_pickle(args.input_file[0])
    emotions = data.iloc[:, 2:]

    # calculate distribution of classes, setup dataframes with assigned classes and confidences
    assigned_emotions = pd.Series(emotions.idxmax(axis='columns'), name='emotion class')
    max_vals = emotions.max(axis=1)
    emotion_vals = pd.concat([assigned_emotions, max_vals], axis=1)
    low_confidence_thresh = 0.5
    low_confidence_idx = max_vals.index[max_vals <= low_confidence_thresh].tolist()
    assigned_emotions.iloc[low_confidence_idx] = 'undecided'
    emotion_vals_undecided = pd.concat([assigned_emotions, max_vals], axis=1)

    if args.distribution:
        plot_emotions_distribution(assigned_emotions, drop_neutral=True)
        plot_emotions_distribution(assigned_emotions, drop_neutral=False)

    if args.confidence:
        plot_emotion_confidence(emotion_vals, emotion_vals_undecided)
