import numpy as np
import os
import librosa
from sklearn.mixture import GaussianMixture
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gmm_train(paths: List[str], n_mfcc, n_components) -> List[object]:

    gmm_list: List[object] = list()

    # training process
    for file in paths:
        # loading train recording
        y, sr = librosa.load(file)

        # extract MFCC features for train recording
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # TODO:
        #   manipulate n_mfcc

        # fitting GMM to MFCC features extracted
        gm = GaussianMixture(n_components=n_components, random_state=0).fit(np.transpose(m))

        # TODO:
        #   manipulate n_components

        gmm_list.append(gm)

    return gmm_list


def gmm_predict(paths: List[str], models, n_mfcc) -> List[int]:

    predicted_labels: List[int] = list()

    # classification
    for file in paths:
        # loading test recording
        y, sr = librosa.load(file)

        # extract MFCC features for train recording
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # iterating over GMM's
        scores = [gmm.score(np.transpose(m)) for gmm in models]

        predicted_label = np.argmax(scores) + 1
        predicted_labels.append(int(predicted_label))

    return predicted_labels


def get_accuracy(train_files, test_files, n_mfcc=39, n_components=32) -> float:

    gmm_list: List[object] = gmm_train(train_files, n_mfcc=n_mfcc, n_components=n_components)

    actual_lbls: List[int] = [int(os.path.split(file)[-1].split(".")[0][-1]) for file in test_files]

    predicted_lbls: List[int] = gmm_predict(test_files, gmm_list, n_mfcc=n_mfcc)

    # calculating accuracy
    accuracy: float = sum([l1 == l2 for l1, l2 in zip(actual_lbls, predicted_lbls)]) / len(train_files)

    return accuracy


def main(train_p, test_p) -> None:

    # reading train files paths
    train_files = [os.path.join(train_p, file) for file in os.listdir(train_p)]
    train_files.sort()

    # reading test files paths
    test_files = [os.path.join(test_p, file) for file in os.listdir(test_p)]
    test_files.sort()

    # set some values for two dynamic parameters to calculate its accuracy
    mfcc_vals = range(10, 41, 5)  # number of mfcc's in MFCC feature extraction
    n_components_vals = range(10, 41, 5)  # number of components in GaussianMixture model

    # get different classifiers accuracies
    accuracies = {(mfcc, comp): get_accuracy(train_files, test_files, n_mfcc=mfcc, n_components=comp)
                  for mfcc in mfcc_vals
                  for comp in n_components_vals}

    # create DataFrame to visualize data
    rows, columns, values = zip(*[(k[0], k[1], v) for k, v in accuracies.items()])

    df = pd.DataFrame({'n_mfcc': rows, 'n_components': columns, 'value': values}).pivot(index='n_mfcc',
                                                                                        columns='n_components',
                                                                                        values='value')
    #
    my_cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

    acc_heatmap = sns.heatmap(df, cmap=my_cmap, annot=True, fmt='.2f', cbar_kws={'label': 'accuracy'})
    acc_heatmap.invert_yaxis()

    acc_heatmap.set_title('Classification accuracies')
    acc_heatmap.set_xlabel('Number of Gaussian Mixture components.')
    acc_heatmap.set_ylabel("Number of MFCC's")

    plt.savefig('accuracies.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    #  insert your absolute path to a downloaded folder with files
    data_path = "/Users/stawager/Studia/s4/biometrics/5/classic/voices"
    assert os.path.isabs(data_path), "Your folder path is not absolute"

    test_path = os.path.join(data_path, os.listdir(data_path)[0])
    # assert test_path == "/Users/stawager/Studia/s4/biometrics/5/classic/voices/test", \
    #     "I don't see your test path"

    train_path = os.path.join(data_path, os.listdir(data_path)[1])
    # assert train_path == "/Users/stawager/Studia/s4/biometrics/5/classic/voices/train", \
    #     "I don't see your train path"

    main(train_path, test_path)

    quit()
