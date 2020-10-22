from configs.database_names import *

from itertools import combinations
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import seaborn as sns
import os


sns.set(style="ticks")


def testing_set_classification(sbj_id="A01"):
    try:
        f = np.load(os.path.join(features_test_folder, f"{sbj_id}E_features.npy"), allow_pickle=True).item()
        f_train = np.load(os.path.join(features_train_folder, f"{sbj_id}T_features.npy"), allow_pickle=True).item()
    except FileNotFoundError as erro:
        raise FileNotFoundError(
            f"Verifique se os arquivos de caracteristicas do sujeito de id {sbj_id} "
            f"existem nas pastas {features_test_folder} e {features_train_folder}"
        )

    first = True
    for i, j in combinations(e_classes, 2):

        f_test = np.array(
            [k for k in f[f'{i}{j}'] if k[-1] in e_dict]
        )

        x_train = f_train[f'{i}{j}'][:, :-1]
        y_train = f_train[f'{i}{j}'][:, -1]

        x_test = f_test[:, :-1]
        y_test = f_test[:, -1]

        linear = True
        if linear:
            svm_model = svm.LinearSVC(C=0.1, max_iter=5000)
        else:
            svm_model = svm.SVC()

        svm_model.fit(x_train, y_train)

        if first is True:
            y_prediction = np.array([svm_model.predict(x_test)]).T
            first = False
        else:
            y_prediction = np.append(y_prediction, np.array([svm_model.predict(x_test)]).T, axis=1)

    y_prediction_final = mode(y_prediction, axis=1).mode
    res = np.array([y_prediction_final == y_test.reshape(-1, 1)])

    confusion_df = pd.DataFrame(
        np.zeros([len(e_classes), len(e_classes)]),
        index=e_classes, columns=e_classes
    )

    for i_cnt, i in enumerate(y_prediction_final):
        confusion_df.loc[e_dict[y_test[i_cnt]], e_dict[y_prediction_final[i_cnt, 0]]] += 1

    confusion_df = confusion_df.rename(
        columns={"l": "Esquerda", "r": "Direita", "f": "Pés", "t": "Língua"},
        index={"l": "Esquerda", "r": "Direita", "f": "Pés", "t": "Língua"}
    )

    confusion = confusion_df.to_numpy()
    confusion_percent = confusion / 72
    pe = np.trace(confusion) / np.sum(confusion)
    po = np.dot(np.sum(confusion, axis=0), np.sum(confusion, axis=1)) / (np.sum(confusion)**2)
    kappa = (pe - po) / (1 - po)

    ax = sns.heatmap(confusion_df, cmap="Blues", annot=confusion_percent, linewidths=1.5)
    plt.yticks(va="center")
    plt.xticks(va="center")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")

    plt.title(f"{sbj_id[1:]}")

    print(f"Taxa de acerto {sbj_id}:", res.mean(), f"kappa: {kappa}")

    ax.get_figure().savefig(f"C:/Users/victo/Desktop/{sbj_id}_confusion.png")
    plt.cla()
    plt.clf()
    plt.close()

    return res.mean(), kappa