from configs.database_names import *

from itertools import combinations
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import seaborn as sns


sns.set(style="ticks")


def testing_set_classification(sbj_id="A01"):
    try:
        f = np.load(os.path.join(features_test_loc, f"{sbj_id}E_features.npy"), allow_pickle=True).item()
        f_train = np.load(os.path.join(features_train_loc, f"{sbj_id}T_features.npy"), allow_pickle=True).item()
    except FileNotFoundError as erro:
        raise FileNotFoundError(
            f"Verifique se os arquivos de caracteristicas do sujeito de id {sbj_id} "
            f"existem nas pastas {features_test_loc} e {features_train_loc}"
        )

    first = True
    for i, j in combinations(e_classes, 2):
        x_train = f_train[f'{i}{j}'][:, :-1]
        y_train = f_train[f'{i}{j}'][:, -1]

        x_test = f[f'{i}{j}'][:, :-1]
        y_test = f[f'{i}{j}'][:, -1]

        svm_model = svm.SVC()
        svm_model.fit(x_train, y_train)

        if first is True:
            y_prediction = np.array([svm_model.predict(x_test)]).T
            first = False
        else:
            y_prediction = np.append(y_prediction, np.array([svm_model.predict(x_test)]).T, axis=1)

    y_prediction_final = mode(y_prediction, axis=1).mode
    res = np.array([y_prediction_final == y_test.reshape(-1, 1)])

    print("Taxa de acerto:", res.mean())

    confusion_df = pd.DataFrame(
        np.zeros([len(e_classes), len(e_classes)]),
        index=e_classes, columns=e_classes
    )

    for i_cnt, i in enumerate(y_prediction_final):
        confusion_df.loc[e_dict[y_test[i_cnt]], e_dict[y_prediction_final[i_cnt, 0]]] += 1

    sns.heatmap(confusion_df, cmap="Blues", annot=True, linewidths=1.5)
    plt.yticks(va="center")
    plt.xticks(va="center")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.show()