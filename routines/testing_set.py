# Configs utilizadas
from configs.timing_config import *
from configs.database_names import *

import utils

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from itertools import combinations, product
from scipy.stats import mode
from sklearn import svm

# import dataset_arrangement as dta
# import time

plt.close('all')
sns.set(style="ticks")


# %%=================== Processamento do conj de Teste ==========================

for s_id, sbj_name in enumerate(f_names_test):

    epoch_filepath = os.path.join(epoch_test_loc, f'{sbj_name}_epoch.npy')
    features_test_filepath = os.path.join(features_test_loc, f'{sbj_name}_features.npy')

    # TODO: Corrigir a forma como pegar o conjunto de treinos dentro da área de teste
    csp_filepath = os.path.join(csp_loc, f'{f_names_train[s_id]}_Wcsp.npy')
    features_train_filepath = os.path.join(features_train_loc, f'{f_names_train[s_id]}_features.npy')

    if os.path.exists(epoch_filepath):
        X = np.load(epoch_filepath, allow_pickle=True).item()

    else:
        X = dict()

        for sbj_idx in range(n_runs):
            # Carrega o arquivo raw e o conjunto de eventos referentes a ele
            raw, eve = utils.pick_file(raw_fif_loc, sbj_name, sbj_idx + 1)

            # Separa o arquivo em epocas e aplica o ica
            x_temp = utils.epoch_raw_data(
                raw, eve, e_dict, t_start, t_end, ica_start, ica_end
            )

            # Tenta adicionar a epoca atual ao dicionário, se não conseguir, reinicia o dicionário
            for i in e_classes:
                if sbj_idx == 0:
                    X[i] = x_temp[i]
                else:
                    X[i].add_epoch(x_temp[i].data)

        utils.save_epoch(epoch_filepath, X)
        del x_temp, raw, eve

    Wfb = np.load(csp_filepath, allow_pickle=True).item()

    # Verifica se já existe um arquivo de caracteristicas de teste
    if os.path.exists(features_test_filepath):
        f = np.load(features_test_filepath, allow_pickle=True).item()

    # Se não existir, cria
    else:
        f = dict()

        first = True
        for k, (i, j) in product(X, combinations(e_classes, 2)):
            # k - Classes do conjunto de dados X
            # i, j - Todas as combinações de CSP possíveis a partir das classes em e_dict
            if k not in e_classes:
                continue

            # Laço Passando por todos os sinais de um conjunto de matrizes
            for n in range(X[k].n_trials):

                # Cálculo dos vetores de características utilizando a corrente classe de W e de X
                f_temp = np.append(
                    Wfb[f'{i}{j}'].csp_feature(X[k].data[:, :, n]).transpose(),
                    [[k_id for k_id in e_dict if e_dict[k_id] == k]], axis=1
                )

                # Tenta adicionar esse vetor de características na matriz de caracteristicas
                try:
                    f[f'{i}{j}'] = np.append(f[f'{i}{j}'], f_temp, axis=0)
                except KeyError:
                    f[f'{i}{j}'] = f_temp

        utils.save_csp(features_test_filepath, f)
        del f_temp, first

    # Salva as matrizes de características em cada um dos arquivos dos sujeitos
    f_train = np.load(features_train_filepath, allow_pickle=True).item()

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

    print(sbj_name, res.mean())

    confusion_df = pd.DataFrame(
        np.zeros([len(e_classes), len(e_classes)]),
        index=e_classes, columns=e_classes
    )

    for i_cnt, i in enumerate(y_prediction_final):
        confusion_df.loc[e_dict[y_test[i_cnt]], e_dict[y_prediction_final[i_cnt, 0]]] += 1

    plt.figure(s_id)
    ax = sns.heatmap(confusion_df, cmap="Blues", annot=True, linewidths=1.5)
    plt.yticks(va="center")
    plt.xticks(va="center")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")

plt.show()
print('fim')
