# -*- coding: utf-8 -*-

# Configs utilizadas
from configs.timing_config import *
from configs.database_names import *
from configs.classification_consts import *

from classes import Epochs
from classes import FBCSP

import utils

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, product
from scipy.stats import mode
from sklearn import svm
from time import time

# import dataset_arrangement as dta
# import time

plt.close('all')

# %%===================== Testar ou treinar ============================

# ans = input('(1) para conj treino e (2) para conj teste: ')
ans = '0'

# %%=============== Processamento do conj de Treino ==========================

t = time()
# O primeiro laço percorre os arquivos de cada um dos sujeitos
for sbj_name in f_names_train:

    if ans == '2':
        break

    try:
        X = Epochs.dict_from_filepath(f'{epoch_train_loc}/{sbj_name}_epoch.npy')

    except (IOError, AssertionError, KeyError):
        X = dict()
        for sbj_idx in range(n_runs):

            # Carrega o arquivo raw e o conjunto de eventos referentes a ele
            raw, eve = utils.pick_file(raw_fif_loc, sbj_name, sbj_idx + 1)
            # Do arquivo raw, retorna as epocas, após ter sido passado pelo processo de ICA
            x_temp = utils.epoch_raw_data(
                raw, eve, e_dict, t_start, t_end, ica_start, ica_end)

            # Tenta adicionar a epoca atual ao dicionário, se não conseguir, reinicia o dicionário
            try:
                for i in e_classes:
                    X[i].add_epoch(x_temp[i].data)
            except KeyError:
                for i in e_classes:
                    X[i] = x_temp[i]

        utils.save_epoch(epoch_train_loc, f'{sbj_name}_epoch.npy', X)

    # %% ============== Calculo das matrizes de projeção Espacial =====================

    # Tenta carregar as matrizes de projeção espacial, senão calcula-as e salva no arquivo
    try:
        W = np.load('dataset_files/csp/{}_Wcsp.npy'.format(sbj_name), allow_pickle=True).item()
    except IOError:
        # Calcula-se a matriz se projeção espacial de um único sujeito
        print('Calculo das matrizes de projeção Espacial do sujeito {}'.format(sbj_name))
        W = dict()

        for i, j in combinations(e_classes, 2):
            W[f'{i}{j}'] = FBCSP(X[i], X[j], m=m, filterbank=fb_freqs)

        utils.save_csp('dataset_files/csp', '{}_Wcsp.npy'.format(sbj_name), W)

    # ====================== Construção do vetor de caracteristicas ==================
    # Tenta carregar diretamente o arquivo com os vetores
    try:
        features = np.load('dataset_files/features/{}_features.npy'.format(sbj_name), allow_pickle=True).item()
    # Se os arquivos não existirem, então calcule-os e salve em seguida
    except IOError:
        print('Criando os vetores de caracteristicas do sujeito {}'.format(sbj_name))
        features = dict()

        # Realiza o CSP entre todas as classes possíveis
        for i, j in combinations(e_classes, 2):
            # Executa a extração das características em todas as combinações de classes
            features[f'{i}{j}'] = W[f'{i}{j}'].generate_train_features(
                X[i], X[j], dict(zip(e_dict.values(), e_dict.keys()))
            )

        utils.save_csp('dataset_files/features', '{}_features.npy'.format(sbj_name), features)


# %%=================== Processamento do conj de Teste ==========================

for s_id, sbj_name in enumerate(f_names_test):
    if ans == '1':
        break

    try:
        X = np.load('dataset_files/epoch_test/{}_epoch.npy'.format(sbj_name), allow_pickle=True).item()
    except FileNotFoundError:
        X = dict()
        for sbj_idx in range(n_runs):
            # Carrega o arquivo raw e o conjunto de eventos referentes a ele
            raw, eve = utils.pick_file(raw_fif_loc, sbj_name, sbj_idx + 1)

            # Separa o arquivo em epocas e aplica o ica
            x_temp = utils.epoch_raw_data(
                raw, eve, e_dict, t_start, t_end, ica_start, ica_end
            )

            # Tenta adicionar a epoca atual ao dicionário, se não conseguir, reinicia o dicionário
            try:
                for i in e_classes:
                    X[i].add_epoch(x_temp[i].data)
            except KeyError:
                for i in e_classes:
                    X[i] = x_temp[i]

            utils.save_epoch('dataset_files/epoch_test', '{}_epoch.npy'.format(sbj_name), X)

    Wfb = np.load('dataset_files/csp/{}_Wcsp.npy'.format(f_names_train[s_id]), allow_pickle=True).item()

    # Tenta carregar os arquivos de características de cada sujeito
    try:
        f = np.load('dataset_files/features_test/{}_features.npy'.format(sbj_name), allow_pickle=True).item()

    # Se não conseguir, irá gerar esse arquivo
    except FileNotFoundError:
        f = dict()

        """
        Equivalente á:
        # Dois primeiros laços passando pelas combinações de CSP
        for i, j in combinations(e_classes, 2):
            # Laço passando por todas as classes em um conjunto de dados
            for k in X:
        """
        for k, (i, j) in product(X, combinations(e_classes, 2)):
            # k - Classes do conjunto de dados X
            # i, j - Todas as combinações de CSP possíveis a partir das classes em e_dict
            if k not in e_classes:
                continue

            # Laço Passando por todos os sinais de um conjunto de matrizes
            for n in range(X[k].data.shape[2]):

                # Cálculo dos vetores de características utilizando a corrente classe de W e de X
                f_temp = np.append(
                    Wfb[f'{i}{j}'].csp_feature(X[k].data[:, :, n]).transpose(),
                    [[k_id for k_id in e_dict if e_dict[k_id] == k]], axis=1
                )

                # Tenta adicionar esse vetor de características na matriz de caracteristicas
                try:
                    f[f'{i}{j}'] = np.append(f[f'{i}{j}'], f_temp, axis=0)
                except (NameError, ValueError, KeyError):
                    f[f'{i}{j}'] = f_temp

        utils.save_csp('dataset_files/features_test', '{}_features.npy'.format(sbj_name), f)

    # Salva as matrizes de características em cada um dos arquivos dos sujeitos
    f_train = np.load('dataset_files/features/{}_features.npy'.format(f_names_train[s_id]), allow_pickle=True).item()

    for i, j in combinations(e_classes, 2):
        x_train = f_train[f'{i}{j}'][:, :-1]
        y_train = f_train[f'{i}{j}'][:, -1]

        x_test = f[f'{i}{j}'][:, :-1]
        y_test = f[f'{i}{j}'][:, -1]

        svm_model = svm.SVC()
        svm_model.fit(x_train, y_train)

        try:
            y_prediction = np.append(y_prediction, np.array([svm_model.predict(x_test)]).T, axis=1)
        except (IndexError, ValueError, NameError):
            y_prediction = np.array([svm_model.predict(x_test)]).T

    y_prediction_final = mode(y_prediction, axis=1).mode
    res = np.array([y_prediction_final == y_test.reshape(-1, 1)])

    print(sbj_name, res.mean())

    del y_prediction

print('fim')
print(time() - t)
