# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import sklearn.neighbors as knn
import numpy as np
import numpy.matlib

import extract
import mods
import os
import mne
from artifact_remove import artifact_remove
from scipy.stats import mode

# import dataset_arrangement as dta
# import time


# %% ==================== Definição das variáveis básicas============================
plt.close('all')

# Nomes que referenciam os sujeitos e seus respectivos conjuntos de teste/treino
f_names_train: list = ["A01T", "A02T", "A03T", "A05T", "A06T", "A07T", "A08T", "A09T"]
f_names_test: list = ["A01E", "A02E", "A03E", "A05E", "A06E", "A07E", "A08E", "A09E"]

# Faz a definição dos eventos (classes de movimentos) com um dicionário.
# Neste caso o dicionário é feito para as seguintes classes:
# l - mão esquerda
# r - mão direita
# f - pés
# t - lígua
# a - movimento das maos
# b - movimento pés ou lingua
e_dict = {1: 'l', 2: 'r', 3: 'f', 4: 't'}
e_keys = []
for i in e_dict:
    if e_dict[i] not in e_keys:
        e_keys.append(e_dict[i])

# Os arquivos 'sorted' são apenas um conjunto de matrizes com as epocas já separadas
# por classe, sendo um arquivo desses por sujeito
sort_loc = "sorted_data"

# Os arquivos 'raw' são os sinais originais provisionados pelo dataset, mantendo
# todas as informações iniciais (trabalha em conjunto com os arquivos de eventos)
raw_loc = "raw_data"

raw_eog_fif_loc = "raw_eog_fif_files"
raw_fif_loc = "raw_fif_files"

# %% =================== Variáveis de tempo que serão utilizadas======================
# Frequencia de amostragem
sfreq = 250

# Tempo de interesse (tempo da trial)
t_trial = 7.5

# Instante inicial do intervalo de interesse em segundos
t_start = 3.5

# Instante final do intervalo de interesse em segundos
t_end = 6

# Instante iniciaç de aplicação da ICA
ica_start = 0

# Instante final de aplicação da ICA
ica_end = 7

# Tempo entre o iníncio e final das epocas
t_epoch = t_end - t_start

# Numero de samples de cada sinal
n_samples = int(t_epoch * sfreq + 1)

# Quantidade de vetores considerados para a extração de características
m = 2

# Numeros de arquivos por pessoa
n_runs = 6

# Intervalos de Frequencia do banco de filtros
fb_freqs = {
    1: [8, 12],
    2: [12, 16],
    3: [16, 20],
    4: [20, 24],
    5: [24, 28],
    6: [28, 32]
}

# %%===================== Testar ou treinar ============================

# ans = input('(1) para conj treino e (2) para conj teste: ')
ans = '2'

# %%=============== Processamento do conj de Treino ==========================

# O primeiro laço percorre os arquivos de cada um dos sujeitos
for sbj_name in f_names_train:

    if ans == '2':
        break

    try:
        X = np.load('epoch_train/{}_epoch.npy'.format(sbj_name), allow_pickle=True).item()

    except IOError:
        X = dict()
        for sbj_idx in range(n_runs):

            # Carrega o arquivo raw e o conjunto de eventos referentes a ele
            raw, eve = mods.pick_file(raw_eog_fif_loc, sbj_name, sbj_idx + 1)
            # Do arquivo raw, retorna as epocas, após ter sido passado pelo processo de ICA
            x_temp = mods.detect_classes(
                raw, eve, e_dict, t_start, t_end, ica_start, ica_end, sfreq, fb_freqs)

            # Tenta adicionar a epoca atual ao dicionário, se não conseguir, reinicia o dicionário
            try:
                for i in e_keys:
                    X[i].add_epoch(x_temp[i].data)
            except KeyError:
                for i in e_keys:
                    X[i] = x_temp[i]

        # Filtra e salva os dados epocados de cada um dos sujeitos
        for i in X:
            X[i].filt()

        mods.save_epoch('epoch_train', '{}_epoch.npy'.format(sbj_name), X)

    # %% ============== Calculo das matrizes de projeção Espacial =====================

    # Tenta carregar as matrizes de projeção espacial, senão calcula-as e salva no arquivo
    try:
        W = np.load('csp/{}_Wcsp.npy'.format(sbj_name), allow_pickle=True).item()
    except IOError:
        # Calcula-se a matriz se projeção espacial de um único sujeito
        print('Calculo das matrizes de projeção Espacial do sujeito {}'.format(sbj_name))
        W = dict()

        for i_cnt, i in enumerate(e_keys[:-1]):
            for j_cnt, j in enumerate(e_keys[i_cnt+1:]):
                W['{}{}'.format(i, j)] = mods.FBCSP(X[i], X[j], m=m)
                W['{}{}'.format(i, j)].csp_calc()

        mods.save_csp('csp', '{}_Wcsp.npy'.format(sbj_name), W)

    # ====================== Construção do vetor de caracteristicas ==================
    # Tenta carregar diretamente o arquivo com os vetores
    try:
        features = np.load('features/{}_features.npy'.format(sbj_name), allow_pickle=True).item()
    # Se os arquivos não existirem, então calcule-os e salve em seguida
    except IOError:
        print('Criando os vetores de caracteristicas do sujeito {}'.format(sbj_name))
        features = dict()

        # Realiza o CSP entre todas as classes possíveis
        for i_cnt, i in enumerate(e_keys[:-1]):
            # Passa pelas classes contando a partir da classe do laço acima
            for j_cnt, j in enumerate(e_keys[i_cnt + 1:]):
                # Executa a extração das características em todas as combinações de classes
                features['{}{}'.format(i, j)] = W['{}{}'.format(i, j)].generate_train_features()

        mods.save_csp('features', '{}_features.npy'.format(sbj_name), features)


# %%=================== Processamento do conj de Teste ==========================

for s_id, sbj_name in enumerate(f_names_test):
    if ans == '1':
        break

    try:
        X = np.load('epoch_test/{}_epoch.npy'.format(sbj_name), allow_pickle=True).item()
    except FileNotFoundError:
        X = dict()
        for sbj_idx in range(n_runs):
            # Carrega o arquivo raw e o conjunto de eventos referentes a ele
            raw, eve = mods.pick_file(raw_eog_fif_loc, sbj_name, sbj_idx + 1)

            # Separa o arquivo em epocas e aplica o ica
            x_temp = mods.detect_classes(
                raw, eve, e_dict, t_start, t_end, ica_start, ica_end, sfreq, fb_freqs)

            # Tenta adicionar a epoca atual ao dicionário, se não conseguir, reinicia o dicionário
            try:
                for i in e_keys:
                    X[i].add_epoch(x_temp[i].data)
            except KeyError:
                for i in e_keys:
                    X[i] = x_temp[i]

            # Filtra e salva os dados epocados de cada um dos sujeitos
            for i in X:
                X[i].filt()
            mods.save_epoch('epoch_test', '{}_epoch.npy'.format(sbj_name), X)

    Wfb = np.load('csp/{}_Wcsp.npy'.format(f_names_train[s_id]), allow_pickle=True).item()

    # Tenta carregar os arquivos de características de cada sujeito
    try:
        f = np.load('features_test/{}_features.npy'.format(sbj_name), allow_pickle=True).item()

    # Se não conseguir, irá gerar esse arquivo
    except FileNotFoundError:
        f = dict()

        # Dois primeiros laços passando pelas combinações de CSP
        for i_cnt, i in enumerate(e_keys[:-1]):
            for j_cnt, j in enumerate(e_keys[i_cnt + 1:]):

                # Laço passando por todas as classes em um conjunto de dados
                for k_cnt, k in enumerate(X):

                    # Laço Passando por todos os sinais de um conjunto de matrizes
                    for n in range(X[k].data.shape[2]):

                        # Cálculo dos vetores de características utilizando a corrente classe de W e de X
                        f_temp = \
                            np.append(Wfb['{}{}'.format(i, j)].csp_feature(X[k], n).transpose(),
                                      [[k_id for k_id in e_dict if e_dict[k_id] == k]], axis=1)

                        # Tenta adicionar esse vetor de características na matriz de caracteristicas
                        try:
                            f['{}{}'.format(i, j)] = np.append(f['{}{}'.format(i, j)], f_temp, axis=0)
                        except (NameError, ValueError, KeyError):
                            f['{}{}'.format(i, j)] = f_temp

        mods.save_csp('features_test', '{}_features.npy'.format(sbj_name), f)

    # Salva as matrizes de características em cada um dos arquivos dos sujeitos
    f_train = np.load('features/{}_features.npy'.format(f_names_train[s_id]), allow_pickle=True).item()

    # Variáveis para contar a média total
    mediatotal = 0
    contador = 0

    for n_knn in range(20):

        for i_cnt, i in enumerate(e_keys[:-1]):
            for j_cnt, j in enumerate(e_keys[i_cnt + 1:]):

                x_train = f_train['{}{}'.format(i, j)][:, :-1]
                y_train = f_train['{}{}'.format(i, j)][:, -1]

                x_test = f['{}{}'.format(i, j)][:, :-1]
                y_test = f['{}{}'.format(i, j)][:, -1]

                KNN_model = knn.KNeighborsClassifier(n_neighbors=n_knn+1)
                KNN_model.fit(x_train, y_train)

                try:
                    y_prediction = np.append(y_prediction, np.array([KNN_model.predict(x_test)]).T, axis=1)
                except (IndexError, ValueError, NameError):
                    y_prediction = np.array([KNN_model.predict(x_test)]).T

        y_prediction_final = mode(y_prediction, axis=1).mode
        res = np.array([y_prediction_final == y_test.reshape(288, 1)])

        print(sbj_name, n_knn+1, res.mean())

        del y_prediction

print('fim')
