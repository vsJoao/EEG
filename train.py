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
# e_dict = {1: 'l', 2: 'r', 3: 'f', 4: 't'}

# Dicionário mostrando as classes. As chaves do dicionário devem ser
# obrigatóriamente os ids dos eventos definidos nos arquivos
e_dict = {1: 'l', 2: 'r', 3: 'f', 4: 't'}

# Marca a quais classes estão relacionados cada um dos grupos
e_groups = {'a': 'lr', 'b': 'ft'}   # NÃO MEXER NESTA PORRA SE VIRA COM ELA DESSE JEITO

# Dicionário de grupos
group_dict = {1: 'a', 2: 'b'}

# Dicionário inverso dos grupos
group_dict2 = dict(zip(group_dict.values(), group_dict.keys()))

# Dicionário inverso de classes
e_dict2 = dict(zip(e_dict.values(), e_dict.keys()))


def identify_group_id_by_class_id(class_id) -> int:
    clas = e_dict[class_id]
    for gp in e_groups:
        if clas in e_groups[gp]:
            return group_dict2[gp]

    return -1


# Os arquivos 'raw' são os sinais originais provisionados pelo dataset, mantendo
# todas as informações iniciais (trabalha em conjunto com os arquivos de eventos)
raw_loc = "raw_data"

raw_eog_fif_loc = "raw_eog_fif_files"
raw_fif_loc = "raw_fif_files"

# %% =================== Variáveis de tempo que serão utilizadas======================
# Frequencia de amostragem
sfreq = 250

# Tempo de duração a partir do evento (tempo da trial)
t_trial = 7.5

# Instante inicial do intervalo de interesse em segundos
t_start = 3.5

# Instante final do intervalo de interesse em segundos
t_end = 6

# Instante inicial de aplicação da ICA
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
                for i in e_dict:
                    X[e_dict[i]].add_epoch(x_temp[e_dict[i]].data)
            except KeyError:
                for i in e_dict:
                    X[e_dict[i]] = x_temp[e_dict[i]]

        # Salva os dados epocados de cada um dos sujeitos
        mods.save_epoch('epoch_train', '{}_epoch.npy'.format(sbj_name), X)

    # %% ============== Calculo das matrizes de projeção Espacial =====================

    # Tenta carregar as matrizes de projeção espacial, senão calcula-as e salva no arquivo
    try:
        W = np.load('csp/{}_Wcsp.npy'.format(sbj_name), allow_pickle=True).item()
    except IOError:
        # Calcula-se a matriz se projeção espacial de um único sujeito
        print('Calculo das matrizes de projeção Espacial do sujeito {}'.format(sbj_name))
        W = dict()

        epc_group = dict()
        # Passa pelos conjuntos de classes
        for i in e_groups:
            # Passa por cada uma das classes desse conjunto
            for k in e_groups[i]:

                # Filtra de acordo com o banco de filtros especificado
                X[k].filt()

                # Junta todas as classes que estiverem em um mesmo conjunto
                try:
                    epc_group[i] = \
                        epc_group[i].concat_epoch(X[k], new_edict=group_dict, new_class=i)
                except KeyError:
                    epc_group[i] = X[k]

                # Aplica os filtros no novo sinal
                epc_group[i].filt()

            # Gera as matrizes de decomposição para cada uma das classes dentro de um grupo
            for c1_cnt, c1 in enumerate(e_groups[i][:-1]):
                for c2 in e_groups[i][c1_cnt + 1:]:
                    W['{}{}'.format(c1, c2)] = mods.FBCSP(X[c1], X[c2], m=m)
                    W['{}{}'.format(c1, c2)].csp_calc()

        # Gera as matrizes de decomposição espacial para cada grupo
        for c1_cnt, c1 in enumerate(list(e_groups)[:-1]):
            for c2 in list(e_groups)[c1_cnt+1:]:
                W['{}{}'.format(c1, c2)] = mods.FBCSP(epc_group[c1], epc_group[c2], m=m)
                W['{}{}'.format(c1, c2)].csp_calc()

        mods.save_csp('csp', '{}_Wcsp.npy'.format(sbj_name), W)

    # ====================== Construção do vetor de caracteristicas ==================
    # Tenta carregar diretamente o arquivo com os vetores
    try:
        features = np.load('features/{}_features.npy'.format(sbj_name), allow_pickle=True).item()
    # Se os arquivos não existirem, então calcule-os e salve em seguida
    except IOError:
        print('Criando os vetores de caracteristicas do sujeito {}'.format(sbj_name))

        # Cria as matrizes de características do sujeito
        features = dict()
        for classes in W:
            features[classes] = W[classes].generate_train_features()

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
                for i in e_dict:
                    X[e_dict[i]].add_epoch(x_temp[e_dict[i]].data)
            except KeyError:
                for i in e_dict:
                    X[e_dict[i]] = x_temp[e_dict[i]]

        # Salva os dados epocados de cada um dos sujeitos
        mods.save_epoch('epoch_test', '{}_epoch.npy'.format(sbj_name), X)

    # Atenção para esta parte, é **muito** mais rápido filtrar os sinais previamente, mas isso
    # não será possível em um sistema online
    print('Filtrando Sinais...')
    for i in X:
        # Filtra os sinais de acordo com o banco de filtros especificado
        X[i].filt()

    # Carrega as matrizes de projeção espacial
    Wfb = np.load('csp/{}_Wcsp.npy'.format(f_names_train[s_id]), allow_pickle=True).item()

    # Tenta carregar os arquivos de características (treino) de cada sujeito
    try:
        f_test = np.load('features_test/{}_features.npy'.format(sbj_name), allow_pickle=True).item()

    # Se não conseguir, irá gerar esse arquivo e salvá-lo
    except FileNotFoundError:
        f_test = dict()
        print('Extraindo características de {}'.format(sbj_name))

        # Laço passando pelas classes de separação definidas em W -> (ab, lr, ft)
        for i in Wfb:

            # Laço passando por todas as classes em um conjunto de dados
            for k_cnt, k in enumerate(X):
                print('Classe: {} -> {}'.format(i, k))

                # Laço Passando por todos os sinais 'E' de um conjunto de dados X[N,T,E]
                for n in range(X[k].data.shape[2]):
                    print('.', end="")

                    # Cálculo dos vetores de características utilizando a corrente classe de W e de X
                    f_temp = \
                        np.append(Wfb[i].csp_feature(X[k], n).transpose(), [[e_dict2[k]]], axis=1)

                    # Tenta adicionar esse vetor de características na matriz de caracteristicas
                    try:
                        f_test[i] = np.append(f_test[i], f_temp, axis=0)
                    except (NameError, ValueError, KeyError):
                        f_test[i] = f_temp

                print(' ')

        # Salva as matrizes de características em cada um dos arquivos dos sujeitos
        mods.save_csp('features_test', '{}_features.npy'.format(sbj_name), f_test)

    # Carrega os vetores de características do conjunto de treino
    f_train = np.load('features/{}_features.npy'.format(f_names_train[s_id]), allow_pickle=True).item()

    # Variáveis para contar a média total
    mediatotal = dict()
    contador = dict()

    # variaveis para guardar os conjuntos de teste e treino
    x_train = dict()
    y_train = dict()

    x_test = dict()
    y_test = dict()

    # Dicionário para guardar cada um dos modelos de classificação (no caso 3)
    KNN_model = dict()

    for i in f_train:
        # Monta-se um classificador para cada possibilidade de classificação
        x_train[i] = f_train[i][:, :-1]
        y_train[i] = f_train[i][:, -1]

        x_test[i] = f_test[i][:, :-1]
        y_test[i] = f_test[i][:, -1]

    for n_neighbors in range(1, 21):

        # Testa todos os 288 vetores do conjunto de teste
        prediction_final = list()
        for test in range(288):

            # Classifica o sinal quanto ao grupo a que pertence
            prediction_possibilities = list()
            for i_cnt, i in enumerate(list(e_groups)[:-1]):
                for j in list(e_groups)[i_cnt+1:]:
                    cl_n = '{}{}'.format(i, j)

                    KNN_model[cl_n] = knn.KNeighborsClassifier(n_neighbors=n_neighbors)
                    KNN_model[cl_n].fit(x_train[cl_n], y_train[cl_n])

                    prediction_possibilities.append(KNN_model[cl_n].predict(x_test[cl_n][test, :].reshape(1, 24)))

            prediction_group = mode(prediction_possibilities).mode[0]

            if prediction_group == 1:
                cl_n = 'lr'

                KNN_model[cl_n] = knn.KNeighborsClassifier(n_neighbors=n_neighbors)
                KNN_model[cl_n].fit(x_train[cl_n], y_train[cl_n])

                prediction = KNN_model[cl_n].predict(x_test[cl_n][test, :].reshape(1, 24))

            elif prediction_group == 2:
                cl_n = 'ft'

                KNN_model[cl_n] = knn.KNeighborsClassifier(n_neighbors=n_neighbors)
                KNN_model[cl_n].fit(x_train[cl_n], y_train[cl_n])

                prediction = KNN_model[cl_n].predict(x_test[cl_n][test, :].reshape(1, 24))

            prediction_final.append(prediction)

        prediction_final = np.array(prediction_final)
        res = (prediction_final == y_test['lr'].reshape(288, 1))

        print(sbj_name, n_neighbors, res.mean())

print('fim')