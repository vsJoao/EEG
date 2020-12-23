# Configs utilizadas
from configs.timing_config import *
from configs.database_names import *
from configs.classification_consts import *

from classes import Epochs
from classes import FBCSP

import utils
import numpy as np

import os
from itertools import combinations


# %%=============== Processamento do conj de Treino ==========================

# O primeiro laço percorre os arquivos de cada um dos sujeitos
def training_data_routine():
    for sbj_name in f_names_train:

        epoch_filepath = os.path.join(epoch_train_loc, f'{sbj_name}_epoch.npy')
        csp_filepath = os.path.join(csp_folder, f'{sbj_name}_Wcsp.npy')
        features_filepath = os.path.join(features_train_folder, f'{sbj_name}_features.npy')

        if os.path.exists(epoch_filepath):
            X = Epochs.dict_from_filepath(epoch_filepath)

        else:
            X = dict()

            for sbj_idx in range(n_runs):

                # Carrega o arquivo raw e o conjunto de eventos referentes a ele
                raw, eve = utils.pick_file(raw_fif_folder, sbj_name, sbj_idx + 1)
                # Do arquivo raw, retorna as epocas, após ter sido passado pelo processo de ICA
                x_temp = utils.epoch_raw_data(
                    raw, eve, e_dict, t_start, t_end, ica_start, ica_end)

                # Salva os dados epocados no dicionário de épocas X
                for i in e_classes:
                    if sbj_idx == 0:
                        X[i] = x_temp[i]
                    else:
                        X[i].add_epoch(x_temp[i].data)

            utils.save_epoch(epoch_filepath, X)
            del x_temp, raw, eve

        # %% ============== Calculo das matrizes de projeção Espacial =====================

        if os.path.exists(csp_filepath):
            W = np.load(csp_filepath, allow_pickle=True).item()

        else:
            # Calcula-se a matriz se projeção espacial de um único sujeito
            print('Calculo das matrizes de projeção Espacial do sujeito {}'.format(sbj_name))
            W = dict()

            for i, j in combinations(e_classes, 2):
                W[f'{i}{j}'] = FBCSP(X[i], X[j], m=m, filterbank=fb_freqs)

            utils.save_csp(csp_filepath, W)

        # ====================== Construção do vetor de caracteristicas ==================

        if not os.path.exists(features_filepath):
            # Se os arquivos não existirem, então calcule-os e salve em seguida
            print('Criando os vetores de caracteristicas do sujeito {}'.format(sbj_name))
            features = dict()

            # Realiza o CSP e extrai as caracteristicas entre todas as classes possíveis
            for i, j in combinations(e_classes, 2):
                # Executa a extração das características em todas as combinações de classes
                features[f'{i}{j}'] = W[f'{i}{j}'].generate_train_features(
                    X[i], X[j], dict(zip(e_dict.values(), e_dict.keys()))
                )

            utils.save_csp(features_filepath, features)