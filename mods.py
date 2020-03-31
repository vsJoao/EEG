# -*- coding: utf-8 -*-

import os
import mne
import numpy as np
from artifact_remove import artifact_remove
import dataset_arrangement as dta


# Função para carregar um arquivo de gravação e retornar seus respectivos eventos e objeto raw
def pick_file(f_loc: str, sbj: str, fnum: int):

    # define qual arquivo de eventos será carregado
    events_f_name = os.path.join(
        f_loc, sbj + '_{}_eve.fif'.format(fnum)
    )

    # define o nome do arquivo que deve ser carregado
    raw_f_name = os.path.join(
        f_loc, sbj + '_{}_raw.fif'.format(fnum)
    )

    # Esse ponto verifica se os arquivos já existem, senão cria eles
    try:
        # Carrega os eventos desse arquivo
        ev = mne.read_events(events_f_name)
        # Carrega o arquivo raw com os dados
        raw = mne.io.read_raw_fif(raw_f_name, preload=True)
    except FileNotFoundError:
        dta.sort()
        dta.sort_raw_fif()
        dta.sort_raw_eog_fif()
        raw, ev = pick_file(f_loc, sbj, fnum)

    return [raw, ev]


# Recebe um arquivo de gravação de eeg e seus eventos, e separa em matrizes tridimensionais de classes: [M, N, C]
# Sendo M o número de eletrodos, N o número de amostras, e C o número de classes no dicionário
def detect_classes(raw, events, e_dict, t_start, t_end, ica_start, ica_end, sfreq):

    # Guarda a quantidade de canais e calcula o numero de amostra das epocas
    ch = raw.pick('eeg').info['nchan']
    n_samp = int((t_end - t_start) * sfreq + 1)

    # Pré aloca um dicionário que será utilizado como retorno da função
    X = dict()
    for i in e_dict:
        X[i] = np.zeros([ch, n_samp, 1])

    # Esse laço roda cada uma das trials dentro de um arquivo
    for n, i in enumerate(events[:, 0] / sfreq):

        # Salva a classe de movimento atual
        class_mov = [i for i in e_dict if e_dict[i] == events[n, 2]][0]

        # Coleta uma amostra de (ica_end - ica_start) segundos para análise
        raw_samp = raw.copy().pick('eeg').crop(tmin=i + ica_start, tmax=i + ica_end)

        # Realiza a remoção de artefatos
        raw_clean, flag = artifact_remove(raw_samp)

        # Adiciona o sinal atual em sua respectiva classe do dicionário X
        X[class_mov] = \
            np.append(X[class_mov], raw_clean.crop(tmin=t_start, tmax=t_end).get_data().reshape(ch, n_samp, 1), axis=2)

    # Recorta os primeiros sinais de cada classe, pois foram pré-alocados com np.zeros
    X_return = dict()
    for i in X:
        X_return[i] = X[i][:, :, 1:]

    return X_return


def save_epoch(local, filename, file):
    try:
        try:
            np.save(os.path.join(local, filename), file)
        except IOError:
            os.makedirs(local)
            np.save(os.path.join(local, filename), file)
    except IOError:
        print('Não foi possível salvar {}'.format(filename))


def save_csp(local, filename, file):
    try:
        try:
            np.save(os.path.join(local, filename), file)
        except IOError:
            os.makedirs(local)
            np.save(os.path.join(local, filename), file)
    except IOError:
        print('Não foi possível salvar {}'.format(filename))
