# -*- coding: utf-8 -*-

import os
import mne
import numpy as np
import extract
from artifact_remove import artifact_remove
import dataset_arrangement as dta
from scipy import signal


class Epochs:
    def __init__(self, X, classe):
        if np.size(np.shape(X)) != 3:
            print('A matriz não possui a forma [N,T,E] requerida')

        else:
            self.data = X
            self.classe = classe
            self.filtered = dict()

    def add_epoch(self, new_data):
        self.data = np.append(self.data, new_data, axis=2)

    def filt(self, f_bank: dict, fs: int):
        for f_int in f_bank:
            sos = signal.iirfilter(N=6, Wn=f_bank[f_int], rs=20, btype='bandpass',
                                   output='sos', fs=fs, ftype='cheby2')
            self.filtered['{}-{}'.format(f_bank[f_int][0], f_bank[f_int][1])] = \
                signal.sosfilt(sos, self.data, axis=1)


class FBCSP:
    def __init__(self, epc1: Epochs, epc2: Epochs):
        self.W = dict()     # Para cada uma das bandas de frquencia
        self.epc1 = epc1    # dicionário de Epocas que passarão pelo FBCSP
        self.epc2 = epc2    # dicionário de Epocas que passarão pelo FBCSP

    def apply(self):
        for fb in self.epc1.filtered:
            # Calcula as matrizes de projeção para cada uma das bandas de frequencias
            self.W[fb] = extract.csp(self.epc1.filtered[fb], self.epc2.filtered[fb])


# =========================================================================================

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
def detect_classes(raw, events, e_dict, t_start, t_end, ica_start, ica_end, sfreq) -> dict:

    # Guarda a quantidade de canais e calcula o numero de amostra das epocas
    ch = raw.pick('eeg').info['nchan']
    n_samp = int((t_end - t_start) * sfreq + 1)

    # Pré aloca um dicionário que será utilizado como retorno da função
    X = dict()

    # Esse laço roda cada uma das trials dentro de um arquivo
    for n, i in enumerate(events[:, 0] / sfreq):

        # Salva a classe de movimento atual
        class_mov = [i for i in e_dict if e_dict[i] == events[n, 2]][0]

        # Coleta uma amostra de (ica_end - ica_start) segundos para análise
        raw_samp = raw.copy().pick('eeg').crop(tmin=i + ica_start, tmax=i + ica_end)

        # Realiza a remoção de artefatos
        raw_clean, flag = artifact_remove(raw_samp)

        # Salva a epoca como uma matriz
        new_epc = Epochs(
            raw_clean.crop(tmin=t_start, tmax=t_end).get_data().reshape(ch, n_samp, 1),
            classe=class_mov
        )

        # Adiciona o sinal atual em sua respectiva classe do dicionário X
        try:
            X[class_mov].add_epoch(new_epc.data)
        except KeyError:
            X[class_mov] = new_epc

    return X


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
