# -*- coding: utf-8 -*-

import os
import mne
import numpy as np
import numpy.matlib
import extract
from artifact_remove import artifact_remove
import dataset_arrangement as dta
from scipy import signal

"""
O objeto Epochs possui o objetivo de guardar um conjunto de dados dentro da amostra
especificada. Dentro é possível guardar a epoca de apenas um sinal como também de vários
apenas um para o caso de realizar o processamento de testes
"""


class Epochs:
    def __init__(self, X: np.ndarray, fs, e_dict, f_bank, classe: str) -> None:
        # Epoca original de dados
        self.data = X
        # Dicionario onde cada chave é uma banda de frequencia do sinal
        self.filtered = dict()
        # Banco de filtros
        self.f_bank = f_bank
        # Classe do conjunto de epocas
        self.classe = classe
        # Classe do conjunto de epocas
        self.e_dict = e_dict
        # Numero referente a classe do movimento
        self.class_id = [i for i in e_dict if e_dict[i] == classe][0]
        # Taxa de amostragem do sinal
        self.fs = fs

        # bloco para verificar principalmente se há mais de uma matriz de epocas
        try:
            self.n_ch = self.data.shape[0]
            self.n_samp = self.data.shape[1]
            self.n_samp = self.data.shape[2]
        except IndexError:
            self.n_ch = self.data.shape[0]
            self.n_samp = self.data.shape[1]
            self.data = self.data.reshape(self.n_ch, self.n_samp, 1)

    # Adiciona uma epoca no conjunto original de dados
    def add_epoch(self, new_data: np.ndarray):
        self.data = np.append(self.data, new_data, axis=2)

    # Aplica o filtro em todos os sinais originais e guarda em self.filtered
    def filt(self):
        for f_int in self.f_bank:

            sos = signal.iirfilter(
                N=6, Wn=self.f_bank[f_int], rs=20, btype='bandpass',
                output='sos', fs=self.fs, ftype='cheby2')

            self.filtered['{}-{}'.format(self.f_bank[f_int][0], self.f_bank[f_int][1])] = \
                signal.sosfilt(sos, self.data, axis=1)


"""
O objeto FBCSP recebe duas epocas e calcula as matrizes de projeção
espacial para cada uma das bandas de frequencias especificadas no
conjunto de epocas.
"""


class FBCSP:
    def __init__(self, epc1: Epochs, epc2: Epochs, m: int):
        self.W = dict()                 # Para cada uma das bandas de frequencia
        self.epc1 = epc1                # Epocas de treino da primeira classe
        self.epc2 = epc2                # Epocas de treino da segunda classe
        self.m = m                      # Quantidade de vetores SCP utilizados

    # Função que calcula todas as matrizes de projeção
    def csp_calc(self):
        for fb in self.epc1.filtered:
            # Calcula as matrizes de projeção para cada uma das bandas de frequencias
            self.W[fb] = extract.csp(self.epc1.filtered[fb], self.epc2.filtered[fb])

    # A função a seguir pega as duas epocas de atributos e gera as caracteristicas
    def generate_train_features(self):
        # Retira as características do primeiro conjunto
        for i in range(self.epc1.data.shape[2]):
            try:
                f1 = np.append(f1, self.csp_feature(self.epc1, i), axis=1)
            except (np.AxisError, UnboundLocalError):
                f1 = self.csp_feature(self.epc1, i)

        for i in range(self.epc2.data.shape[2]):
            try:
                f2 = np.append(f2, self.csp_feature(self.epc2, i), axis=1)
            except (np.AxisError, UnboundLocalError):
                f2 = self.csp_feature(self.epc2, i)

        f1 = f1.transpose()
        f2 = f2.transpose()

        f = np.append(
            np.append(f1, np.matlib.repmat(self.epc1.class_id, f1.shape[0], 1), axis=1),
            np.append(f2, np.matlib.repmat(self.epc2.class_id, f2.shape[0], 1), axis=1),
            axis=0
        )

        return f

    # Extrai um vetor de características de um sinal recebido como parametro
    # Utilizando os dados do objeto fbcsp
    def csp_feature(self, X: Epochs, epoch_idx: int = 0) -> np.ndarray:

        # Gera os indices dos m primeiras e m ultimas linhas da matriz
        m_int = np.hstack((np.arange(0, self.m), np.arange(-self.m, 0)))

        # Pré-aloca um vetor de caracteristicas
        f = np.zeros([(self.m * 2) * len(self.W), 1])

        for n, f_band in enumerate(self.W):
            # Calcula-se a decomposição por CSP do sinal na banda de freq e seleciona as linhas [m_int]
            Z = np.dot(self.W[f_band], X.filtered[f_band][:, :, epoch_idx])[m_int, :]

            # Calcula-se a variancia dessas linhas e em seguida o seu somatório
            var_z = np.var(Z, axis=1)
            var_sum = np.sum(var_z)

            # Constrói-se o vetor de características desse sinal
            f[n * (2 * self.m):(2 * self.m) * (n + 1)] = np.log(var_z / var_sum).reshape(self.m*2, 1)

        return f


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
def detect_classes(raw, events, e_dict, t_start, t_end, ica_start, ica_end, sfreq, fb_freqs) -> dict:

    # Guarda a quantidade de canais e calcula o numero de amostra das epocas
    ch = raw.pick('eeg').info['nchan']
    n_samp = int((t_end - t_start) * sfreq + 1)

    # Pré aloca um dicionário que será utilizado como retorno da função
    X = dict()

    # Esse laço roda cada uma das trials dentro de um arquivo
    for n, i in enumerate(events[:, 0] / sfreq):

        # Salva a classe de movimento atual
        class_mov = e_dict[events[n, 2]]

        # Coleta uma amostra de (ica_end - ica_start) segundos para análise
        raw_samp = raw.copy().pick('eeg').crop(tmin=i + ica_start, tmax=i + ica_end)

        # Realiza a remoção de artefatos
        raw_clean, flag = artifact_remove(raw_samp)

        # Salva a epoca
        new_epc = \
            raw_clean.crop(tmin=t_start, tmax=t_end).get_data().reshape(ch, n_samp, 1)

        # Adiciona o sinal atual em sua respectiva classe do dicionário X
        try:
            X[class_mov].add_epoch(new_epc)
        except KeyError:
            X[class_mov] = Epochs(
                X=new_epc,
                classe=class_mov,
                f_bank=fb_freqs,
                e_dict=e_dict,
                fs=sfreq,
            )

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
