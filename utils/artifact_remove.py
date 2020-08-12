# -*- coding: utf-8 -*-

from mne.preprocessing import ICA
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

# %% 

# Realiza a remoção de artefatos a partir de um sinal raw[M,N] sendo:
# M - O número de amostras do sinal
# N - O numero de eletrodos do sinal de EEG


def artifact_remove(
        raw, n_comp=15, print_all=False, print_psd=False, print_res=False, print_overlay=False, print_ICA=False):
    """ Realiza a decomposição do sinal e remove os artefatos

    Parameters
    ----------
    raw: mne.RawArray
        O sinal que será analisado para realizar a limpeza
    n_comp: int
        A quantidade máxima de componentes independentes que será gerada na análise do sinal
    print_all: bool
        Realiza a impressão de todos os gráficos intermediários
    print_psd: bool
        Realiza a impressão dos espectogramas das componentes independentes
    print_res: bool
        Imprime uma mensagem informando se havia componente a ser limpo
    print_overlay: bool
        Imprime um grafico com os sinais sobrepostos mostrando o resultado da limpeza
    print_ICA: bool
        Imprime os gráficos das componentes intependentes

    """

    if n_comp > raw.pick('eeg').info['nchan']:
        n_comp = raw.pick('eeg').info['nchan']

    if print_all is True:
        print_psd = True
        print_res = True
        print_overlay = True
        print_ICA = True

    # Aplicação do ICA e sepação das componentes
    
    raw_filt = raw.copy()       # Cria-se as cópias do sinal que serão utilizadas
    raw_reconst = raw.copy()
    
    raw_filt.filter(l_freq=1, h_freq=None)     # aplica-se um filtro
    
    # Criação do objeto de ICA
    ica = ICA(n_components=n_comp, method='fastica')
    
    # Calculo da matriz de mistura
    ica.fit(raw_filt)
    
    # Imprime as componentes estimadas
    if print_ICA is True:
        ica.plot_sources(raw_filt)
    
    # %% Estimação dos Espectogramas
    
    # Cria-se uma matriz com os sinais das componentes independentes
    X = ica.get_sources(raw_filt).get_data()
    
    # É estimado o psd das componentes independentes
    freqs, psd = signal.welch(X, axis=1, fs=0.2)

    if print_psd is True:     
        plt.figure(10)
        for i in range(n_comp):
            plt.plot(freqs, - i * 30 + psd[i, :], linewidth=0.7)
        
        print(np.max(psd, axis=1))
        print(np.max(psd))
    
    # %% Seleciona quais as componentes que estão realmente infectadas
    
    # Cria um vetor com os valores máximos de cada sinal do psd
    # São analisados os 5 primeiras componentes independentes
    a = np.max(psd[0:4, :], axis=1)
    
    # Decide se esse sinal está ou não contaminado baseado no valor máximo
    for i, j in enumerate(a):
        # TODO: procurar uma forma automática de conseguir esse valor
        if j > 390:
            a[i] = 1
        else:
            a[i] = 0
    
    # Cria um vetor com a posição das componentes referentes aos artefatos
    exc = np.where(a)[0]
    
    if print_psd is True:
        print('exc:')
        print(exc)
        
    # %% Subtrai os canais poluidos do sinal original
    
    # Se houver contaminação do sinal
    if np.size(exc) != 0:
        
        if print_overlay is True:
            ica.plot_components()
            plt.show()
            ica.plot_overlay(raw_reconst, exclude=exc, start=0, stop=2500)
            plt.show()
        
        # Realize a exclusão desses sinais do objeto de ica
        ica.exclude = exc
        
        # Aplica o ICA no sinal
        ica.apply(raw_reconst)
        
        if print_res is True:
            print('Artefato detectado e sinal limpo com sucesso')
         
        # retorna o sinal reconstruido e limpo e uma flag de limpeza
        return raw_reconst, 1
    
    elif print_res is True:
        print('O sinal não possui artefato')
    
    # Senão possui artefatos, retorne o mesmo sinal e uma flag indicando
    # que não há artefatos
    return raw.copy(), 0