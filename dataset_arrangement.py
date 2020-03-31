# -*- coding: utf-8 -*-

"""
# %% Reorganização do Dataset

# Esse código tem por objetivo organizar o dataset utilizado com o objetivo de
# facilitar a escrita dos códigos futuros, deixando-os mais limpos e com uma
# leitura mais fácil. O dataset é intitulado _Four class motor imagery (001-2014)
# e pode ser baixado pelo link: (http://bnci-horizon-2020.eu/database/data-sets).

# No site os arquivos encontram-se em formato .mat para ser aberto na linguagem
# MatLab, logo é preciso fazer um tratamento nesses dados antes de utilizá-los
# em pyhton. Ao carregar o arquivo com o comando '''readmat''' é carregada uma
# variável dicionário na qual o indice de interesse é o 'data'. Após carregar
# essa instância do arquivo, é criada uma variável do tipo array de 8 dimensões.

# %% Carregamento das bibliotecas e nome dos arquivos do dataset:
"""
from numpy.distutils.fcompiler import none
from scipy.io import loadmat
import numpy as np
import os
import mne

# Carrega os arquivos de cada uma das pessoas em uma lista (Sem o sujeito 4, pois seu
# dataset está danificado)
list_of_subj = ["A01T.mat", "A02T.mat", "A03T.mat", "A05T.mat",
                "A06T.mat", "A07T.mat", "A08T.mat", "A09T.mat",
                "A01E.mat", "A02E.mat", "A03E.mat", "A05E.mat",
                "A06E.mat", "A07E.mat", "A08E.mat", "A09E.mat"]

# nome da pasta onde está o dataset original
pasta = "dataset_files"

# Nome e sequencia dos canais utilizados no dataset
names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
         'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

"""
# %% Descrição do Dataset

# O resultado do carregamento feito por a seguir será um array (data) de 7 dimensões
# que pode ser acessado pelos" seguintes indices:


# "file = loadmat("A01T.mat")
# data = file['data']

# **data[id1][id2][id3][id4][id5][id6][id7]**

# Cada um dos indices desse array são referentes a uma informação do dataset:
# * id1: Indexa todo o conjunto de informações deve ser deixado como zero
# * id2: Indexa o número de um dos 9 conjuntos de testes que foram realizados (0-8)
# * id3: É um indice apenas para reunir os próximos tipos de dados (deve ser deixado como zero)
# * id4: É um indice apenas para reunir os próximos tipos de dados (deve ser deixado como zero)
# * id5: Esse indice indica um tipo de informação específico que será indexado pelas próxi- 
#        mas dimensões do array:
#     * 0: Dados de amostragem (X)
#     * 1: Guarda o indice de onde inciam os testes nos dados de amostragem (trial)
#     * 2: Classe referente a cada um dos testes indexados anteriormente (y)
#     * 3: Frequencia de amostragem do dataset (fs)
#     * 4: Nomes de cada uma das classes indexadas enteriormente (classes)
#     * 5: Indica em quais testes indexados anteriormente estão presentes artefatos (Artifacts)
#     * 6: Genero da pessoa que está sendo analisada (gender)
#     * 7: Idade da pessoa que está sendo analisada (age)
# * id6: Esse indice faz referencia ao indice da entrada anterior quando esta for um vetor
#        (Para o caso de dados de amostragem, esse indice fará refencia a cada uma das
#        amostragens feitas no experimento de acordo com a frequencia de amostragem estabe- 
#        lecida em fs)
# * id7: Esse indice está disponível apenas para os dados de amostragem e irá referenciar
#        cada um dos 25 eletrodos utilizados na amostragem
"""


# %% Código da organização do dataset

def sort() -> None:
    for sbj in list_of_subj:  # Carrega o dataset de cada uma das pessoas
        try:
            local = pasta + '\\' + sbj
            file = loadmat(local)
            data = file['data']
            print("{} carregado com sucesso".format(sbj))
        except IOError:
            print('Não foi possível carregar {}'.format(sbj))
            continue

        # Pré-alocamento das variáveis que guardarão as epocas
        fs = 250
        epoch_rhand = np.zeros([int(fs * 2.5), 22, 72])
        epoch_lhand = np.zeros([int(fs * 2.5), 22, 72])
        epoch_feet = np.zeros([int(fs * 2.5), 22, 72])
        epoch_tongue = np.zeros([int(fs * 2.5), 22, 72])

        cnt = np.array([0, 0, 0, 0])  # Array para contar quantas vezes cada classe aparece por pessoa

        # Variaveis para guardar o dataset bruto de forma mais organizada
        raw_x = []
        raw_trials = []
        raw_y = []
        raw_classes = []

        for i in range(3, 9):  # Carrega as 6 (das 9) runs de interesse de cada pessoa
            # Carrega em variáveis as informações do dataset
            x = data[0][i][0][0][0].copy()
            trial = data[0][i][0][0][1].copy()
            y = data[0][i][0][0][2].copy()
            clas = data[0][i][0][0][4][0].copy()

            # Adiciona nas variaveis Raw as informações
            raw_x.append(x)
            raw_trials.append(trial)
            raw_y.append(y)
            raw_classes = clas

            for j, idx in enumerate(trial):  # Dentro de cada run, seleciona os intervalos com testes (trials)

                i1 = int(250 * 3.5 + idx)  # Primeiro limite do intervalo de interesse
                i2 = int(250 * 6.0 + idx)  # Segundo limite do intervalo de interesse

                # Verifica em qual classe pertence a esse intervalo e atribui a um conjunto de variaveis
                if clas[y[j] - 1][0] == ['left hand']:
                    epoch_lhand[:, :, cnt[0]] = x[i1:i2, :22]
                    cnt[0] += 1
                    # print("{}: mao direita".format(cnt[0]))

                elif clas[y[j] - 1][0] == ['right hand']:
                    epoch_rhand[:, :, cnt[1]] = x[i1:i2, :22]
                    cnt[1] += 1
                    # print("{}: mao esquerda".format(cnt[1]))

                elif clas[y[j] - 1][0] == ['feet']:
                    epoch_feet[:, :, cnt[2]] = x[i1:i2, :22]
                    cnt[2] += 1
                    # print("{}: Pés".format(cnt[2]))

                elif clas[y[j] - 1][0] == ['tongue']:
                    epoch_tongue[:, :, cnt[3]] = x[i1:i2, :22]
                    cnt[3] += 1
                    # print("{}: Língua".format(cnt[3]))

                else:
                    print("Classe não identificada")

            # Salva as epocas criando um arquivo para cada pessoa
            '''
            try:
                try:
                    np.savez("sorted_data/{}_epoch".format(sbj.strip(".mat")),
                             left=epoch_lhand, right=epoch_rhand,
                             feet=epoch_feet, tongue=epoch_tongue)
                except IOError:
                    os.makedirs("sorted_data")
                    np.savez("sorted_data/{}_epoch".format(sbj.strip(".mat")),
                             left=epoch_lhand, right=epoch_rhand,
                             feet=epoch_feet, tongue=epoch_tongue)
            except IOError:
                print("Não foi possível salvar os arquivos 'epoch'! ")
            '''

        # Após carregar todas as runs de uma pessoa, salva um arquivo com os dados dessa pessoa
        try:
            try:
                np.savez("raw_data/{}_raw".format(sbj.strip(".mat")),
                         X=raw_x, trial=raw_trials, y=raw_y,
                         classes=raw_classes)
            except IOError:
                os.makedirs("raw_data")
                np.savez("raw_data/{}_raw".format(sbj.strip(".mat")),
                         X=raw_x, trial=raw_trials, y=raw_y,
                         classes=raw_classes)
        except IOError:
            print("Não foi possível salvar os arquivos 'raw'! ")


# %%

# noinspection PyPep8Naming
def sort_montage():
    # Carrega o arquivo
    file = np.loadtxt('plotting_1005.txt', dtype={
        'names': ['ch', 'x', 'y', 'z'],
        'formats': ['S6', 'f4', 'f4', 'f4']
    })

    # Cria a variavel que ira guardar o nome dos canais
    ch_nums = len(file)
    ch_names = []
    coord = np.zeros([342, 3])

    # Passa pelo arquivo linha por linha
    for ii, jj in enumerate(file):
        # Converte a primeira coluna para string e salva na lista
        ch_names.append(file[ii][0].decode('ascii'))

        # Converte as coordenadas para float e guarda na matriz
        for coo in range(3):
            coord[ii, coo] = float(file[ii][coo + 1])

    # Salva em uma matriz as posições de cada um dos canais rferenciados em 'names'
    ch_coord = coord[np.where([ch_names[i] in names for i in range(ch_nums)])[0]]

    # Salva a posição do eletrodo Nasio para referencia
    Nz_pos = coord[np.where([ch_names[i] in ['Nz'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo lpa para referencia
    LPA_pos = coord[np.where([ch_names[i] in ['LPA'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo rpa para referencia
    RPA_pos = coord[np.where([ch_names[i] in ['RPA'] for i in range(ch_nums)])[0]].reshape(3)

    # Cria o dicionario de montagem do layout
    ch_pos = {k: v for k, v in zip(names, ch_coord)}

    # Cria o objeto de montagem do laout
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=Nz_pos,
        lpa=LPA_pos,
        rpa=RPA_pos
    )

    return montage


# %%

def sort_raw_fif():
    # Todos os eletrodos que serão utilizados são de eeg
    chanel = ['eeg'] * 22
    # Frequencia de amostragem do dataset
    sfreq = 250

    # Carrega a montagem dos sensores:
    mnt = sort_montage()

    # Cria a informação de todos os arquivos
    info = mne.create_info(ch_names=names,
                           sfreq=sfreq,
                           ch_types=chanel,
                           montage=mnt)

    # Pre carrega um array para os eventos
    eve = np.zeros([48, 3])

    filename: str = ' '

    for sbj in list_of_subj:  # Carrega cada um dos arquivos _raw
        sbj = sbj.strip('.mat')  # Retira a extensão .mat dos arquivos
        try:  # Tenta carregar o arquivo _raw.npz
            filename = sbj + "_raw.npz"
            local = "raw_data/" + filename
            file = np.load(local, allow_pickle=True)
            print("{} carregado com sucesso".format(filename))
        except IOError:
            print("Não foi possível carregar {}".format(filename))
            continue

        for i in range(6):  # Carrega cada uma das 6 runs dentro de um arquivo
            # Carrega o conjunto de dados
            X = file['X'][i, :, :]  # Salva uma cópia dos dados na variavel X
            X = X.transpose()[:22]  # Faz com que os sinais fiquem nas linhas

            # Carrega os dados para criação dos arquivos de eventos
            eve[:, [0]] = file['trial'][i, :]
            eve[:, [2]] = file['y'][i, :]

            # Cria o objeto RawArray com os dados
            raw = mne.io.RawArray(X, info)

            # Salva os arquivos _raw.fif
            try:
                try:
                    raw.save('raw_fif_files/{}_{}_raw.fif'.format(sbj, str(i + 1)))
                except IOError:
                    os.makedirs("raw_fif_files")
                    raw.save('raw_fif_files/{}_{}_raw.fif'.format(sbj, str(i + 1)))
            except IOError:
                print('Não foi possível salvar {}_{}_raw.fif'.format(sbj, str(i + 1)))

            # Salva os arquivos _eve.fif
            try:
                try:
                    mne.write_events('raw_fif_files/{}_{}_eve.fif'.format(sbj, str(i + 1)),
                                     eve)
                except IOError:
                    os.makedirs("raw_fif_files")
                    mne.write_events('raw_fif_files/{}_{}_eve.fif'.format(sbj, str(i + 1)),
                                     eve)
            except IOError:
                print('Não foi possível salvar {}_{}_eve.fif'.format(sbj, str(i + 1)))


# %%

def sort_montage_eog():
    # Carrega o arquivo
    file = np.loadtxt('plotting_1005.txt', dtype={
        'names': ['ch', 'x', 'y', 'z'],
        'formats': ['S6', 'f4', 'f4', 'f4']
    })

    # Cria a variavel que ira guardar o nome dos canais
    ch_nums = len(file)
    ch_names = []
    coord = np.zeros([342, 3])

    # Passa pelo arquivo linha por linha
    for ii, jj in enumerate(file):
        # Converte a primeira coluna para string e salva na lista
        ch_names.append(file[ii][0].decode('ascii'))

        # Converte as coordenadas para float e guarda na matriz
        for coo in range(3):
            coord[ii, coo] = float(file[ii][coo + 1])

    # Salva em uma matriz as posições de cada um dos canais rferenciados em 'names'
    ch_coord = coord[np.where([ch_names[i] in names for i in range(ch_nums)])[0]]

    # Salva a posição do eletrodo Nasio para referencia
    Nz_pos = coord[np.where([ch_names[i] in ['Nz'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo lpa para referencia
    LPA_pos = coord[np.where([ch_names[i] in ['LPA'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo rpa para referencia
    RPA_pos = coord[np.where([ch_names[i] in ['RPA'] for i in range(ch_nums)])[0]].reshape(3)

    # Cria o dicionario de montagem do layout
    ch_pos = {k: v for k, v in zip(names, ch_coord)}

    # Cria o objeto de montagem do laout
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=Nz_pos,
        lpa=LPA_pos,
        rpa=RPA_pos
    )

    return montage


# %%

def sort_raw_eog_fif():
    names_n = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
               'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
               'EOG1', 'EOG2', 'EOG3']

    # Todos os eletrodos que serão utilizados são de eeg
    chanel = ['eeg'] * 22 + ['eog'] * 3
    # Frequencia de amostragem do dataset
    sfreq = 250

    # Carrega a montagem dos sensores:
    mnt = sort_montage_eog()

    # Cria a informação de todos os arquivos
    info = mne.create_info(ch_names=names_n,
                           sfreq=sfreq,
                           ch_types=chanel,
                           montage=mnt)

    # Pre carrega um array para os eventos
    eve = np.zeros([48, 3])

    for sbj in list_of_subj:  # Carrega cada um dos arquivos _raw
        sbj = sbj.strip('.mat')  # Retira a extensão .mat dos arquivos
        filename = sbj + "_raw.npz"
        try:  # Tenta carregar o arquivo _raw.npz
            local = "raw_data/" + filename
            file = np.load(local, allow_pickle=True)
            print("{} carregado com sucesso".format(filename))
        except IOError:
            print("Não foi possível carregar {}".format(filename))
            continue

        for i in range(6):  # Carrega cada uma das 6 runs dentro de um arquivo
            # Carrega o conjunto de dados
            X = file['X'][i, :, :]  # Salva uma cópia dos dados na variavel X
            X = X.transpose()  # Faz com que os sinais fiquem nas linhas

            # Carrega os dados para criação dos arquivos de eventos
            eve[:, [0]] = file['trial'][i, :]
            eve[:, [2]] = file['y'][i, :]

            # Cria o objeto RawArray com os dados
            raw = mne.io.RawArray(X, info)

            # Salva os arquivos _raw.fif
            try:
                try:
                    raw.save('raw_eog_fif_files/{}_{}_raw.fif'.format(sbj, str(i + 1)))
                except IOError:
                    os.makedirs("raw_eog_fif_files")
                    raw.save('raw_eog_fif_files/{}_{}_raw.fif'.format(sbj, str(i + 1)))
            except IOError:
                print('Não foi possível salvar {}_{}_raw.fif'.format(sbj, str(i + 1)))

            # Salva os arquivos _eve.fif
            try:
                try:
                    mne.write_events('raw_eog_fif_files/{}_{}_eve.fif'.format(sbj, str(i + 1)),
                                     eve)
                except IOError:
                    os.makedirs("raw_eog_fif_files")
                    mne.write_events('raw_eog_fif_files/{}_{}_eve.fif'.format(sbj, str(i + 1)),
                                     eve)
            except IOError:
                print('Não foi possível salvar {}_{}_eve.fif'.format(sbj, str(i + 1)))
