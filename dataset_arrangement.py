"""
# %% Reorganização do Dataset

# Esse código tem por objetivo organizar o dataset utilizado com o objetivo de
# facilitar a escrita dos códigos futuros, deixando-os mais limpos e com uma
# leitura mais fácil. Além de estabelecer um padrão para que os dados entrem no
# algoritmo. O dataset é intitulado _Four class motor imagery (001-2014)
# e pode ser baixado pelo link: (http://bnci-horizon-2020.eu/database/data-sets).

# No site os arquivos encontram-se em formato .mat para ser aberto na linguagem
# MatLab, logo é preciso fazer um tratamento nesses dados antes de utilizá-los
# em pyhton. Ao carregar o arquivo com o comando '''readmat''' é carregada uma
# variável dicionário na qual o indice de interesse é o 'data'. Após carregar
# essa instância do arquivo, é criada uma variável do tipo array de 8 dimensões.

# %% Carregamento das bibliotecas e nome dos arquivos do dataset:
"""
from configs.database_names import *

from scipy.io import loadmat
import numpy as np
import os
import mne

"""
Descrição do Dataset

O resultado do carregamento feito por a seguir será um array (data) de 7 dimensões
que pode ser acessado pelos" seguintes indices:


"file = loadmat("A01T.mat")
data = file['data']

**data[id1][id2][id3][id4][id5][id6][id7]**

Cada um dos indices desse array são referentes a uma informação do dataset:
* id1: Indexa todo o conjunto de informações deve ser deixado como zero
* id2: Indexa o número de um dos 9 conjuntos de testes que foram realizados (0-8)
* id3: É um indice apenas para reunir os próximos tipos de dados (deve ser deixado como zero)
* id4: É um indice apenas para reunir os próximos tipos de dados (deve ser deixado como zero)
* id5: Esse indice indica um tipo de informação específico que será indexado pelas próxi- 
       mas dimensões do array:
    * 0: Dados de amostragem (X)
    * 1: Guarda o indice de onde inciam os testes nos dados de amostragem (trial)
    * 2: Classe referente a cada um dos testes indexados anteriormente (y)
    * 3: Frequencia de amostragem do dataset (fs)
    * 4: Nomes de cada uma das classes indexadas enteriormente (classes)
    * 5: Indica em quais testes indexados anteriormente estão presentes artefatos (Artifacts)
    * 6: Genero da pessoa que está sendo analisada (gender)
    * 7: Idade da pessoa que está sendo analisada (age)
* id6: Esse indice faz referencia ao indice da entrada anterior quando esta for um vetor
       (Para o caso de dados de amostragem, esse indice fará refencia a cada uma das
       amostragens feitas no experimento de acordo com a frequencia de amostragem estabe- 
       lecida em fs)
* id7: Esse indice está disponível apenas para os dados de amostragem e irá referenciar
       cada um dos 25 eletrodos utilizados na amostragem
"""


def sort() -> None:
    """Primeira organização do dataset

    Cada dataset deve ter sua função **sort()** especifica para colocar os dados no formato do programa.

    Essa função irá salvar os arquivos na pasta raw_data, devendo conter dois arquivos para cada pessoa
    sendo um para treino e outro para teste.

    Os nomes dos arquivos de saída devem estar no formato {sbj}{T/E}_raw.npz, onde sbj é o
    identificador da pessoa e deve ter a letra "T" para o arquivo de teste ou "E" para arquivo
    de treino.

    Arquivos npz:
    -------------
    X: list of ndarray
        Uma lista para guardar todas as gravações (ndarray) desse sujeito. O ndarray deve estar no
        formato [n_of_samples, n_of_chanels] (e.g. [95385, 22]).
    trial: list of ndarray
        Uma lista contendo um ndarray para cada gravação, marcando (o indice) o inicio de cada trial
        dessa gravação. Cada ndarray deve estar no formato [n_of_trials, 1].
    y: list of ndarray
        Uma lista contendo um ndarray para cada gravação. Cada ndarray é composto de números que refe-
        renciam a classe de cada uma das trials da gravação. Cada ndarray deve estar no formato [n_of_trials, 1]
    classes: ndarray
        Um array indicando, em ordem, os nomes que cada classe da lista y significam. (Ainda não implementado)
    sfreq: int
        Frequencia de Amostragem dos dados de EEG.

    """

    list_of_subj = os.listdir(originals_folder)

    for sbj in list_of_subj:  # Carrega o dataset de cada uma das pessoas
        if sbj.startswith("A04"):
            continue
        try:
            local = os.path.join(originals_folder, sbj)
            file = loadmat(local)
            data = file['data']
            print("{} carregado com sucesso".format(sbj))
        except IOError:
            print('Não foi possível carregar {}'.format(sbj))
            continue

        # Variaveis para guardar o dataset bruto de forma mais organizada
        raw_x = []
        raw_trials = []
        raw_y = []
        raw_classes = []
        sfreq = None

        for i in range(3, 9):  # Carrega as 6 (das 9) runs de interesse de cada pessoa
            # Carrega em variáveis as informações do dataset
            x = data[0][i][0][0][0].copy()
            trial = data[0][i][0][0][1].copy()
            y = data[0][i][0][0][2].copy()
            clas = data[0][i][0][0][4][0].copy()
            sfreq = data[0][i][0][0][3][0][0].copy()

            # Adiciona nas variaveis Raw as informações
            raw_x.append(x)
            raw_trials.append(trial)
            raw_y.append(y)
            raw_classes = clas

        # Após carregar todas as runs de uma pessoa, salva um arquivo com os dados dessa pessoa
        try:
            try:
                np.savez(os.path.join(raw_folder, f"{sbj.strip('.mat')}_raw"),
                         X=raw_x, trial=raw_trials, y=raw_y,
                         classes=raw_classes, sfreq=sfreq)
            except IOError:
                os.makedirs(raw_folder)
                np.savez(os.path.join(raw_folder, f"{sbj.strip('.mat')}_raw"),
                         X=raw_x, trial=raw_trials, y=raw_y,
                         classes=raw_classes, sfreq=sfreq)
        except IOError:
            print("Não foi possível salvar os arquivos 'raw'! ")


def sort_montage_eog(dataset_ch_names):
    # Carrega o arquivo
    file = np.loadtxt('plotting_1005.txt', dtype={
        'names': ['ch', 'x', 'y', 'z'],
        'formats': ['S6', 'f4', 'f4', 'f4']
    })

    # Cria a variavel que ira guardar o nome dos canais
    ch_nums = len(file)
    all_ch_names = []
    coord = np.zeros([342, 3])

    # Passa pelo arquivo linha por linha
    for ii, jj in enumerate(file):
        # Converte a primeira coluna para string e salva na lista
        all_ch_names.append(file[ii][0].decode('ascii'))

        # Converte as coordenadas para float e guarda na matriz
        for coo in range(3):
            coord[ii, coo] = float(file[ii][coo + 1]) / 10

    # Salva em uma matriz as posições de cada um dos canais rferenciados em 'names'
    ch_coord = coord[np.where([all_ch_names[i] in dataset_ch_names for i in range(ch_nums)])[0]]

    # Salva a posição do eletrodo Nasio para referencia
    Nz_pos = coord[np.where([all_ch_names[i] in ['Nz'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo lpa para referencia
    LPA_pos = coord[np.where([all_ch_names[i] in ['LPA'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo rpa para referencia
    RPA_pos = coord[np.where([all_ch_names[i] in ['RPA'] for i in range(ch_nums)])[0]].reshape(3)

    # Cria o dicionario de montagem do layout
    ch_pos = {k: v for k, v in zip(dataset_ch_names, ch_coord)}

    # Cria o objeto de montagem do laout
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=Nz_pos,
        lpa=LPA_pos,
        rpa=RPA_pos
    )

    return montage


def sort_raw_fif():
    import matplotlib.pyplot as plt

    list_of_subj = os.listdir(originals_folder)

    # Todos os eletrodos que serão utilizados são de eeg
    chanel = ['eeg'] * 22 + ['eog'] * 3
    # Frequencia de amostragem do dataset
    sfreq = 250

    # Carrega a montagem dos sensores:
    mnt = sort_montage_eog(ch_names)

    # Cria a informação de todos os arquivos
    info = mne.create_info(ch_names=ch_names,
                           sfreq=sfreq,
                           ch_types=chanel)

    # Pre carrega um array para os eventos
    eve = np.zeros([48, 3])

    for sbj in list_of_subj:  # Carrega cada um dos arquivos _raw
        sbj = sbj.strip('.mat')  # Retira a extensão .mat dos arquivos
        filename = sbj + "_raw.npz"
        try:  # Tenta carregar o arquivo _raw.npz
            local = os.path.join(raw_folder, filename)
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
            raw = mne.io.RawArray(X, info).set_montage(mnt)

            # Salva os arquivos _raw.fif
            try:
                try:
                    raw.save(f'{raw_fif_folder}/{sbj}_{i + 1}_raw.fif')
                except IOError:
                    os.makedirs(raw_fif_folder)
                    raw.save(f'{raw_fif_folder}/{sbj}_{i + 1}_raw.fif')
            except IOError:
                print('Não foi possível salvar {}_{}_raw.fif'.format(sbj, str(i + 1)))

            # Salva os arquivos _eve.fif
            try:
                try:
                    mne.write_events(
                        f'{raw_fif_folder}/{sbj}_{i + 1}_eve.fif',
                        eve
                    )
                except IOError:
                    os.makedirs(raw_fif_folder)
                    mne.write_events(
                        f'{raw_fif_folder}/{sbj}_{i + 1}_eve.fif',
                        eve
                    )
            except IOError:
                print('Não foi possível salvar {}_{}_eve.fif'.format(sbj, str(i + 1)))