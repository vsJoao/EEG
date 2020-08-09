import mne
import os
import dataset_arrangement as dta
import numpy as np


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
        raw, ev = pick_file(f_loc, sbj, fnum)

    return [raw, ev]


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