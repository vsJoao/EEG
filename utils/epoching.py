from classes import Epochs
from utils.artifact_remove import artifact_remove


# Recebe um arquivo de gravação de eeg e seus eventos, e separa em matrizes tridimensionais de classes: [M, N, C]
# Sendo M o número de eletrodos, N o número de amostras, e C o número de classes no dicionário
def epoch_raw_data(raw, events, e_dict, t_start, t_end, ica_start, ica_end) -> dict:

    # Guarda a quantidade de canais e calcula o numero de amostra das epocas
    ch = raw.pick('eeg').info['nchan']
    n_samp = int((t_end - t_start) * raw.info["sfreq"] + 1)

    # Pré aloca um dicionário que será utilizado como retorno da função
    X = dict()

    # Esse laço roda cada uma das trials dentro de um arquivo
    for n, i in enumerate(events[:, 0] / raw.info["sfreq"]):

        if events[n, 2] not in e_dict:
            continue

        # Salva a classe de movimento atual
        class_mov = e_dict[events[n, 2]]

        # Coleta uma amostra de (ica_end - ica_start) segundos para análise
        raw_samp = raw.copy().pick('eeg').crop(tmin=i+ica_start, tmax=i+ica_end)

        # Realiza a remoção de artefatos
        raw_clean, flag = artifact_remove(raw_samp)
        # raw_clean = raw_samp.copy()

        # Salva a epoca
        new_epc = \
            raw_clean.crop(tmin=t_start, tmax=t_end).get_data().reshape(ch, n_samp, 1)

        # Adiciona o sinal atual em sua respectiva classe do dicionário X
        try:
            X[class_mov].add_epoch(new_epc)
        except KeyError:
            X[class_mov] = Epochs(
                x=new_epc,
                classe=class_mov,
                fs=raw.info["sfreq"],
            )

    return X
