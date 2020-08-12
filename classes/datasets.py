import numpy as np


class Epochs:
    def __init__(self, x: np.ndarray, fs, classe: str) -> None:
        # Epoca original de dados
        self.data = x
        # Classe do conjunto de epocas
        self.classe = classe
        # Taxa de amostragem do sinal
        self.fs = fs

        # bloco para verificar principalmente se há mais de uma matriz de epocas
        try:
            self.n_ch = self.data.shape[0]
            self.n_samp = self.data.shape[1]
            self.n_trials = self.data.shape[2]
        except IndexError:
            self.n_ch = self.data.shape[0]
            self.n_samp = self.data.shape[1]
            self.n_trials = 1
            self.data = self.data.reshape(self.n_ch, self.n_samp, 1)

    # Adiciona uma epoca no conjunto original de dados
    def add_epoch(self, new_data: np.ndarray):
        self.data = np.append(self.data, new_data, axis=2)
        self.n_trials += 1

    # Aplica o filtro em todos os sinais originais
    @classmethod
    def filt(cls, x: np.ndarray, freq_band: list, fs: int, ordem=6, rs=20):
        from scipy import signal

        sos = signal.iirfilter(
            N=ordem, Wn=freq_band, rs=rs, btype='bandpass',
            output='sos', fs=fs, ftype='cheby2'
        )

        filtered = signal.sosfilt(sos, x, axis=1)

        return filtered

    @classmethod
    def dict_from_filepath(cls, file_path):
        return np.load(file_path, allow_pickle=True).item()