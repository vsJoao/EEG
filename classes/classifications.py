from classes import Epochs
import numpy as np
from utils import CSP


class FBCSP:

    def __init__(self, epc1: Epochs, epc2: Epochs, filterbank: dict, m: int = 2):
        self._w = dict()                    # Para cada uma das bandas de frequencia
        self.classe1 = epc1.classe          # Epocas de treino da primeira classe
        self.classe2 = epc2.classe          # Epocas de treino da segunda classe
        self.m = m                          # Quantidade de vetores CSP utilizados
        self.filterbank_dict = filterbank   # Dicionário das freqências utilizadas no banco de filtro

        self._fbcsp_calc(epc1, epc2)          # Salva as Matrizes de projeção espacial em self._w

    # Função que calcula e configura todas as matrizes de projeção em todos os filtros indicados
    def _fbcsp_calc(self, epc1: Epochs, epc2: Epochs):
        for f_band in self.filterbank_dict.values():
            # Calcula as matrizes de projeção para cada uma das bandas de frequencias
            self._w[f"{f_band[0]}-{f_band[1]}"] = \
                CSP.csp(Epochs.filt(epc1.data, f_band, epc1.fs), Epochs.filt(epc2.data, f_band, epc2.fs))

    # A função a seguir pega as duas epocas de atributos e gera as caracteristicas
    def generate_train_features(self, epc1: Epochs, epc2: Epochs, e_dict: dict) -> np.ndarray:
        # Retira as características do primeiro conjunto
        for i in range(epc1.data.shape[2]):
            try:
                f1 = np.append(f1, self.csp_feature(epc1.data[:, :, i]), axis=1)
            except (np.AxisError, UnboundLocalError):
                f1 = self.csp_feature(epc1.data[:, :, i])

        for i in range(epc2.data.shape[2]):
            try:
                f2 = np.append(f2, self.csp_feature(epc2.data[:, :, i]), axis=1)
            except (np.AxisError, UnboundLocalError):
                f2 = self.csp_feature(epc2.data[:, :, i])

        f1 = f1.transpose()
        f2 = f2.transpose()

        f = np.append(
            np.append(f1, np.tile(e_dict[epc1.classe], (f1.shape[0], 1)), axis=1),
            np.append(f2, np.tile(e_dict[epc2.classe], (f2.shape[0], 1)), axis=1),
            axis=0
        )

        return f

    # Extrai um vetor de características de um sinal recebido como parametro
    # Utilizando os dados do objeto fbcsp
    def csp_feature(self, x: np.ndarray) -> np.ndarray:

        if not self._w:
            raise ValueError("Ainda não existe uma matriz de projeção espacial na instancia")

        # Gera os indices dos m primeiras e m ultimas linhas da matriz
        m_int = np.hstack((np.arange(0, self.m), np.arange(-self.m, 0)))

        # Pré-aloca um vetor de caracteristicas
        f = np.zeros([(self.m * 2) * len(self._w), 1])

        for n, f_band in enumerate(self.filterbank_dict.values()):
            # Calcula-se a decomposição por CSP do sinal na banda de freq e seleciona as linhas [m_int]
            z = np.dot(
                self._w[f"{f_band[0]}-{f_band[1]}"],
                Epochs.filt(x, f_band, fs=250)
            )[m_int, :]

            # Calcula-se a variancia dessas linhas e em seguida o seu somatório
            var_z = np.var(z, axis=1)
            var_sum = np.sum(var_z)

            # Constrói-se o vetor de características desse sinal
            f[n * (2 * self.m):(2 * self.m) * (n + 1)] = np.log(var_z / var_sum).reshape(self.m*2, 1)

        return f
