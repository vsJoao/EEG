from classes import Epochs
import numpy as np
from utils import csp


class FBCSP:
    """ Cria uma instancia do Objeto FBCSP

    Uma instancia será um modelo que conterá as matrizes de projeção espacial
    de um conjunto de sinais de duas classes diferentes dado um banco de filtros.

    Atributes
    ---------
    _w: dict of np.ndarray
        Cada indice desse dicionário será referente a uma banda de frequencia e a
        matriz irá fazer a decomposição do sinal naquela faixa de frequência
    classe1: str
        Nome dado a classe do primeiro conjunto de sinais
    classe2: str
        Nome dado a classe do segundo conjunto de sinais
    m: int
        Quantidade de linhas a serem utilizadas para extração do vetor de caracteristicas
        na projeção espacial
    filterbank_dict: dict
        Banco de filtros que será utilizado nesse modelo de FBCSP. Cada banda é representado
        por uma lista da forma [low_freq, high_freq].

    Methods
    -------
    generate_train_features(epc1, epc2, e_dict)
        Gera um conjunto de caracteristicas de um conjunto de treino
    csp_feature(x)
        Gera um vetor de características do sinal x utilizando o modelo salvo na instância


    """
    def __init__(self, epc1: Epochs, epc2: Epochs, filterbank: dict, m: int = 2):
        """ Cria uma instancia de FBCSP

        Parameters
        ----------
        epc1: Epochs Object
            Conjunto de dados da primeira classe
        epc2: Epochs Object
            Conjunto de dados da segunda classe
        filterbank: dict
            Banco de filtros no formato de dicionário de listas
        m: int
            Quantidade de linhas que será utilizada na extração de características da projeção espacial

        """
        self._w = dict()                    # Para cada uma das bandas de frequencia
        self.classe1 = epc1.classe          # Epocas de treino da primeira classe
        self.classe2 = epc2.classe          # Epocas de treino da segunda classe
        self.m = m                          # Quantidade de vetores CSP utilizados
        self.filterbank_dict = filterbank   # Dicionário das freqências utilizadas no banco de filtro

        self._fbcsp_calc(epc1, epc2)        # Salva as Matrizes de projeção espacial em self._w

    def _fbcsp_calc(self, epc1: Epochs, epc2: Epochs):
        """ Calcula as matrizes de projeção espacial do modelo

        Dados dois conjuntos de Epocas de classes diferentes, encontra-se as matrizes de
        projeção espacial que serão utilizadas para gerar a decomposição por CSP dos sinais
        e classificá-los. Guarda as matrizes de projeção espacial na instancia para utiliza-
        ção na geração das características.

        Parameters
        ----------
        epc1: Epochs Object
            Conjunto de dados da primeira classe.

        epc2: Epochs Object
            Conjunto de dados da segunda classe.

        """
        for f_band in self.filterbank_dict.values():
            # Calcula as matrizes de projeção para cada uma das bandas de frequencias
            self._w[f"{f_band[0]}-{f_band[1]}"] = \
                csp.csp(Epochs.filt(epc1.data, f_band, epc1.fs), Epochs.filt(epc2.data, f_band, epc2.fs))

    def generate_train_features(self, epc1: Epochs, epc2: Epochs, e_dict: dict) -> np.ndarray:
        """ Gera um conjunto de características dos dois conjuntos de dados

        O conjunto de dados passados, é ideal que seja o mesmo anteriormente utilizado
        para gerar as matrizes de projeção espacial, já que essa função tem o objetivo
        de gerar as características de treino.

        Parameters
        ----------
        epc1: Epochs Object
            Conjunto de dados da primeira classe.
        epc2: Epochs Object
            Conjunto de dados da segunda classe.
        e_dict: dict
            Um dicionário que converte o nome da classe para um id do tipo int válido.

        Returns
        -------
        f: np.ndarray
            Uma matriz contendo como vetores linha, os vetores de características
            extraidos dos dois sinais. A ultima coluna será o id de cada classe, como
            definido no dicionário.

        """
        # Retira as características do primeiro conjunto
        f1 = self.csp_feature(epc1.data[:, :, 0])
        for i in range(1, epc1.n_trials):
            f1 = np.append(f1, self.csp_feature(epc1.data[:, :, i]), axis=1)

        # Retira as caracteristicas do segundo conjunto
        f2 = self.csp_feature(epc2.data[:, :, 0])
        for i in range(1, epc2.n_trials):
            f2 = np.append(f2, self.csp_feature(epc2.data[:, :, i]), axis=1)

        f1 = f1.transpose()
        f2 = f2.transpose()

        f = np.append(
            np.append(f1, np.tile(e_dict[epc1.classe], (f1.shape[0], 1)), axis=1),
            np.append(f2, np.tile(e_dict[epc2.classe], (f2.shape[0], 1)), axis=1),
            axis=0
        )

        return f

    def csp_feature(self, x: np.ndarray) -> np.ndarray:
        """ Extrai um vetor de características de um sinal multivariado x

        Parameters
        ----------
        x: np.ndarray
            Um sinal de EEG mutivariado no formato de uma matriz MxN, onde M é o número
            de canais e N é p número de amostras.

        Returns
        -------
        f: np.ndarray
            Um vetor de características extraído da matriz x utilizando o modelo FBCSP
            armazenado na instancia. O tamanho do vetor depende da quantidade de filtros
            utilizados no modelo e da quantidade de linhas utilizadas na decomposição CSP
            (self.m)

        """
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