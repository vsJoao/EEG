# -*- coding: utf-8 -*-

# %% Código de ICA:

#
# Este código implementa o algoritmo fast ica, responável por realizar a separação das com-
# ponentes independentes de um conjunto de sinais por meio de Blind Source Separation.

# As primeiras etapas do código são as seguintes:

# - CentralizaÃ§Ã£o: Essa etapa consiste em retirar a média dos sinais e subtraÃ­-la dos sinais
#                  originais, para obter um sinal sem média.

# - Branqueamento: Branqueamento é o nome de um processo realizado sobre os sinais de amos-
#                  tra para torná-los variaveis descorrelacionadas, isto é, a matriz de co-
#                  variancia dos sinais Ã© a matriz identidade. O branqueamento é obtido da 
#                  seguinte forma:
    
#                      $$ z = Vx, \text{ onde } V = ED^{-1/2}E^T $$
                    
#                  em que z são os novos sinais brancos, V é a matriz de branqueamento, E é
#                  a matriz ortogonal de autovetores da matriz de covariancia e D é a matriz
#                  diagonal de autovalores da matriz de covariancia.
    
# - Algoritmo: Para realizar a estimação de uma componente inde-
#              pendente, realiza-se o seguinte algorÃ­tmo:
   
#     - 1. Escolhe-se um vetor aleatório $w_0$ (para cada componente) para iniciar a iteração
#          de ponto fixo;
#     - 2. Atualiza-se o vetor $w$ da seguinte forma:
#                     $$ w^* = E\{ z g(w^T_i z) \} - E\{ g'(w^T_i z) \} w_i $$
#                     $$ w_{i+1} = \dfrac{w^*}{\vert\vert w^* \vert\vert} $$
#     onde g(x) e g'(x) são funções nÃ£o quadraticas, impar e par, respectivamente;
#     - 3. Se não convergir, retornar ao passo anterior.
#

# %%

import numpy as np
from scipy.linalg import sqrtm

# %% Funções para a realização de ICA em um conjunto de funções


def center(X, ax=1):  # Centralização dos dados
    X = np.array(X)
    media = X.mean(axis=ax, keepdims=True)
    return X - media


def whitening(X):       # Branqueamento dos dados
    X = np.array(X)
    cov = np.cov(X)
    
    [d, E] = np.linalg.eig(cov)
    D = np.diag(d)
    
    D_inv = sqrtm(np.linalg.inv(D))
    V = np.dot(E, np.dot(D_inv, E.T))
    return np.dot(V, X)


def root(x):            # Calcula a raiz de um array
    x = np.array(x)
    return np.sqrt(x)


def g(x):               # Define-se a função g como a tanh
    return np.tanh(x)


def g_der(x):           # E a derivada da função g
    return 1 - g(x) * g(x)


def fast_ICA(x, err_min=0.001, print_it=False, max_it=100):
    x = np.array(x)     # Garante que os sinais sejam arrays do numpy
    z = center(x)       # Centraliza os dados de entrada
    z = whitening(z)    # Faz o branqueamento dos sinais
    
    n_comp, n_samp = np.array(np.shape(x))    # Captura as medidas do conjunto de sinais
    
    w = np.random.rand(n_comp, n_comp)       # Gera todos os vetores wi iniciais aleatórios
    err = np.ones([n_comp, 1])                 # Inicia os erros com valores um
    it = np.zeros(n_comp)                     # Inicia o contador de iterações com valor zero
    it_ext = 0                                 # Contador de iterações do laço mais externo
    
    temp_err = np.ones([n_comp, 1])
    
    while np.max(err) > err_min:    # Realiza o loop principal até atingir o erro
        
        if np.max(it) >= max_it:      # Verifica se já alcançou o maximo de iterações
            break
        
        it_ext = it_ext + 1           # Incrementa a iteração
        
        for i in range(n_comp):       # Realiza um laço para estimar cada uma das componentes por iteração 
            
            if err[i] < err_min:      # Se esta componente já atingiu o erro especificado, pule a iteração
                continue              # Então pule esta iteração
            
            it[i] = it[i] + 1    # Incrementa o contador de iterações para esta componente
            
            # Faz a aproximação desta componente utilziando negentropia
            temp = (z * g(np.dot(w[:, i].T, z))).mean(axis=1) - g_der(np.dot(w[:, i], z)).mean() * w[:, i]
            temp = temp / np.sqrt((temp ** 2).sum())
            
            # Calcula o erro baseado no novo e no vetor anterior
            temp_err[i] = err[i]
            err[i] = np.abs(1 - np.abs(np.dot(w[:, i], temp)))
            w[:, i] = temp   # Guarda o novo vetor

        # Ortogonalização da matriz de vetores aproximados:
        w_ort = sqrtm(np.linalg.inv(w.dot(w.T)))
        w = np.dot(w_ort, w)
        
        # Verifica se o erro dessa iteração é aproximadamente o valor da anterior
        dif_err = np.abs(temp_err - err)
        # E para o laço caso o erro esteja se repetindo
        if np.max(dif_err) <= 1e-14:
            break
            
    # Imprime quantas iterações foram necessárias para calcular as componentes
    if print_it is True:
        print(it)
    
    return w.T.dot(z)
