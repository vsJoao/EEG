# -*- coding: utf-8 -*-

"""
Implementação das funções responsáveis por realizar a extração de característi-
cas por meio da técnica de CSP, maximizando a diferença de variância entre os
sinais de duas classes distintas e também técnicas de separação de sub-bandas
para análises utilizando banco de filtros FBCSP
"""

# %%

import numpy as np
import scipy.linalg as li
import mods

# %%


# Calculo da matriz de covariância espacial
def cov_esp(E):
    C = np.dot(E, E.transpose())
    C = C / (np.dot(E, E.transpose()).trace())
    return C


def eig_sort(X, cresc=False):
    value, vector = li.eig(X)
    value = np.real(value)
    vector = np.real(vector)

    if cresc is False:
        idx = np.argsort(value)[::-1]
    else:
        idx = np.argsort(value)
    value = value[idx]
    value = np.diag(value)
    vector = vector[:, idx]
    return [value, vector]
    
    
# calcula-se o csp de um conjunto de ensaios com a matrix x no formato [N, T, E] 
# sendo N o numero de canais, T o número de amostras e E o número de ensaios
def csp(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    
    # verifica os tamanhos das matrizes X e Y
    try:
        nx, tx, ex = X.shape
    except ValueError:
        nx, tx = X.shape
        ex = 1
    
    try:
        ny, ty, ey = Y.shape
    except ValueError:
        ny, ty = Y.shape
        ey = 1
    
    # verifica se os dois arrays possuem a mesma quantidade de canais
    if ny != nx:
        return 0
    else:
        n = nx
        del nx, ny
    
    # Calcula-se a média das matrizes de covariancia espacial para as duas classes
    Cx = np.zeros([n, n])
    for i in range(ex):
        Cx += cov_esp(X[:, :, i])
    
    Cx = Cx / ex
    
    Cy = np.zeros([n, n])
    for i in range(ey):
        Cy += cov_esp(Y[:, :, i])
    
    Cy = Cy / ey
    
    # calculo da variância espacial composta
    Cc = Cx + Cy
    Ac, Uc = eig_sort(Cc)

    # matriz de branquemento
    P = np.dot(np.sqrt(li.inv(Ac)), Uc.transpose())
    
    # Aplicando a transformação P aos Cx e Cy
    Sx = P.dot(Cx).dot(P.transpose())
    Sy = P.dot(Cy).dot(P.transpose())

    Ax, Ux = eig_sort(Sx, cresc=False)

    W = np.dot(P.transpose(), Ux).transpose()

    return np.real(W)