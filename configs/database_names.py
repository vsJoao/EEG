"""Variáveis que caracterizam o banco de dados"""
import os

# Nomes que referenciam os sujeitos e seus respectivos conjuntos de teste/treino
f_names_train = ["A01T", "A02T", "A03T", "A05T", "A06T", "A07T", "A08T", "A09T"]
f_names_test = ["A01E", "A02E", "A03E", "A05E", "A06E", "A07E", "A08E", "A09E"]

# Canais do dataset:
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
            'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
            'EOG1', 'EOG2', 'EOG3']

# Faz a definição dos eventos (classes de movimentos) com um dicionário.
# Neste caso o dicionário é feito para as seguintes classes:
# l - mão esquerda
# r - mão direita
# f - pés
# t - lígua
# a - movimento das maos
# b - movimento pés ou lingua
e_dict = {1: 'l', 2: 'f'}
# Pega os valores de forma não duplicada
e_classes = list()
for val in e_dict.values():
    e_classes.append(val) if val not in e_classes else ...
# Numeros de arquivos por pessoa
n_runs = 6

# Os arquivos 'sorted' são apenas um conjunto de matrizes com as epocas já separadas
# por classe, sendo um arquivo desses por sujeito
base_loc = "dataset_files"
epoch_train_loc = os.path.join(base_loc, "epoch_train")

# Os arquivos 'raw' são os sinais originais provisionados pelo dataset, mantendo
# todas as informações iniciais (trabalha em conjunto com os arquivos de eventos)
raw_loc = os.path.join(base_loc, "raw_data")

raw_fif_loc = os.path.join(base_loc, "raw_fif_files")