import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import mne
import csv
from utils.files_io import pick_file
from configs.database_names import *
from utils.artifact_remove import *

sns.set(style="ticks")


""" Pairs Plots """
# features = np.load("dataset_files_fbcsp_9/features_train/A01T_features.npy", allow_pickle=True).item()
# sns.set(style="ticks")
#
# lista = ['lr', 'lf', 'lt', 'rf', 'rt', 'ft']
# plt.figure(1)

# for n, pair in enumerate(lista):
#     features_df = pd.DataFrame(features[pair])
#
#     features_df[24] = features_df[24].map({1: "Mão Direita", 2: "Mão Esquerda", 3: "Pés", 4: "Língua"})
#     features_df = features_df.rename(columns={24: "Classes"})
#     ax = sns.pairplot(
#         features_df[[0, 1, 2, 3, "Classes"]], hue="Classes",
#         palette={"Mão Direita": "#9b59b6", "Mão Esquerda": "#3498db", "Pés": "#34495e", "Língua": "#e74c3c"}
#     )
#     ax._legend.remove()
#
#     ax.savefig(f"C:/Users/victo/Desktop/{pair}_figure.png")


""" Gráfico de Barras """
# res = [(0.7638888888888888, 0.6851851851851851),
#        (0.5069444444444444, 0.34259259259259256),
#        (0.7465277777777778, 0.6620370370370371),
#        (0.4444444444444444, 0.25925925925925924),
#        (0.4479166666666667, 0.2638888888888889),
#        (0.7048611111111112, 0.6064814814814815),
#        (0.7326388888888888, 0.6435185185185185),
#        (0.59375, 0.4583333333333333)]
#
# res_array = np.array(res)
# sbj = np.array(["01", "02", "03", "05", "06", "07", "08", "09"])
#
# sbj = sbj.reshape(-1, 1)
# np.append(sbj, res_array, axis=1)
# res_aum = np.append(sbj, res_array, axis=1)
# df = pd.DataFrame(res_aum, columns=["Convidado", "Taxa de Acerto", "Kappa"])
#
# df2 = df.melt(id_vars=["Convidado"], var_name="Variável", value_name="Valores")
#
# g = sns.barplot(x="Convidado", y="Valores", hue="Variável", data=df2, palette=["#3498db", "#34495e"])
# g.set_ylim(bottom=0, top=1)
# g.spines["right"].set_visible(False)
# g.spines["top"].set_visible(False)
#
# for index, row in df.iterrows():
#     g.text(
#         float(index)-0.2,
#         float(row["Taxa de Acerto"])+0.02,
#         round(float(row['Taxa de Acerto']), 4),
#         color='black', ha="center"
#     )
#     g.text(
#         float(index)+0.2,
#         float(row["Kappa"])+0.02,
#         round(float(row['Kappa']), 4),
#         color='black', ha="center"
#     )
#
# sns.despine(top=True, right=True)
# plt.show()


"""Dataset do Hermes"""
# from classes import InsightCSVReader
#
# dataset_path = "dataset_folder"
# file_name = "D francisquinha JV 3_24.06.20_18.39.03.md.pm.bp.csv"
# filepath = os.path.join(dataset_path, file_name)
#
# instancia = InsightCSVReader(filepath)
# instancia.set()
# print(instancia.info)  # Dicionário com informações da gravação
# print(instancia.data)  # Dataframe do pandas com todos os dados


""" Impressão de Sinais """
data, eve = pick_file(f_loc=raw_fif_folder, sbj="A01T", fnum=1)
data.plot(events=None, n_channels=25, duration=10, scalings={'eeg': 40, 'eog': 50}, start=32)
plt.savefig('antes_ica')
raw_clean, flag = artifact_remove(data.crop(tmin=32, tmax=42), print_all=True)
raw_clean.plot(events=None, n_channels=22, duration=10, scalings={'eeg': 40})
plt.savefig('apos_ica')
plt.show()

