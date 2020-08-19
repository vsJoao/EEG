# -*- coding: utf-8 -*-

import routines.training_set
import routines.testing_set
import routines.classification

from configs.database_names import f_names_test

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="ticks")

# routines.training_set.training_data_routine()
# routines.testing_set.testing_data_routine()
# print([routines.classification.testing_set_classification(i[:-1]) for i in f_names_test])


res = [(0.7256944444444444, 0.6342592592592592),
       (0.5416666666666666, 0.38888888888888884),
       (0.7430555555555556, 0.6574074074074074),
       (0.3611111111111111, 0.14814814814814814),
       (0.4722222222222222, 0.2962962962962963),
       (0.65625, 0.5416666666666666),
       (0.7256944444444444, 0.6342592592592592),
       (0.6180555555555556, 0.49074074074074076)]
res_array = np.array(res)
sbj = np.array(["01", "02", "03", "05", "06", "07", "08", "09"])

sbj = sbj.reshape(-1, 1)
np.append(sbj, res_array, axis=1)
res_aum = np.append(sbj, res_array, axis=1)
df = pd.DataFrame(res_aum, columns=["Convidados", "Taxa de Acerto", "Kappa"])

sns.barplot(x="Convidados", y="Taxa de Acerto", data=df)
plt.show()