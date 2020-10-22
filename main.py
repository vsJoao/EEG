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

routines.training_set.training_data_routine()
routines.testing_set.testing_data_routine()
print([routines.classification.testing_set_classification(i[:-1]) for i in f_names_test])
