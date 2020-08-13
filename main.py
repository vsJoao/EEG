# -*- coding: utf-8 -*-

# Configs utilizadas
from configs.timing_config import *
from configs.database_names import *
from configs.classification_consts import *

from classes import Epochs
from classes import FBCSP

import utils

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from itertools import combinations, product
from scipy.stats import mode
from sklearn import svm
from time import time

