import csv
import pandas as pd
import os
from collections import deque


class InsightCSVReader:

    def __init__(self, filepath: str):
        if os.path.exists(filepath):
            self._filepath = filepath
        else:
            raise FileNotFoundError("O caminho informado não existe.")

        self.info = dict()
        self._data = pd.DataFrame()

    def _generate_info(self, info_array: list):
        for i in info_array:
            key, value = i.split(":")

            try:
                value = float(value)
            except ValueError:
                pass

            self.info[key] = value

    def set(self):
        with open(self._filepath) as f:
            data = csv.reader(f)
            iterable_data = iter(data)

            self._generate_info(next(iterable_data))

            header = next(iterable_data)
            temp = deque()
            while (row := next(iterable_data, None)) is not None:
                temp.append(row)

            self._data = pd.DataFrame(temp, columns=header, dtype=float)

    @property
    def data(self):
        return self._data