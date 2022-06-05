"""UCI Wine dataset

https://archive.ics.uci.edu/ml/datasets/Wine
"""

import numpy as np
import pandas as pd
from src.dataloaders.uci.uci_base import UCIData

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


class WineData(UCIData):
    """Dataloader for wine data

    Args:
        file_path (str / pathlib.Path)
    """
    def __init__(self, file_path="winequality-red.csv", url=URL):
        super().__init__(file_path=file_path, url=url)

    def load_full_data(self, shuffle=False):
        """Load csv data into np array"""
        np.random.seed(self.seed)
        data = pd.read_csv(self.file_path, header=0, delimiter=';').values
        if shuffle:
            self.data = data[np.random.permutation(np.arange(len(data)))]
        else:
            self.data = data
