"""UCI Boston housing dataset
http://lib.stat.cmu.edu/datasets/boston
"""

import numpy as np
import pandas as pd
from src.dataloaders.uci.uci_base import UCIData

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"


class BostonData(UCIData):
    """Dataloader for Boston housing data

    Args:
        file_path (str / pathlib.Path)
    """
    def __init__(self, file_path, url=URL):
        super().__init__(file_path, url)

    def load_full_data(self, shuffle=False):
        """Load csv data into np array"""
        np.random.seed(self.seed)
        data = pd.read_csv(self.file_path, header=None, delimiter="\s+").values
        if shuffle:
            self.data = data[np.random.permutation(np.arange(len(data)))]
        else:
            self.data = data
