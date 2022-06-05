"""UCI dataset"""
from abc import abstractmethod
import logging
from pathlib import Path
import numpy as np
import torch.utils.data as torch_data
from sklearn.model_selection import KFold
import urllib.request

# TODO: Check all url:s


class UCIData():
    """UCI base class"""
    def __init__(self, file_path, url, seed=0):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.url = url
        self.file_path, valid_path = self._validate_file_path(file_path)
        if not valid_path:
            self.download_remote()
        self.data = None
        self.output_dim = 1
        self.seed = seed
        self.load_full_data()
        self.num_samples, self.input_dim = self.data.shape
        self.input_dim -= 1

    def _validate_file_path(self, file_path):
        """Validate path"""
        file_path = Path(file_path)
        file_path = file_path.expanduser()
        valid = True
        if not file_path.exists():
            self._log.warning(
                "Dataset does not exist locally, downloading from: {}".format(
                    self.url))
            valid = False
        return file_path, valid

    @abstractmethod
    def load_full_data(self):
        """Load UCI data into np array"""
        pass

    def download_remote(self):
        """Download UCI dataset from remote location"""
        urllib.request.urlretrieve(url=self.url, filename=self.file_path)

    def datasplit_generator(self, num_splits, batch_size, transform=False):
        """Create a generator of datasplits"""
        split_generator = KFold(n_splits=num_splits).split(self.data)
        for idx in split_generator:
            train_idx, test_idx = idx
            x_train, y_train = self.data[train_idx, :self.
                                         input_dim], self.data[train_idx, self.
                                                               input_dim:]

            x_mean, x_std = get_stats(x_train)
            y_mean, y_std = get_stats(y_train)

            x_test, y_test = self.data[test_idx, :self.input_dim], self.data[
                test_idx, self.input_dim:]

            x_train = (x_train - x_mean) / x_std
            y_train = (y_train - y_mean) / y_std

            x_test = (x_test - x_mean) / x_std
            y_test = (y_test - y_mean) / y_std

            train = uci_dataloader(x_train, y_train, batch_size)
            test = uci_dataloader(x_test, y_test, y_test.shape[0])
            yield train, test

    def create_train_val_split(self, split_ratio):
        """Create simple data split

        Args:
            split_ratio (float): train / val ratio.
        """
        if self.data is None:
            self._log.error("Data is not loaded for {}".format(self))
            return None
        indices = np.arange(0, self.num_samples)
        np.random.shuffle(indices)
        num_train_samples = int(split_ratio * self.num_samples)
        train_data = self.data[:num_train_samples, :]
        val_data = self.data[num_train_samples:, :]

        x_train, y_train = train_data[:, :-1], train_data[:, -1:]
        x_val, y_val = val_data[:, :-1], val_data[:, -1:]
        return x_train, y_train, x_val, y_val


class _UCIDataset(torch_data.Dataset):
    """Internal representation of a subset of UCI data"""
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.num_samples, self.input_dim = self.x_data.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index, :]


def uci_dataloader(x_data, y_data, batch_size):
    """Generate a dataloader"""
    dataset = _UCIDataset(x_data, y_data)
    return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_stats(data):
    return data.mean(axis=0), data.var(axis=0)**0.5
