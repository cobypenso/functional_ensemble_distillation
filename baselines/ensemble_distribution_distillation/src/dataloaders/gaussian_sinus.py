"""One dimensional dataset,
Gaussian with sinus wave mean, variance increasing in x

Taken from paper: "https://arxiv.org/abs/1906.01620"
"""
import logging
from pathlib import Path
import csv
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt


class GaussianSinus(torch.utils.data.Dataset):
    """1D sinusoidal data with noise, increasing with x"""
    def __init__(self,
                 store_file,
                 train=True,
                 range_=(-3, 3),
                 reuse_data=False,
                 n_samples=1000):

        super(GaussianSinus).__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.n_samples = n_samples
        self.range = range_
        self.train = train

        self.file = Path(store_file)
        if self.file.exists() and reuse_data:
            self.validate_dataset()
        else:
            self._log.info("Sampling new data")
            self.sample_new_data()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = None
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            for count, row in enumerate(csv_reader):
                if count == index:
                    sample = row
                    break
        inputs = sample[:-1]
        targets = sample[-1]

        return (np.array(inputs, dtype=np.float32),
                np.array([targets], dtype=np.float32))

    @staticmethod
    def x_to_y_mapping(x):
        """Noisy mapping defining the dataset"""
        mu = np.sin(x)
        sigma = 0.15 * 1 / (1 + np.exp(-x))
        y = np.random.multivariate_normal(mean=mu, cov=(np.diag(sigma)))
        return y, mu, sigma

    def sample_new_data(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        lower, upper = self.range
        if self.train:
            x = np.random.uniform(low=lower, high=upper, size=self.n_samples)

        else:
            x = np.random.uniform(low=lower, high=upper, size=self.n_samples)

        y, _, _ = self.x_to_y_mapping(x)
        combined_data = np.column_stack((x, y))
        np.random.shuffle(combined_data)
        np.savetxt(self.file, combined_data, delimiter=",")

    def validate_dataset(self):
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            assert self.n_samples - 1 == sum(1 for row in csv_reader)

    def get_full_data(self, sorted_=False):
        """Get full dataset as numpy array"""
        tmp_raw_data = list()
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]
        full_data = np.array(tmp_raw_data, dtype=float)
        if sorted_:
            full_data = full_data[full_data[:, 0].argsort()]
        return full_data


def plot_reg_data(data, ax):
    inputs = data[:, :-1]
    targets = data[:, -1]
    ax.scatter(inputs, targets)
    plt.show()


def plot_uncert(ax, data, x):
    inputs = data[:, :-1]
    targets = data[:, -1]
    mu = np.sin(x)
    sigma = 0.15 * 1 / (1 + np.exp(-x))
    ax.scatter(inputs, targets)

    ax.plot(x, mu, "r-", label="$\mu(x)$")
    ax.fill_between(x,
                    mu + sigma,
                    mu - sigma,
                    facecolor="blue",
                    alpha=0.5,
                    label="$\mu(x) \pm \sigma(x)$")
    plt.legend(prop={'size': 20})
    plt.show()


def main():
    _, ax = plt.subplots()
    dataset = GaussianSinus(store_file=Path("data/1d_gauss_sinus_1000"))
    start = -3
    end = 3
    step = 0.25
    x_length = int((end - start) / step)
    x = np.arange(start=start, stop=end, step=step, dtype=np.float)
    plot_uncert(ax, dataset.get_full_data(), x)


if __name__ == "__main__":
    main()
