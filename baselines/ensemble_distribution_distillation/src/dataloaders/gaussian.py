from pathlib import Path
import csv
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import logging


class SyntheticGaussianData(torch.utils.data.Dataset):
    def __init__(self,
                 mean_0,
                 cov_0,
                 mean_1,
                 cov_1,
                 store_file,
                 reuse_data=False,
                 n_samples=1000,
                 sample=True,
                 ratio_0_to_1=0.5):
        super(SyntheticGaussianData).__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.mean_0 = np.array(mean_0)
        self.mean_1 = np.array(mean_1)
        self.cov_0 = np.array(cov_0)
        self.cov_1 = np.array(cov_1)
        self.n_samples = n_samples
        self.sample = sample
        self.ratio_0_to_1 = ratio_0_to_1
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
        labels = int(float(sample[-1]))
        return (np.array(inputs,
                         dtype=np.float32), np.array(labels, dtype=np.long))

    def get_instance_of_label(self, label_requested):
        sample = None
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            for row in csv_reader:
                sample = row
                label_found = int(float(sample[-1]))
                if label_found == label_requested:
                    break
            if not sample:
                self._log.error("No data points with label {} found".format(
                    label_requested))
        inputs = sample[:-1]
        return (np.array(inputs,
                         dtype=np.float32), np.array(label_found,
                                                     dtype=np.long))

    def sample_new_data(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)

        if self.sample:
            size_0 = int(np.floor(self.n_samples * self.ratio_0_to_1))
            size_1 = self.n_samples - size_0

            sampled_x_0 = np.random.multivariate_normal(mean=self.mean_0,
                                                        cov=self.cov_0,
                                                        size=size_0)
            y_0 = np.zeros((size_0, 1))
            sampled_x_1 = np.random.multivariate_normal(mean=self.mean_1,
                                                        cov=self.cov_1,
                                                        size=size_1)
            y_1 = np.ones((size_1, 1))

            all_x = np.row_stack((sampled_x_0, sampled_x_1))
            all_y = np.row_stack((y_0, y_1))

        else:
            x_min = -4
            x_max = 3
            num_points = 1000
            x_0, x_1 = np.linspace(x_min, x_max, num_points), np.linspace(x_min, x_max, num_points)
            x_0, x_1 = np.meshgrid(x_0, x_1)
            all_x = torch.tensor(np.column_stack((x_0.reshape(num_points ** 2, 1), x_1.reshape(num_points ** 2, 1))),
                                 dtype=torch.float32)
            # Note: this is just a placeholder
            all_y = np.zeros((num_points**2, 1))

        combined_data = np.column_stack((all_x, all_y))
        np.random.shuffle(combined_data)
        np.savetxt(self.file, combined_data, delimiter=",")

    def validate_dataset(self):
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            assert len(self.mean_0) == len(next(csv_reader)) - 1
            assert self.n_samples - 1 == sum(1 for row in csv_reader)

    def get_full_data(self):
        tmp_raw_data = list()
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]
        return np.array(tmp_raw_data, dtype=float)


def plot_2d_data(data, ax):
    inputs = data[:, :-1]
    print(inputs)
    labels = data[:, -1]
    label_1_inds = labels == 0
    label_2_inds = labels == 1
    ax.scatter(inputs[label_1_inds, 0], inputs[label_1_inds, 1], label="blue")
    ax.scatter(inputs[label_2_inds, 0], inputs[label_2_inds, 1], label="red")
    plt.show()


def main():
    _, ax = plt.subplots()
    dataset = SyntheticGaussianData(mean_0=[0, 0],
                                    mean_1=[10, 0],
                                    cov_0=np.eye(2),
                                    cov_1=np.eye(2),
                                    store_file=Path("data/2d_gaussian_1000"))
    plot_2d_data(dataset.get_full_data(), ax)


if __name__ == "__main__":
    main()
