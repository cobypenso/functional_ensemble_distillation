import logging
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py


class Cifar10DataCorrupted:
    """CIFAR data with corruptions, wrapper. To create an h5py file from .npy files, use the make_h5py_data_file()
        function below
    """

    def __init__(self, corruption, intensity, data_dir="data/", torch_data=True, ind=None):
        self._log = logging.getLogger(self.__class__.__name__)

        corruption_list = ["test", "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                           "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
                           "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
                           "zoom_blur"]

        if corruption not in corruption_list or intensity > 5:
            self._log.info("Data not found: corruption or intensity does not exist")

        else:

            if corruption == "test":
                self.set = torchvision.datasets.CIFAR10(root="/data",
                                                        train=False,
                                                        download=True)

                data = np.array(self.set.data)
                labels = self.set.targets

            else:

                filepath = data_dir + "dataloaders/data/CIFAR-10-C/corrupted_data.h5"
                with h5py.File(filepath, 'r') as f:
                    grp = f[corruption]
                    data = grp["data"][()]
                    grp = f["labels"]
                    labels = grp["labels"][()]

                set_size = 10000
                data = data[((intensity - 1) * set_size):(intensity * set_size), :, :, :]
                labels = labels[((intensity - 1) * set_size):(intensity * set_size)]

            if ind is not None:
                data = data[ind, :, :, :]
                labels = [ind]

            self.set = CustomSet(data, labels, torch_data=torch_data)

            self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                            "horse", "ship", "truck")
            self.num_classes = len(self.classes)
            self.set = CustomSet(data, labels, torch_data=torch_data)

            self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                            "horse", "ship", "truck")
            self.num_classes = len(self.classes)


class CustomSet:

    def __init__(self, data, labels, torch_data=True):
        self.data = data
        self.labels = labels
        self.input_size = self.data.shape[0]
        self.torch_data = torch_data

    def __len__(self):
        return self.input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.torch_data:
            img = transforms.ToTensor()(Image.fromarray(img))
        else:
            img = img / 255

        target = self.labels[index]

        return img, target


def make_h5py_data_file():
    """Save all corrupted data sets into one h5 file"""

    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "motion_blur", "pixelate", "saturate",
                       "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]

    data_dir = "data/CIFAR-10-C/"
    hf = h5py.File(data_dir + "corrupted_data.h5", 'w')
    grp = hf.create_group("labels")
    labels = np.load(data_dir + "labels.npy")
    grp.create_dataset("labels", data=labels)

    for corruption in corruption_list:
        print(corruption)
        grp = hf.create_group(corruption)

        data = np.load(data_dir + corruption + ".npy")
        grp.create_dataset("data", data=data)
    hf.close()


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = Cifar10DataCorrupted(corruption="brightness")
    loader = torch.utils.data.DataLoader(data.set,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)
    dataiter = iter(loader)
    img, labels = dataiter.next()

    # show images
    plt.imshow(np.transpose(torchvision.utils.make_grid(img).numpy(), (1, 2, 0)))
    plt.show()

    # print labels
    print(" ".join("%5s" % data.classes[labels[j]] for j in range(4)))


if __name__ == "__main__":
    main()