import os
import numpy as np
import matplotlib.pyplot as plt


NAMES = np.array([
    "MNIST",
    "KMNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR10_HMC",
    "CIFAR100",
    "TinyImageNet200",
    "SVHN",
])

EXISTS = np.array([os.path.exists(e) for e in NAMES])
NAMES = NAMES[EXISTS]

fig, axes = plt.subplots(nrows=sum(EXISTS), ncols=10, figsize=(15, sum(EXISTS)*1.5))
for row_idx in range(axes.shape[0]):
    images = np.load(os.path.join(NAMES[row_idx], "train_images.npy"))
    axes[row_idx][0].set_ylabel(NAMES[row_idx])
    for col_idx in range(axes.shape[1]):
        axes[row_idx][col_idx].imshow(images[col_idx], cmap='gray' if images.shape[3] == 1 else None)
        axes[row_idx][col_idx].set_xlim([0, images.shape[1] - 1])
        axes[row_idx][col_idx].set_xticks([0, images.shape[1] - 1])
        axes[row_idx][col_idx].xaxis.tick_top()
        axes[row_idx][col_idx].set_ylim([images.shape[2] - 1, 0])
        axes[row_idx][col_idx].set_yticks([images.shape[2] - 1, 0])

plt.tight_layout()
plt.savefig("preview.png")
