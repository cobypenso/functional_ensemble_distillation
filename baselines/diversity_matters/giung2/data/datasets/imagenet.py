import os
import torchvision
from itertools import chain
from tabulate import tabulate
from typing import Callable, List, Optional


__all__ = [
    "ImageNet1k",
]


DATA_AUGMENTATION = {
    "NONE": torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]),
    "STANDARD_TRAIN_TRANSFORM": torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]),
}


class ImageNet1k(torchvision.datasets.ImageFolder):

    def __init__(
            self,
            root: str,
            split: str = "train",
            indices: List[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ) -> None:

        assert split in ["train", "valid"]

        if split == "train":
            root = os.path.join(root, "ImageNet1k/train/")
        elif split == "valid":
            root = os.path.join(root, "ImageNet1k/val/")

        if indices is None:
            indices = list(range(1281167)) if split == "train" else list(range(50000))

        super(ImageNet1k, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.samples = [self.samples[idx] for idx in indices]
        self.targets = [self.targets[idx] for idx in indices]

    def describe(self) -> str:
        NUM_COL = 5
        DATA = [[self.class_to_idx[c], len([e for e in self.targets if e == self.class_to_idx[c]]),] for c in self.classes]
        DATA.append(["Total", sum(e[1] for e in DATA)])
        DATA = [
            list(chain(*e)) for e in zip(
                *[(DATA + [["-", "-"]] * (NUM_COL - (len(DATA) % NUM_COL)))[i::NUM_COL] for i in range(NUM_COL)]
            )
        ]
        return tabulate(DATA, headers=["Class", "# Examples",] * NUM_COL)
