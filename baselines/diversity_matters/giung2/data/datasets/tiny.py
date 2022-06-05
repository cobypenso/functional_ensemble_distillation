import os
import numpy as np
import torchvision
from PIL import Image
from itertools import chain
from tabulate import tabulate
from typing import Any, Callable, List, Optional, Tuple


__all__ = [
    "TinyImageNet200",
]


DATA_AUGMENTATION = {
    "NONE": torchvision.transforms.ToTensor(),
    "STANDARD_TRAIN_TRANSFORM": torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(64, 4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]),
}


class TinyImageNet200(torchvision.datasets.VisionDataset):

    def __init__(
            self,
            root: str,
            split: str = "train",
            indices: List[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ) -> None:

        assert split in ["train", "test"]

        if indices is None:
            indices = list(range(100000)) if split == "train" else list(range(10000))

        super(TinyImageNet200, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.images  = np.load(os.path.join(root, f"TinyImageNet200/{split}_images.npy"))[indices]
        self.labels  = np.load(os.path.join(root, f"TinyImageNet200/{split}_labels.npy"))[indices]
        self.classes = list(range(200))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = self.images[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)

    def describe(self) -> str:
        NUM_COL = 5
        DATA = [[idx,len([e for e in self.labels if e == idx]),] for idx in self.classes]
        DATA.append(["Total", sum(e[1] for e in DATA)])
        DATA = [
            list(chain(*e)) for e in zip(
                *[(DATA + [["-", "-"]] * (NUM_COL - (len(DATA) % NUM_COL)))[i::NUM_COL] for i in range(NUM_COL)]
            )
        ]
        return tabulate(DATA, headers=["Class", "# Examples",] * NUM_COL)
