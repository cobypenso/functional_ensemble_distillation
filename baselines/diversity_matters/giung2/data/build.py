import random
from typing import Dict
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fvcore.common.config import CfgNode
from giung2.data.datasets import *


def build_dataloaders(
        cfg: CfgNode,
        root: str = "./datasets/",
        num_replicas: int = None,
        rank: int = None,
        is_distributed: bool = False,
        drop_last: bool = False,
    ) -> Dict[str, DataLoader]:
    """
    Build the dictionary of dataloaders.

    Args:
        cfg (CfgNode) : configs.
        root (str) : root directory which contains datasets.

    Returns:
        dataloaders (dict) :
            its keys are ["dataloader", "trn_loader", "val_loader", "tst_loader"],
            and only "dataloader" returns examples with data augmentation.
    """
    assert cfg.DATASETS.NAME in [
        "MNIST", "FashionMNIST", "KMNIST",
        "CIFAR10", "CIFAR100", "CIFAR10_HMC",
        "TinyImageNet200",
        "ImageNet1k",
    ], f"Unknown cfg.DATASETS.NAME = \"{cfg.DATASETS.NAME}\""

    dataloaders = dict()

    if cfg.DATASETS.NAME in ["MNIST", "FashionMNIST", "KMNIST",]:

        # set data augmentation strategy
        from .datasets.mnist import DATA_AUGMENTATION
        trn_transform = DATA_AUGMENTATION[cfg.DATASETS.MNIST.DATA_AUGMENTATION]
        tst_transform = transforms.ToTensor()

        # split training and validation examples
        indices = list(range(60000))
        if cfg.DATASETS.MNIST.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)

        trn_indices = indices[cfg.DATASETS.MNIST.TRAIN_INDICES[0] : cfg.DATASETS.MNIST.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.MNIST.VALID_INDICES[0] : cfg.DATASETS.MNIST.VALID_INDICES[1]]

        # get datasets
        _MNIST = FastMNIST if cfg.DATASETS.MNIST.DATA_AUGMENTATION == "None" else MNIST
        dataset = _MNIST(root=root, name=cfg.DATASETS.NAME, split="train", indices=trn_indices, transform=trn_transform)
        trn_set = _MNIST(root=root, name=cfg.DATASETS.NAME, split="train", indices=trn_indices, transform=tst_transform)
        tst_set = _MNIST(root=root, name=cfg.DATASETS.NAME, split="test",  indices=None,        transform=tst_transform)
        val_set = _MNIST(root=root, name=cfg.DATASETS.NAME, split="train", indices=val_indices, transform=tst_transform) if val_indices else tst_set

    elif cfg.DATASETS.NAME in ["CIFAR10", "CIFAR100", "CIFAR10_HMC",]:

        # set data augmentation strategy
        from .datasets.cifar import DATA_AUGMENTATION
        trn_transform = DATA_AUGMENTATION[cfg.DATASETS.CIFAR.DATA_AUGMENTATION]
        tst_transform = transforms.ToTensor()

        # split training and validation examples
        indices = list(range(50000))
        if cfg.DATASETS.CIFAR.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)

        trn_indices = indices[cfg.DATASETS.CIFAR.TRAIN_INDICES[0] : cfg.DATASETS.CIFAR.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.CIFAR.VALID_INDICES[0] : cfg.DATASETS.CIFAR.VALID_INDICES[1]]

        # get datasets
        _CIFAR = FastCIFAR if cfg.DATASETS.CIFAR.DATA_AUGMENTATION == "None" else CIFAR
        dataset = _CIFAR(root=root, name=cfg.DATASETS.NAME, split="train", indices=trn_indices, transform=trn_transform)
        trn_set = _CIFAR(root=root, name=cfg.DATASETS.NAME, split="train", indices=trn_indices, transform=tst_transform)
        tst_set = _CIFAR(root=root, name=cfg.DATASETS.NAME, split="test",  indices=None,        transform=tst_transform)
        val_set = _CIFAR(root=root, name=cfg.DATASETS.NAME, split="train", indices=val_indices, transform=tst_transform) if val_indices else tst_set

    elif cfg.DATASETS.NAME in ["TinyImageNet200",]:

        # set data augmentation strategy
        from .datasets.tiny import DATA_AUGMENTATION
        trn_transform = DATA_AUGMENTATION[cfg.DATASETS.TINY.DATA_AUGMENTATION]
        tst_transform = transforms.ToTensor()

        # split training and validation examples
        indices = list(range(100000))
        if cfg.DATASETS.TINY.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)

        trn_indices = indices[cfg.DATASETS.TINY.TRAIN_INDICES[0] : cfg.DATASETS.TINY.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.TINY.VALID_INDICES[0] : cfg.DATASETS.TINY.VALID_INDICES[1]]

        # get datasets
        dataset = TinyImageNet200(root=root, split="train", indices=trn_indices, transform=trn_transform,)
        trn_set = TinyImageNet200(root=root, split="train", indices=trn_indices, transform=tst_transform,)
        tst_set = TinyImageNet200(root=root, split="test",  indices=None,        transform=tst_transform,)
        val_set = TinyImageNet200(root=root, split="train", indices=val_indices, transform=tst_transform,) if val_indices else tst_set

    elif cfg.DATASETS.NAME in ["ImageNet1k",]:

        # set data augmentation strategy
        from .datasets.imagenet import DATA_AUGMENTATION
        trn_transform = DATA_AUGMENTATION[cfg.DATASETS.IMAGENET.DATA_AUGMENTATION]
        tst_transform = DATA_AUGMENTATION["NONE"]

        # split training and validation examples
        indices = list(range(1281167))
        if cfg.DATASETS.IMAGENET.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)

        trn_indices = indices[cfg.DATASETS.IMAGENET.TRAIN_INDICES[0] : cfg.DATASETS.IMAGENET.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.IMAGENET.VALID_INDICES[0] : cfg.DATASETS.IMAGENET.VALID_INDICES[1]]

        # get datasets
        dataset = ImageNet1k(root=root, split="train", indices=trn_indices, transform=trn_transform,)
        trn_set = ImageNet1k(root=root, split="train", indices=trn_indices, transform=tst_transform,)
        tst_set = ImageNet1k(root=root, split="valid", indices=None,        transform=tst_transform,)
        val_set = ImageNet1k(root=root, split="train", indices=val_indices, transform=tst_transform,) if val_indices else tst_set

    # get dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=int(cfg.SOLVER.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False if cfg.NUM_GPUS > 1 else True,
        sampler=DistributedSampler(dataset, num_replicas, rank, shuffle=True) if cfg.NUM_GPUS > 1 else None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    trn_loader = DataLoader(
        dataset=trn_set,
        batch_size=int(cfg.SOLVER.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        sampler=DistributedSampler(trn_set, num_replicas, rank, shuffle=False) if is_distributed else None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=int(cfg.SOLVER.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        sampler=DistributedSampler(val_set, num_replicas, rank, shuffle=False) if is_distributed else None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    tst_loader = DataLoader(
        dataset=tst_set,
        batch_size=int(cfg.SOLVER.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        sampler=DistributedSampler(tst_set, num_replicas, rank, shuffle=False) if is_distributed else None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
    )

    dataloaders["dataloader"] = dataloader
    dataloaders["trn_loader"] = trn_loader
    dataloaders["val_loader"] = val_loader
    dataloaders["tst_loader"] = tst_loader

    return dataloaders
