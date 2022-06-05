
import torch
import random 
import torchvision
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets import DatasetFolder, ImageFolder
from torch.utils.data import TensorDataset, DataLoader, Dataset

'''
This file contains handling the data on which the experiments are done on.
'''
def mixup(org_img, mix_img, alpha=0.2, use_cuda=True):
    # Mixup two images (without carrying about labels)
    lam = np.random.beta(alpha, alpha)
    image = lam * org_img + (1 - lam) * mix_img
    return image, lam

def get_cinic10(data_path):
    '''
    Returns train, valid, test, train+valid dataloaders.
    '''
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trainset = torchvision.datasets.ImageFolder(data_path + '/train',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    # cinic_train = torch.utils.data.DataLoader(trainset,
    #                                           batch_size=128,
    #                                           shuffle=True)
    validset = torchvision.datasets.ImageFolder(data_path + '/valid',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)]))                                              
    # cinic_valid = torch.utils.data.DataLoader(validset,
    #                                           batch_size=128, 
    #                                           shuffle=False)
    testset = torchvision.datasets.ImageFolder(data_path + '/test',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)]))                                              
    # cinic_test = torch.utils.data.DataLoader(testset,
    #                                          batch_size=128, 
    #                                          shuffle=False)
    
    train_and_valid = ConcatDataset([trainset, validset])
    # cinic_train_valid = torch.utils.data.DataLoader(train_and_valid,
    #                                           batch_size=128,
    #                                           shuffle=True)

    return trainset, validset, testset, train_and_valid

# Data
def get_cifar10(data_path, split = 0.7): 
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    trainset, validset = torch.utils.data.random_split(trainset, [int(len(trainset)*split), len(trainset)-int(len(trainset)*split)], generator=torch.Generator().manual_seed(42))
    return trainset, validset, testset


def get_stl10(data_path, split = 0.7, train_transform = True): 
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    
    if train_transform:
        trainset = torchvision.datasets.STL10(root=data_path, split='train', download=True, transform=transform_train)
    else:
        trainset = torchvision.datasets.STL10(root=data_path, split='train', download=True, transform=transform_test)
    testset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transform_test)
    unlabeledset = torchvision.datasets.STL10(root=data_path, split='unlabeled', download=True, transform=transform_test)

    trainset, validset = torch.utils.data.random_split(trainset, [int(len(trainset)*split), len(trainset)-int(len(trainset)*split)], generator=torch.Generator().manual_seed(42))
    return trainset, validset, testset, unlabeledset


def get_cifar100(data_path, split = 0.7): 
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    trainset, validset = torch.utils.data.random_split(trainset, [int(len(trainset)*split), len(trainset)-int(len(trainset)*split)], generator=torch.Generator().manual_seed(42))

    return trainset, validset, testset

def get_lsun(data_path, dataset='bedroom_test'):
    """LSUN dataloader with (128, 128) sized images.
    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Get dataset
    lsun_dset = torchvision.datasets.LSUN(root=data_path, classes=[dataset], transform=transform)
    return lsun_dset


def get_svhn(data_path, split = 0.7, train_transform = True): 
    print('==> Preparing data..')
    training_transform_augmented = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(size=(32, 32)),
        transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

    training_transform_not_augmented = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

    if train_transform:
        training_transform = training_transform_augmented
    else:
        training_transform = training_transform_not_augmented

    trainset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=training_transform)
    trainset, validset = torch.utils.data.random_split(trainset, [int(len(trainset)*split), len(trainset)-int(len(trainset)*split)], generator=torch.Generator().manual_seed(42))
    testset = torchvision.datasets.SVHN(root=data_path, split='test', download=True, transform=training_transform_not_augmented)
    extraset = torchvision.datasets.SVHN(root=data_path, split='extra', download=True, transform=training_transform_not_augmented)
    return trainset, validset, testset, extraset


def get_celebA(data_path): 
    print('==> Preparing data..')
    transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       #transforms.Grayscale(),                                       
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])

    trainset = torchvision.datasets.CelebA(root=data_path, split='train', download=True, transform=transform)
    validset = torchvision.datasets.CelebA(root=data_path, split='valid', download=True, transform=transform)
    testset = torchvision.datasets.CelebA(root=data_path, split='test', download=True, transform=transform)
    all = torchvision.datasets.CelebA(root=data_path, split='all', download=True, transform=transform)
    return trainset, validset, testset, all


def loader_func(path: str):
    return torch.load(path)


class MyDataset(DatasetFolder):
    def __init__(self, root_dir, transform=None): 
        super(MyDataset, self).__init__(root = root_dir, 
                                        loader = loader_func, 
                                        extensions = ('.pt'),
                                        transform=None)
        self.custom_transform = transform
        self.root_dir = root_dir

    def __getitem__(self, index):
        x, label = super(MyDataset, self).__getitem__(index)

        if len(x) == 2:
            image, probs = x
        else:
            image, probs, label = x
        if self.custom_transform:
            image = self.custom_transform(image)
        
        if len(x) == 2:
            return image, probs
        else:
            return image, probs, label

        
class CustomTensorDataset(Dataset):
    """
        TensorDataset with support of transforms and mixup.
    """
    def __init__(self, dataset, transform=None, use_mixup = False, mixup_factor = 2):
        self.dataset = dataset
        self.transform = transform
        self.use_mixup = use_mixup
        self.mixup_factor = mixup_factor

    def __getitem__(self, index):
        org_idx = index
        index = index % len(self.dataset) #Keep it in boundries
        item = self.dataset[index]
        x = item[0]
        y = item[1:]
        if self.transform:
            x = self.transform(x)

        #Do mixup 1/mixup_factor of the time
        if int(org_idx / len(self.dataset)) >= 1 and self.use_mixup:
            
            # if self.use_mixup and random.randint(1,self.mixup_factor) == 1:
            idx = random.randint(0, len(self.dataset) - 1)
            ref_item = self.dataset[idx]
            if self.transform:
                x_ref = self.transform(ref_item)
            x, lam = mixup(x, ref_item[0])
            #if Lam less than 0.5, then the second pic is more dominant
            #  (so change the label accordingly)
            if lam < 0.5:
                y = ref_item[1:]
                
        return (x, *y)

    def __len__(self):
        if self.use_mixup:
            return (self.mixup_factor+1) * len(self.dataset)
        else:
            return len(self.dataset)

def save_dataset(dataset, path):
    torch.save(dataset, path)

def load_dataset(path):
    dataset = torch.load(path)
    return dataset