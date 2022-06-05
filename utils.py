import os
import dill
import torch
import random 
import numpy as np
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import torchvision.transforms as transforms
from data import CustomTensorDataset, MyDataset
from torch.utils.data import TensorDataset, DataLoader

##############################################################################################

def create_dataset_for_gm_training(predictions, dataset, batch_size, dataset_path, format = 'mnc'):
    '''
        This function responsible for creating the auxiliry dataset for the generator training.
        The dataset will be in the form of a .pt file.
        Saving TensorDatase. Each element in the form: (image, prob vectors, label)

        @param predictions - probability vectors from the ensemble to be distilled.
        @param dataset - (image,label).
        @param dataset_path - path to which to save the new dataset.
        @param batch_size - batch size, for the dataloader that wrap the new dataset.
        @param format - 'mnc' or 'nmc' (m - models, n - num of examples, c - num of classes) 

        @note Important that the dataset and predictions are synced in their order of elements!!
    '''
    
    label_size = predictions[0].shape[-1]
    if type(predictions) != torch.Tensor:
        predictions = torch.cat(predictions,0).view(len(predictions), -1, label_size)
    # predictions [M, N, C] #
    if format == 'mnc':
        predictions = predictions.permute(1,0,2)
    # predictions [N, M, C] #
    loader = DataLoader(dataset, batch_size=len(dataset))
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    if len(next(iter(loader))) == 3:
        labels = next(iter(loader))[2]
    else:
        labels = next(iter(loader))[1]

    images = next(iter(loader))[0]
    new_dataset = TensorDataset(images, predictions, labels) # create your datset
    dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    torch.save(new_dataset, dataset_path)
    
    return new_dataset, dataloader


##############################################################################################

def load_dataset_for_gm_training(batch_size, dataset_path, dataset_type = 'augment', drop_last = True):
    '''
        This function responsible for loading the auxiliry dataset for the generator training.
        The dataset in the form of a .pt file.
        Each element in the form: (image, prob vectors, label)

        @param batch_size - batch size for the dataloader
        @param dataset_path - path to load the dataset from
        @param dataset_type - if 'augment' then apply augmentation. o.w dont.
        @param drop_last
    '''
    dataset = torch.load(dataset_path, map_location=torch.device('cpu'))

    if dataset_type == 'augment':
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip()
        ])
    else:
        transform = None

    # wrap dataset loaded from file with transforms.
    ds = CustomTensorDataset(dataset, transform) 
    dataloader = DataLoader(ds, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            drop_last=drop_last, 
                            num_workers = 4,
                            pin_memory = False,
                            prefetch_factor = 8,
                            persistent_workers = True)
    return ds, dataloader


    
##############################################################################################

def collate_fn_custom(batch):
    sample = [torch.zeros((len(batch),*batch[0][0].shape)), [], []]
    for i in range(len(batch)):        
        sample[0][i] = batch[i][0]
        sample[1].append(batch[i][1])
        sample[2].append(batch[i][2])
    return sample

def load_dataset_for_gm_training_from_folder(batch_size, dataset_path, dataset_type = 'augment', bagging = False):
    '''
        This function responsible for loading the auxiliry dataset for the generator training.
        The dataset in the form of a folder with file per example.
        Each example is in the form: (image, prob vectors, label)

        @param batch_size - batch size for the dataloader
        @param dataset_path - path to load the dataset from
        @param dataset_type - if 'augment' then apply augmentation. o.w dont.
        @param bagging - if True, then this dataset created using bagging technique. i.e each example has different
                         number of probability vectors. Thus custom collate_fn is needed for the dataloader.
    '''
    
    if dataset_type == 'augment':
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
        # transforms.GaussianBlur(kernel_size=(3), sigma=(0.1, 0.5)),
        ])
    else:
        transform = None

    dataset_with_aug = MyDataset(dataset_path, transform)
    if bagging:
        collate_fn = collate_fn_custom
    else:
        collate_fn = None
        
    dataloader = DataLoader(dataset_with_aug, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            drop_last=True, 
                            num_workers = 2,
                            pin_memory = False,
                            prefetch_factor = 4,
                            persistent_workers = True,
                            collate_fn = collate_fn)
                            
    return dataset_with_aug, dataloader

##############################################################################################

def get_optimizer(opt_type, lr, wd, model, ckpt):
    '''
    get optimizer
    @param opt_type - optimizer type ['Adam', 'RMSProp', 'SGD']
    @param lr, wd - optimizer params
    @param model - the model the optimizer will optimize.
    @param ckpt - checkpoint path, to load optimizer state dict. None if no checkpoint is used.
    '''
    if opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=wd)

    if ckpt:
        optimizer.load_state_dict(torch.load(ckpt))

    return optimizer

##############################################################################################

def get_scheduler(optimizer, schedule = 'f1', last_epoch = -1):
    '''
        get scheduler
        @param optimizer - optimizer to schedule.
        @param schedule - specify the scheduling procedure.
        @param last_epoch - in case we continue from specific epoch. 
        
    '''    
    if schedule == 'f1':
        milestones = [60, 80, 120, 150]
    elif schedule == 'f2':
        milestones = [50, 65, 80, 100]
    elif schedule == 'f3':
        milestones = [35, 45, 55, 70, 80]
    else:
        pass
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=milestones,
                                                        gamma=0.33,
                                                        last_epoch = last_epoch) 
    return lr_scheduler


##############################################################################################
################################     Util functions           ################################
##############################################################################################

def save_to_file(data_to_store, path):
    with open(path, mode= 'wb') as f:
	    dill.dump(data_to_store, f)

def upload_from_file(path):
    with open(path, 'rb') as f:
        data = dill.load(f)
        return data

def save_model(file_name, model):
    torch.save(model.state_dict(), file_name + ".pt")

def upload_model(file_name, model):
    model.load_state_dict(torch.load(file_name))
    return model

def imshow(img, save_name = 'img_show.png'):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(npimg,  cmap='gray')
    #fig.show(figsize=(1,1))
    
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(npimg.reshape(28,28),  cmap='gray', interpolation='nearest')
    plt.savefig(save_name)

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def initialize_tensor(
        tensor: torch.Tensor,
        initializer: str,
        init_values: List[float] = [],
    ) -> None:

    if initializer == "zeros":
        nn.init.zeros_(tensor)

    elif initializer == "ones":
        nn.init.ones_(tensor)

    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])

    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])

    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0 * init_values[1] * torch.bernoulli(
                    torch.zeros_like(tensor) + init_values[0]
                ) - init_values[1]
            )

    else:
        raise NotImplementedError(
            f"Unknown initializer: {initializer}"
        )
