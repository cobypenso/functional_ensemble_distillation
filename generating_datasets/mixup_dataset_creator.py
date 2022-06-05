'''
This script responsible for creating a Mixup auxiliary dataset.
Used for Generator training.

Two options for mixup dataset creation:
1. Regular mixup examples
2. Hard mixup examples - i.e only mixup example with high entropy in their probability vectors.

Usage:
CUDA_VISIBLE_DEVICES=0 python mixup_dataset_creator.py --data_path ../datasets/cifar10
--split 0.8 --dataset_size 150000 --dataset_path ./mixup.pt --ensemble_size 120
--arch resnet18_gn_ws_cifar10 --checkpoints_dir ../checkpoints_CIFAR10/
'''

import sys
sys.path.append('../')
import argparse
import wandb
from data import *
from models import *
from CSGHMC.cyclic_sghmc_predict import csghmc_predict_unsupervised, csghmc_predict
from classification_generators.generators import *
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from scipy.stats import entropy

# ------------------------------------------------------------

def create_dataset(predictions, images, dataset_path, idx_list):
    ensemble_size = predictions[0].shape[0]
    # predictions = torch.cat(predictions,0).view(len(predictions), -1, 10)
    predictions = torch.cat(predictions,0).view(len(predictions), ensemble_size, -1)
    # predictions [M, N, classes] #
    predictions = predictions.permute(1,0,2)
    predictions = predictions[:, idx_list, :]
    # predictions [N, M, classes] #
    new_dataset = TensorDataset(images.cpu(), predictions.cpu()) # create your datset
    torch.save(new_dataset, dataset_path)
    return new_dataset

# ------------------------------------------------------------

def create_dataset_folder(predictions, images, dataset_path, idx_list, initial_idx = 0):
    ensemble_size = predictions[0].shape[0]
    # predictions = torch.cat(predictions,0).view(len(predictions), -1, 10)
    predictions = torch.cat(predictions,0).view(len(predictions), ensemble_size, -1)
    # predictions [M, N, classes] #
    predictions = predictions.permute(1,0,2)
    predictions = predictions[:, idx_list, :]
    # predictions [N, M, classes] #
    new_dataset = TensorDataset(images.cpu(), predictions.cpu()) # create your datset
    
    for i in range(len(new_dataset)):
        x,y = new_dataset[i]
        x_copy = x.clone()
        y_copy = y.clone()
        torch.save((x_copy, y_copy), args.dataset_path + '/all/' + str(i + initial_idx) + '.pt')


# ------------------------------------------------------------

def entropy_calc(dataset):
    entropy_list = []
    for i in range(len(dataset)):
        entropy_list.append(entropy(dataset[i]))
    return entropy_list

# ------------------------------------------------------------

def create_hard_mixup_dataset(net, dataset, args, idx_list, size = None):
    if size == None:
        size = len(dataset)

    images = np.zeros((2 * size, *dataset[0][0].shape))
    i = 0

    while (2*size) != i:
        idx1 = random.randint(0, len(dataset) - 1)
        idx2 = random.randint(0, len(dataset) - 1)
        if idx1 == idx2:
            continue
        img, lam = mixup(dataset[idx1][0], dataset[idx2][0], alpha=0.2, use_cuda=True)
        if lam > 0.9 or lam < 0.1:
            continue
        images[i] = img
        i += 1
    
    predictions = csghmc_predict_unsupervised(net, images, args.checkpoints_dir, args.ensemble_size, True)
    full_predictions = predictions['full_predictions']
    ensemble_softmax = predictions['ensemble_softmax'].cpu()

    entropy_list = entropy_calc(ensemble_softmax)

    _, indices = torch.Tensor(entropy_list).topk(size)

    images_filtered = np.zeros((size, *dataset[0][0].shape))
    j = 0
    for i in indices:
        images_filtered[j] = images[i][0]
    
    full_predictions = [i[indices] for i in full_predictions]
    create_dataset(full_predictions, torch.from_numpy(images_filtered), args.dataset_path, idx_list)

# ------------------------------------------------------------

def create_mixup_dataset(net, dataset, args, idx_list, size = None, initial_idx = 0):
    if size == None:
        size = len(dataset)

    images = np.zeros((size, *dataset[0][0].shape))
    i = 0

    while (size) != i:
        idx1 = random.randint(0, len(dataset) - 1)
        idx2 = random.randint(0, len(dataset) - 1)
        if idx1 == idx2:
            continue
        img, lam = mixup(dataset[idx1][0], dataset[idx2][0], alpha=0.2, use_cuda=True)
        images[i] = img.cpu() 
        i += 1
    
    
    predictions = csghmc_predict_unsupervised(net, images, args.checkpoints_dir, args.ensemble_size, True)
    full_predictions = predictions['full_predictions']
    
    create_dataset_folder(full_predictions, torch.from_numpy(images), args.dataset_path, idx_list, initial_idx)

# ------------------------------------------------------------



def main(gpu = None, args = None):
    torch.cuda.set_device(gpu)
    args.local_rank = gpu
    if args.dataset_name == 'cifar10':
        trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
    elif args.dataset_name == 'stl10':
        trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split, train_transform = False)
    elif args.dataset_name == 'cifar100':
        trainset, validset, testset = get_cifar100(args.data_path, split = args.split)
    
    net = get_net(args.arch)
    
    if args.dataset == 'trainset':
        dataset = trainset
    else:
        dataset = validset
    
    dataset = CustomTensorDataset(dataset, None, use_mixup = False)
    idx_list = np.arange(args.ensemble_size)
    if args.hard_mixup:
        create_hard_mixup_dataset(net, trainset, args, idx_list, size = args.dataset_size)
    else:
        create_mixup_dataset(net, trainset, args, idx_list, size = args.dataset_size, initial_idx = args.initial_idx)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generator Training')
    parser.add_argument('--checkpoints_dir', type=str, default=None, required=True, help='path to save checkpoints (default: None)')
    parser.add_argument('--arch', type=str, default="resnet18",
                        help='architecture name')
    parser.add_argument('--ensemble_size', type=int, default=12)
    parser.add_argument('--data_path', type=str, default=None, required=False, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--split', type=float, default=0.8,
                        help='portion of trainset (out of train+valid)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument("--dataset_path", type=str, default='mixup.pt')
    parser.add_argument("--dataset", type=str, default='trainset')
    parser.add_argument("--dataset_size", type=int, default = 50000)
    parser.add_argument("--hard_mixup", type=bool, default = False)
    parser.add_argument('--initial_idx', type=int, default=0)
    args = parser.parse_args()

    
    gpu = torch.cuda.current_device()
    args.world_size = 1
    main(gpu = gpu, args = args)


