'''
This script responsible for ensemble training using cSGHMC algorithm.
This code is almost identical to the original provided code of cSGHMC paper:
https://github.com/ruqizhang/csgmcmc
https://arxiv.org/abs/1902.03932

Usage:
    Regular training:
        CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 2000 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 3 --cycle 50 --samples 120 --threshold 0.1
    
    Kfold/Bagging training:
    The example here is for 10-Bagging: (specifically bag 0)
        CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 200 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 3 --cycle 50 --samples 12 --threshold 0.1 --bag 0 --idx_type bagging --idxs_path ./idxs_path.pkl
'''

import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from data import *
import argparse

from models import *
from torch.autograd import Variable
import numpy as np
import random
import pickle

parser = argparse.ArgumentParser(description='cSG-MCMC Training')
parser.add_argument('--checkpoints_dir', type=str, default=None, required=True, help='path to save checkpoints (default: None)') 
parser.add_argument('--idxs_path', type=str, default='idxs_path.pkl')
parser.add_argument('--idx_type', type=str, default='kfold')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--initial_lr', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--alpha', type=float, default=0.9,
                    help='1: SGLD; <1: SGHMC')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--test_every_epoch',type = bool, default=False)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1./50000,
                    help='temperature (default: 1/dataset_size)')
parser.add_argument('--arch', type=str, default="resnet18",
                    help='architecture name')
parser.add_argument('--split', type=float, default=0.8,
                    help='portion of trainset (out of train+valid)')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--bag', type=int, default=-1, help='bag number in case of bagging or KFold, -1 if not')
parser.add_argument('--isi', type=int, default=0, help='Inital Sample index - specify in case the samples are added to old ones')
parser.add_argument('--spc', type=int, default=3, help='samples per cycle')
parser.add_argument('--cycle', type=int, default=50, help='epochs per cycle')
parser.add_argument('--samples', type=int, default=120, help='samples in total')
parser.add_argument('--threshold', type=float, default=40)
args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Data
if args.dataset == 'cifar10':
    trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
elif args.dataset == 'stl10':
    trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split)
elif args.dataset == 'cifar100':
    trainset, validset, testset = get_cifar100(args.data_path, split = args.split)

# in case of Bagging or KFold

if args.bag != -1:
    # load idx file which describe the bags.
    idxs = pickle.load(open(args.idxs_path, 'rb'))[args.bag]
    if args.idx_type == 'kfold':
        idxs = idxs[0]
    else:
        pass # bagging - for readability
        
    train_subsampler = torch.utils.data.SubsetRandomSampler(idxs)
    shuffle = False
else:
    train_subsampler = None
    shuffle = True

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=args.batch_size, 
                                          shuffle=shuffle, 
                                          num_workers=4,
                                          sampler = train_subsampler)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=args.batch_size, 
                                         shuffle=False, 
                                         num_workers=0,
                                         sampler = None)

# Model
net = get_net(args.arch)

if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True

def update_params(lr,epoch):
    for p in net.parameters():
        if not hasattr(p,'buf'):
            p.buf = torch.zeros(p.size()).cuda(device_id)
        try:
            d_p = p.grad.data
        except:
            continue
        d_p.add_(weight_decay, p.data)
        buf_new = (1-args.alpha)*p.buf - lr*d_p
        if (epoch%cycle)+1>(cycle-samples_per_cycle - 2): # was - if (epoch%50)+1>(45)
            eps = torch.randn(p.size()).cuda(device_id)
            buf_new += (2.0*lr*args.alpha*args.temperature/datasize)**.5*eps
        p.data.add_(buf_new)
        p.buf = buf_new

def adjust_learning_rate(epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0
    return lr

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
        net.zero_grad()
        lr = adjust_learning_rate(epoch,batch_idx)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        update_params(lr,epoch)

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx%100==0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total)) 
                
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx%100==0:
                print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss/len(testloader), correct, total,
    100. * correct.item() / total))
    
    return (100. * correct.item() / total)

weight_decay = args.weight_decay
datasize = len(trainset)
if args.temperature == 0:
    args.temperature = len(trainset)
print('Temp:', args.temperature)
num_batch = datasize/args.batch_size+1
lr_0 = args.initial_lr # initial lr
M = int(args.epochs / args.cycle) # number of cycles
T = args.epochs*num_batch # total number of iterations
criterion = nn.CrossEntropyLoss()
mt = args.isi
counter = 0
cycle = args.cycle
samples_per_cycle = args.spc
print(args.threshold)
for epoch in range(args.epochs):
    train(epoch)
    if args.test_every_epoch:
        acc = test(epoch)
    if (epoch%cycle)+1>(cycle-samples_per_cycle): # save 3 models per cycle
        acc = test(epoch)
        print('with acc:',acc)
        net.cpu()
        if acc > args.threshold:
            torch.save(net.state_dict(),args.checkpoints_dir + '/cifar_csghmc_%i.pt'%(mt))
            mt += 1
            counter += 1
            print('save!')
        else:
            print('Skiped saving, accuracy was {0} at epoch {1}'.format(acc, epoch))
        net.cuda(device_id)
    if counter >= args.samples:
        print('Finished at epoch', epoch)
        break