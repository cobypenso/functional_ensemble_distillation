'''
This script responsible for cSGHMC ensemble evaluation.
This code is almost identical to the original provided code of cSGHMC paper:
https://github.com/ruqizhang/csgmcmc
https://arxiv.org/abs/1902.03932

Usage:
    CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_eval.py --checkpoints_dir ../checkpoints_CIFAR10/ 
    --data_path ../datasets/cifar10/ --num_samples 120 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset_name cifar10
    --acc_threshold 0.1 --dataset testset
    
'''

import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
from data import *
from models import *
from torch.autograd import Variable
import numpy as np
import random

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Ensemble')
parser.add_argument('--checkpoints_dir', type=str, default=None, required=True, help='path to checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--arch', type=str, default="resnet18_gn_ws_cifar10", help='architecture name')
parser.add_argument('--num_samples', type=int, default=12, help='number of samples from posterior')
parser.add_argument('--split', type=float, default=0.8, help='portion of trainset (out of train+valid)')
parser.add_argument('--dataset_name', type=str, default='cifar10')
parser.add_argument('--acc_threshold', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default='testset', help = 'on which dataset part to eval')                                     

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Data
print('==> Preparing data..')
if args.dataset_name == 'cifar10':
    trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
elif args.dataset_name == 'stl10':
    trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split, train_transform = False)
elif args.dataset_name == 'cifar100':
    trainset, validset, testset = get_cifar100(args.data_path, split = args.split)
    
if args.dataset == 'trainset':
    testloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=0)
elif args.dataset == 'validset':
    testloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False, num_workers=0)
elif args.dataset == 'testset':
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
net = get_net(args.arch)

if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True

criterion = nn.CrossEntropyLoss()

def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
             right += 1.0
    return right/len(truth)

def test(loader = testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_list = []
    truth_res = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            truth_res += list(targets.data)
            outputs = net(inputs)
            pred_list.append(F.softmax(outputs,dim=1))
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss/len(testloader), correct, total, 100. * correct.item() / total))
    pred_list = torch.cat(pred_list,0)
    return pred_list,truth_res, (correct.item() / total)

pred_list = []
num_model = args.num_samples
isi = 0
acc_list = []
for m in range(num_model):
    if not os.path.isfile(args.checkpoints_dir + '/cifar_csghmc_%i.pt'%(m)):
            continue
    net.load_state_dict(torch.load(args.checkpoints_dir + '/cifar_csghmc_%i.pt'%(m + isi)))
    pred, truth_res, acc = test(loader = testloader)
    # --- add only if pass theshold --- #
    if acc >= args.acc_threshold:
        print ('added')
        acc_list.append(acc)
        pred_list.append(pred)
    # --------------------------------- #
    
print('ensemble size', len(pred_list))
fake = sum(pred_list)/len(pred_list)
values, pred_label = torch.max(fake,dim = 1)
pred_res = list(pred_label.data)
acc = get_accuracy(truth_res, pred_res)
print('Accuracy:',acc)
print('Individual Accuracy: ', acc_list)