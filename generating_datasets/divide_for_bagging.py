'''
This script responsible for dividing the entire dataset into 
bags using Bagging partition.
For example: Dataset with 10000 examples and 10-Bagging 
---> Corresponds to 10 bags, each of the size 10000, with examples
that sampled with repetition from the original dataset

Then, save index list into file in the form of
bag <-> idxs. 
This will be used for Ensemble training and for auxiliary dataset creation
for the Generator training.

Run command:
CUDA_VISIBLE_DEVICES=0 python divide_for_bagging.py --bags 10 --split 0.8
 --data_path ../datasets/cifar10/ --dataset_name cifar10 --output_path ./idx_list.pkl
'''


import sys
sys.path.append('../')
import pickle
import argparse 
from data import *
from random import randrange
from random import random, sample

parser = argparse.ArgumentParser(description='Generating New Dataset')
parser.add_argument('--bags', type=int, default=10)
parser.add_argument('--split', type=float, default=0.8)
parser.add_argument('--data_path', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--output_path', type=str, default='idx_list.pkl')
args = parser.parse_args()


##############################################################

def subsample(dataset, ratio = 1.0):
    sample_idx = []
    sample = []
    n_sample = round(len(dataset) * ratio)
    while len(sample_idx) < n_sample:
        index = randrange(len(dataset))
        sample_idx.append(index)
        sample.append(dataset[index])
    return sample, sample_idx

##############################################################

# load dataset
if args.dataset_name == 'cifar10':
    trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
elif args.dataset_name == 'stl10':
        trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split)
elif args.dataset_name == 'cifar100':
    trainset, validset, testset = get_cifar100(args.data_path, split = args.split)
        
# divide into args.bags
idx_list = list()
for i in range(args.bags):
    _, sample_idx = subsample(trainset, ratio=1.0)
    idx_list.append(sample_idx)

# save idx list 
with open(args.output_path,'wb') as f:
    pickle.dump(idx_list, f)