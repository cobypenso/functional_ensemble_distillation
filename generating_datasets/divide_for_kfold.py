'''
This script responsible for dividing the entire dataset into 
Folds.
For example: Dataset with 10000 examples and 10-Fold 
---> [0:1000], [1000:2000], ..., [9000:10000] folds.

Then, save index list into file in the form of
bag <-> idxs. 
This will be used for Ensemble training and for auxiliary dataset creation
for the Generator training.

Run command:
CUDA_VISIBLE_DEVICES=0 python divide_for_kfold.py --n_folds 10 --fold kfold
--split 0.8 --data_path ../datasets/cifar10/ --dataset_name cifar10 --output_path ./idx_list.pkl
'''
import sys
sys.path.append('../')
import pickle
import argparse 
from data import *
from sklearn.model_selection import KFold, StratifiedKFold

parser = argparse.ArgumentParser(description='Generating New Dataset')
parser.add_argument('--n_folds', type=int, default=10)
parser.add_argument('--fold', type=str, default='kfold')
parser.add_argument('--split', type=float, default=0.8)
parser.add_argument('--data_path', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--output_path', type=str, default='idx_list.pkl')
args = parser.parse_args()

# load dataset
if args.dataset_name == 'cifar10':
    trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
elif args.dataset_name == 'stl10':
        trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split)
elif args.dataset_name == 'cifar100':
    trainset, validset, testset = get_cifar100(args.data_path, split = args.split)
elif args.dataset_name == 'pets':
    trainset, validset, testset = get_pets()
        
# pick folding strategy
if args.fold == 'kfold':
    kfold = KFold(n_splits = args.n_folds, shuffle = True, random_state = 1)
else:
    kfold = StratifiedKFold(n_splits = args.n_folds, shuffle = True, random_state = 1)

# divide
idx_list = list()
for train_ix, test_ix in kfold.split(trainset):
    idx_list.append((train_ix, test_ix))

# save idx list 
with open(args.output_path,'wb') as f:
    pickle.dump(idx_list, f)