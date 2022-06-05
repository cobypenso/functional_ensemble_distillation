'''
In this script:

Create dataset for Generator training.

1. For every example in the trainset:
    1.1 Find in which models in the ensemble it was used as held-out
    1.2 Predict using the models from 1.1
    1.3 add the example and it's probability vectors to the new dataset in creation
2. Save dataset 

Note: In this case, Bagging used for the dataset division
Note: For Bagging - dataset saved in folder.
      For K-Fold - dataset saved in file.

Usage:
CUDA_VISIBLE_DEVICES=0 python bagging_dataset_creator.py --models_per_bag 12 --split 0.8
 --data_path ../datasets/cifar10/ --idx_path ./idx_list.pkl --arch resnet18_gn_ws_cifar10
 --std kfold --dataset_name cifar10 --output_path kfold_ds.pt
'''

import sys
sys.path.append('../')
import pickle
import argparse
from data import *
from utils import *
from models import *
from numpy import arange
from CSGHMC.cyclic_sghmc_predict import csghmc_predict_with_idxs

#######################################################################

parser = argparse.ArgumentParser(description='Generating New Dataset')
parser.add_argument('--models_per_bag', type=int, default=12)
parser.add_argument('--split', type=float, default=0.8)
parser.add_argument('--data_path', type=str, default = '../dataset/cifar10', help = 'path to dataset')
parser.add_argument('--idx_path', type=str, default = 'idx_list.pkl', help='for example idx_list.pkl')
parser.add_argument('--arch', type=str, default='resnet18_gn_ws_cifar10')
parser.add_argument('--stg', type=str, default='kfold', choices=['kfold, bagging'])
parser.add_argument('--dataset_name', type=str, default='cifar10')
parser.add_argument('--output_path', type=str, default = 'kfold_ds.pt', help='path for new dataset')

# used to speed-up the creation by dividing the dataset to parts and running several scipts in parallel.
# If no parallelization needed, set part = 0 and part_size = #dataset (dataset size)
parser.add_argument('--part', type=int, default=0)
parser.add_argument('--part_size', type=int, default=5000)

parser.add_argument('--checkpoints_dir', type=str, help = 'path to the ensemble')
args = parser.parse_args()

#######################################################################

def find_bags(example_idx, idxs_list):
    '''
    Find in which bag the example used as testset
    '''
    bags = []
    for i in range(len(idxs_list)):
        current_list = idxs_list[i]
        # TODO - make sure correct behaviour in KFold scenario
        if len(current_list) == 2: # means that we are working with KFold (not bagging)
            current_list = current_list[0]
        if not (example_idx in current_list):
            bags.append(i)
    return bags

##############################################################################################

def find_models(example_idx, idx_list):
    '''
    find model indexs to use for the specific example
    '''
    bags = find_bags(example_idx, idx_list)
    model_idxs = []
    for item in bags:
        model_idxs.extend(arange(item * args.models_per_bag, (item + 1) * args.models_per_bag))

    return model_idxs

##############################################################################################

def create_dataset_for_gm_training_with_bagging(predictions, dataset, batch_size, dataset_path, idx_start):

    loader = DataLoader(dataset, batch_size=len(dataset))

    images = next(iter(loader))[0]
    labels = next(iter(loader))[1]
    
    for i in range(len(predictions)):
        idx = idx_start + i
        torch.save((images[i].clone(), predictions[i].cpu().clone(), labels[i].item()), dataset_path + '/all/' + str(idx) + '.pt')


##############################################################################################

# load dataset
if args.dataset_name == 'cifar10':
    trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
elif args.dataset_name == 'stl10':
    trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split, train_transform = False)
elif args.dataset_name == 'cifar100':
    trainset, validset, testset = get_cifar100(args.data_path, split = args.split)
    
net = get_net(args.arch)

idx_list = pickle.load(open(args.idx_path, 'rb'))
dataset = CustomTensorDataset(trainset, None, use_mixup = False)
predictions = []
min_val = 1000 # set to high value, just to get updated to new min in the first iteration.

not_included = 0 
dataset_new = []

# parralelize
if len(dataset) % args.part_size != 0:
    raise Exception()
idx_start, idx_end = args.part * args.part_size , (args.part + 1) * args.part_size
if len(dataset) < idx_end:
    raise Exception()

# predict
for i in range(idx_start, idx_end):
    model_idxs = find_models(i, idx_list)
    if len(model_idxs) < min_val:
        min_val = len(model_idxs)
    if len(model_idxs) == 0:
        not_included += 1
        continue

    dataset_new.append(dataset[i])
    prediction = csghmc_predict_with_idxs(net, dataset[i], args.checkpoints_dir, model_idxs, True, verbose=False)['full_predictions']
    prediction = torch.cat(prediction).view(len(model_idxs), -1)
    predictions.append(prediction)

# if Bagging - save to folder. if KFold - save to file.
if args.stg == 'bagging':
    create_dataset_for_gm_training_with_bagging(predictions, dataset_new, 8, args.output_path, idx_start)
else:
    gen_dataset, gen_dataloader = create_dataset_for_gm_training(predictions, dataset_new, 8, args.output_path, format = 'nmc')
    
print("Dataset created!")
print("Not included", not_included) # not_included is the number of examples which did not included in the new dataset. Can happend due the stochastic nature of Bagging split.
print("New dataset length", len(dataset_new))