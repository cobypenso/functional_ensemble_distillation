'''
In this script:

Create dataset for Generator training.

1. For every example in the trainset:
    1.1 pass through the ensemble to get probability vectors (or logits vectors)
    1.3 add the new dataset in creation
2. Save dataset 

Note: The new dataset is in the form of .pt file

Usage:
CUDA_VISIBLE_DEVICES=0 python dataset_creator.py --gen_ds_train_path ./train.pt
 --gen_ds_test_path ./test.pt --arch resnet18_gn_ws_cifar10 --checkpoints_dir ../checkpoints_CIFAR10/
 --data_path ../datasets/cifar10/ --split 0.8 --ensemble_size 120
 --dataset trainset --dataset_name cifar10 --acc_threshold 0
'''

import sys
sys.path.append('../')
import argparse
from data import *
from utils import *
from models import *
from torch.utils.data import DataLoader
from CSGHMC.cyclic_sghmc_predict import csghmc_predict, csghmc_predict_unsupervised


parser = argparse.ArgumentParser(description='Generating New Dataset')
parser.add_argument('--gen_ds_train_path', type=str, default='./train.pt')
parser.add_argument('--gen_ds_test_path', type=str, default='./test.pt')
parser.add_argument('--arch', type=str, default='resnet18_gn_ws_cifar10')
parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints_CIFAR10/')
parser.add_argument('--data_path', type=str, default='../datasets/cifar10/')
parser.add_argument('--ensemble_size', type=int, default=120)
parser.add_argument('--split', type=float, default=0.8)
parser.add_argument('--dataset', type=str, default='trainset')
parser.add_argument('--dataset_name', type=str, default='cifar10')
parser.add_argument('--acc_threshold', type=float, default=0.1) 
parser.add_argument('--save_only_train', type=bool, default=False)
parser.add_argument('--logits', type=bool, default=False)
parser.add_argument('--bs', type=int, default=256)
args = parser.parse_args()

# Choose workon dataset
if args.dataset_name == 'cifar10':
    trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
elif args.dataset_name == 'stl10':
    trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split, train_transform = False)
elif args.dataset_name == 'cifar100':
    trainset, validset, testset = get_cifar100(args.data_path, split = args.split)
        

# load network
net = get_net(args.arch)

# choose if to predict on trainset or validset
dataset = (validset if args.dataset == 'validset' else trainset)


dataset = CustomTensorDataset(dataset, None, use_mixup = False)

if args.dataset == 'unlabeled':
    # used for STL10 unlabeled dataset
    dataset = unlabeledset
    predictions_ = csghmc_predict_unsupervised(net, unlabeledset, args.checkpoints_dir, args.ensemble_size, True, batch_size = args.bs) 
else:
    # all other cases - where the data is labeled (train/val/test)
    predictions_ = csghmc_predict(net, dataset, args.checkpoints_dir, args.ensemble_size, True, batch_size = args.bs)


# Pick logits or prob vectors to save in the new dataset
# logits is used for EnDD baseline support. 
if not args.logits:
    predictions = predictions_['full_predictions']
else:
    predictions = predictions_['full_logits']

# move predictions to cpu
for i in range(len(predictions)):
    predictions[i] = predictions[i].cpu()

# In case testset creation is needed 
if not args.save_only_train:
    test_predictions = csghmc_predict(net, testset, args.checkpoints_dir, args.ensemble_size, True,  batch_size = args.bs)
    if not args.logits:
        test_predictions = test_predictions['full_predictions']
    else:
        test_predictions = test_predictions['full_logits']
        
    for i in range(len(test_predictions)):
        test_predictions[i] = test_predictions[i].cpu()
    

# Filter by a threshold on the accuracy
acc_threshold = args.acc_threshold
full_predictions = []
test_full_predictions = []
if args.acc_threshold > 0:
    # filter by some threshold.
    loader = DataLoader(dataset, batch_size=len(dataset))
    labels = next(iter(loader))[1]
    for i in range(len(predictions)):
        preds = torch.argmax(predictions[i], dim=1).cpu()
        acc = (preds == labels).sum() / len(labels)
        if acc > acc_threshold:
            full_predictions.append(predictions[i])
            if not args.save_only_train:
                test_full_predictions.append(test_predictions[i])
else:
    # dont filter
    for i in range(len(predictions)):
        full_predictions.append(predictions[i])
        if not args.save_only_train:
            test_full_predictions.append(test_predictions[i])


print ('Number of samples after filtering: {0}'.format(len(full_predictions)))
# create .pt datasets for generator training
gen_dataset, gen_dataloader = create_dataset_for_gm_training(full_predictions, dataset, 8, args.gen_ds_train_path)
if not args.save_only_train:
    gen_testset, gen_testloader = create_dataset_for_gm_training(test_full_predictions, testset, 8, args.gen_ds_test_path)
print("Dataset created!")