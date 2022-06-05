# Functional Ensemble Distillation

# Framework

![highres_arch](https://user-images.githubusercontent.com/70380018/169815229-c36b0aaf-aad1-47fd-baf9-0b517db1b301.png)

# Generator architecture

![highres_noise](https://user-images.githubusercontent.com/70380018/169815290-a9d56054-3c26-4806-b9e6-f83a884f189e.png)


# Ensemble
Architecures available:
1. resnet18_gn_ws_cifar10
2. resnet18_gn_ws_cifar100
3. resnet18_gn_ws_stl10
Pick the right according to the workon dataset.

## Ensemble Training 
Ensemble training is based heavily on cSGHMC paper - 
https://github.com/ruqizhang/csgmcmc
https://arxiv.org/abs/1902.03932

Usage:

### Regular training
```bash
  CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 2000 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 3 --cycle 50 --samples 120 --threshold 0.1
```
### 10-Fold training
Training on fold 0:
```bash
CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 200 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 3 --cycle 50 --samples 12 --threshold 0.1 --bag 0 --idx_type kfold --idxs_path ./idxs_path.pkl
```
Training on fold X:
```bash
CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 200 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 3 --cycle 50 --samples 12 --threshold 0.1 --bag X --idx_type kfold --idxs_path ./idxs_path.pkl --isi X*12
```
### 10-Bagging training
Training on bag 0:
```bash
CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 200 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 3 --cycle 50 --samples 12 --threshold 0.1 --bag 0 --idx_type bagging --idxs_path ./idxs_path.pkl
```

Training on bag X:
```bash
CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 200 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 3 --cycle 50 --samples 12 --threshold 0.1 --bag X --idx_type bagging --idxs_path ./idxs_path.pkl --isi X*12
```

### 120-Bagging training
Training on bag 0:
```bash
CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 200 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 1 --cycle 50 --samples 1 --threshold 0.1 --bag 0 --idx_type bagging --idxs_path ./idxs_path.pkl
```

Training on bag X:
```bash
CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_train.py --checkpoints_dir ../checkpoints_CIFAR10/ 
        --data_path ../datasets/cifar10/ --epochs 200 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset cifar10
        --spc 1 --cycle 50 --samples 1 --threshold 0.1 --bag X --idx_type bagging --idxs_path ./idxs_path.pkl --isi X
```
## Ensemble Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python cyclic_sghmc_eval.py --checkpoints_dir ../checkpoints_CIFAR10/ 
    --data_path ../datasets/cifar10/ --num_samples 120 --arch resnet18_gn_ws_cifar10 --split 0.8 --dataset_name cifar10
    --acc_threshold 0.1 --dataset testset
```

# Generator
Architecures available:
1. cifar10_cat_gn_with_noise
2. cifar100_cat_gn_with_noise
3. stl10_cat_gn_with_noise
Pick the right according to the workon dataset.

## Creating an Auxiliary dataset
### Regular trainset
```bash
CUDA_VISIBLE_DEVICES=0 python dataset_creator.py --gen_ds_train_path ./train.pt
 --gen_ds_test_path ./test.pt --arch resnet18_gn_ws_cifar10 --checkpoints_dir ../checkpoints_CIFAR10/
 --data_path ../datasets/cifar10/ --split 0.8 --ensemble_size 120
 --dataset trainset --dataset_name cifar10 --acc_threshold 0
```
### Mixup dataset

```bash
CUDA_VISIBLE_DEVICES=0 python mixup_dataset_creator.py --data_path ../datasets/cifar10
--split 0.8 --dataset_size 150000 --dataset_path ./mixup.pt --ensemble_size 120
--arch resnet18_gn_ws_cifar10 --checkpoints_dir ../checkpoints_CIFAR10/
```
### KFold
```bash
CUDA_VISIBLE_DEVICES=0 python bagging_dataset_creator.py --models_per_bag 12 --split 0.8
 --data_path ../datasets/cifar10/ --idx_path ./idx_list.pkl --arch resnet18_gn_ws_cifar10
 --std kfold --dataset_name cifar10 --output_path kfold_ds.pt
```
### Bagging
```bash
CUDA_VISIBLE_DEVICES=0 python bagging_dataset_creator.py --models_per_bag 12 --split 0.8
 --data_path ../datasets/cifar10/ --idx_path ./idx_list.pkl --arch resnet18_gn_ws_cifar10
 --std bagging --dataset_name cifar10 --output_path bagging_ds/
```

## Generator Training
Training a generator on a cifar10 task with mixup auxiliary dataset.
```bash
CUDA_VISIBLE_DEVICES=0 python generator_training.py --checkpoints_dir ../checkpoints_CIFAR10/ --dataset_name cifar10 --split 0.8
 --gen_ds_train_path ../checkpoints_CIFAR10/mixup_ds/ --gen_ds_test_path ../checkpoints_CIFAR10/test.pt --design_option regular
 --gen cifar10_cat_gn_with_noise --latent_std 0.1 --noise_std 0.1 --epochs 200 --virtual_ensemble_size 8 --generator_lr 0.0001 
 --generator_batch_size 64 --epochs_per_valid 10 --save_dir ../checkpoints_CIFAR10/mixup/rbf_2_10_20_50/ --name cifar10_experiment 
 --aug augment --kernel_bw 2 10 20 50 --kernel_type rbf --weight_decay 0 --learn_noise True --ds_format folder --milestones_freq f3

```
## Generator Evaluation
```
generator_predict.py:
Two api functions available:
1. gen_predict - inference over a dataset.
2. gen_predict_single_image - inference over a single image.
```

## Usage recommendations

- Works best with cuda available.
- GPU with at least 10GB.

