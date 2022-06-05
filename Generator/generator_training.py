'''
Script that responsible for Generator Training.

Usage:

CUDA_VISIBLE_DEVICES=0 python generator_training.py --checkpoints_dir ../checkpoints_CIFAR10/ --dataset_name cifar10 --split 0.8
 --gen_ds_train_path ../checkpoints_CIFAR10/mixup_ds/ --gen_ds_test_path ../checkpoints_CIFAR10/test.pt --design_option regular
 --gen cifar10_cat_gn_with_noise --latent_std 0.1 --noise_std 0.1 --epochs 200 --virtual_ensemble_size 8 --generator_lr 0.0001 
 --generator_batch_size 64 --epochs_per_valid 10 --save_dir ../checkpoints_CIFAR10/mixup/rbf_2_10_20_50/ --name cifar10_experiment 
 --aug augment --kernel_bw 2 10 20 50 --kernel_type rbf --weight_decay 0 --learn_noise True --ds_format folder --milestones_freq f3

Explaination:
* Two options for training:
    1. Regular - using a primary trainset (can be mixup or any other auxiliary dataset)
    2. Combination - using a primary and a secondary datasets (for example, trainset + mixup dataset)
* Usage above describes the 'regular' training option with mixup dataset, which yeild best results in our experiments.
'''

import sys
sys.path.append('../')
import argparse
import wandb
from data import *
from training_helper import *
from utils import *
from models import *
import torch.distributed as dist
import torch.multiprocessing as mp
from classification_generators.generators import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

##############################################################################################

def train_generator(generator, dataloader, testloader, args, gpu=0):
    '''
        @brief Train generator model.

        @param - dataloader - train dataloader. each example is (image, prob vectors, label).
        @param - testloader - test dataloader. each example is (image, prob vectors, label).
        @param - args - parameter to passed on.
        @param - gpu - used for multi-GPU option.
    '''
    print ('=== Start Training ===')
    with wandb.init(project=args.name, notes=args.notes):

        gen = train(generator = generator,
                    trainloader = dataloader,
                    optimizer = 'Adam',
                    device=args.device,
                    testloader=testloader,
                    args=args,
                    gpu = gpu,
                    train_type = args.design_option)

    return gen

##############################################################################################

def train_generator_with_combination(generator, dataloader, trainloader_secondary, testloader, args, gpu=0):
    '''
        @brief Train generator model.

        @param - dataloader - train dataloader. each example is (image, prob vectors, label).
        @param - trainloader_secondary - secondary train dataloader. each example is (image, prob vectors, label).
        @param - testloader - test dataloader. each example is (image, prob vectors, label).
        @param - args - parameter to passed on.
        @param - gpu - used for multi-GPU option.
    '''

    print ('=== Start Training ===')
    with wandb.init(project=args.name, notes=args.notes):
        gen = train_with_combination(generator = generator,
                    trainloader = dataloader,
                    trainloader_secondary = trainloader_secondary,
                    optimizer = 'Adam',
                    device=args.device,
                    testloader=testloader,
                    args=args,
                    gpu = gpu,
                    train_type = args.design_option)
    return gen


##############################################################################################    

def main(gpu = None, args = None):
    # Multi Processing on several GPUs #
    if args.mp:
        rank = args.nr * args.gpus + gpu	                          
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=args.world_size,                              
            rank=rank                                               
        )

    torch.cuda.set_device(gpu)
    args.local_rank = gpu
    # -------------------------------- #
    
    # load trainset - generated usin dataset_creator.py/mixup_dataset_creator.py/... in generating_datasets folder
    if args.ds_format == 'file':
        gen_dataset, gen_dataloader = load_dataset_for_gm_training(args.generator_batch_size, args.gen_ds_train_path, args.aug)
    else:
        gen_dataset, gen_dataloader = load_dataset_for_gm_training_from_folder(args.generator_batch_size, args.gen_ds_train_path, args.aug, args.bagging)
    
    # load testset
    gen_testset, gen_testloader = load_dataset_for_gm_training(4, args.gen_ds_test_path, 'no_augment')

        
    if args.mp:
        sampler = DistributedSampler(gen_dataset, 
                                     num_replicas = args.world_size, 
                                     rank = rank, 
                                     shuffle=True, 
                                     seed=args.seed)
        gen_dataloader = DataLoader(gen_dataset, 
                                    batch_size=args.generator_batch_size, 
                                    shuffle=False, 
                                    drop_last=True,
                                    sampler = sampler)

    # Create generator instance based on args (gen arch, checkpoint ...)
    generator = get_generator(args)

    if args.mp:
        generator.cuda(gpu)
        generator = DDP(generator, device_ids=[gpu], output_device = gpu)
        
    #Train generator
    if args.design_option in ['regular']:
        generator = train_generator(generator, gen_dataloader, gen_testloader, args, gpu)
    elif args.design_option in ['combination']:
        if args.secondary_ds_format == 'file': 
            secondary_dataset, trainloader_secondary = load_dataset_for_gm_training(args.generator_batch_size, args.secondary_ds_path, args.secondary_ds_aug)
        else:
            secondary_dataset, trainloader_secondary = load_dataset_for_gm_training_from_folder(args.generator_batch_size, args.secondary_ds_path, args.secondary_ds_aug)
        
        generator = train_generator_with_combination(generator, gen_dataloader, trainloader_secondary, gen_testloader, args)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generator Training')
    ################## Training ########################################
    parser.add_argument("--design_option", type=str, default='regular', choices=['regular','combination'])
    parser.add_argument("--bagging", type=bool, default=False, help = 'If dataset created using bagging technique - set to True')

    parser.add_argument('--ensemble_size', type=int, default=12)
    parser.add_argument('--virtual_ensemble_size', type=int, default=12)
    
    parser.add_argument("--generator_batch_size", nargs='?', default=8, type=int)
    parser.add_argument("--gradient_accumulate", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_dir", type=str, help='path to save generators')
    parser.add_argument("--generator_lr", type=float, default=0.1)
    parser.add_argument("--epochs_per_valid", type=int, default=10)
    parser.add_argument("--epoch_per_checkpoint", type=int, default=20)

    parser.add_argument("--secondary_loss_coeff", type=float, default=0.5)
    parser.add_argument("--stat_only_for_main_data", type=bool, default=True)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--milestones_freq", type=str, default='f3')

    ################## Data ########################################
    parser.add_argument('--ds_format', type=str, default = 'file', help='file or folder')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10','cifar100', 'stl10'], help = 'used only for tracking')
    parser.add_argument('--gen_ds_train_path', type=str, default = './ds_train.pt')
    parser.add_argument('--gen_ds_test_path', type=str, default = './ds_test.pt')

    parser.add_argument('--secondary_ds_path', type=str, default = './secondaty_ds_train.pt')
    parser.add_argument('--secondary_ds_format', type=str, default = 'file')

    ################## Resume Training ################################
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--gen_ckpt", type=str, default=None)
    parser.add_argument("--opt_ckpt", type=str, default=None)

    ################## MMD Loss ##########################################
    parser.add_argument("--kernel_type", type = str, default='rbf', choices=['rbf', 'multiscale', 'linear'])
    parser.add_argument('--kernel_bw', metavar='N', type=float, nargs='+')

    ################## Generator properties ##############################
    parser.add_argument('--gen', type=str, default = 'cifar10_cat_gn_with_noise', help='Generator type')
    parser.add_argument("--latent_std", type=float, default=0.1, help = 'std of the latent in the input to the generator')
    parser.add_argument("--noise_std", type=float, default=0.1, help = 'std of additive noise to each block in resnet')
    parser.add_argument("--learn_noise", type=bool, default = False)

    ################## Others ############################################
    parser.add_argument('--device_id',type = int, help = 'device id to use')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument("--name", type=str, default='cifar10_exp')
    parser.add_argument("--notes", type=str, default='')

    ################## Augmentations #####################################

    parser.add_argument('--aug', type = str, default='no_augment', help = 'Augmentation on primary train dataset')
    parser.add_argument('--secondary_ds_aug', type = str, default='no_augment', help = 'Augmentation on secondary dataset')
    
    ################## multi GPU related #################################  

    parser.add_argument("--mp", type=bool, default=False)
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--mp_port', type=str, default='1245')
    

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.mp:
        #########################################################
        args.world_size = args.gpus * args.nodes                #
        os.environ['MASTER_ADDR'] = 'localhost'                 #
        os.environ['MASTER_PORT'] = args.mp_port                #
        mp.spawn(main, nprocs=args.gpus, args=(args,))          #
        #########################################################
    else:
        gpu = torch.cuda.current_device()
        args.world_size = 1
        main(gpu = gpu, args = args)
