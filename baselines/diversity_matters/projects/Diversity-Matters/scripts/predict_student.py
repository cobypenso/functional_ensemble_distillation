import os
import sys
import shutil

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from models import get_net
from data import get_stl10, get_cifar100, get_cifar10, get_svhn
import torch
import torch.nn.functional as F
import logging
import argparse
from tabulate import tabulate
from torch.autograd import Variable 
from torch.utils.data import TensorDataset, DataLoader

from giung2.config import get_cfg
from giung2.engine import launch, create_ddp_model, default_setup
from giung2.engine.utils import synchronize, get_rank
from giung2.data.build import build_dataloaders
from giung2.modeling.build import build_model
from giung2.solver.build import build_optimizer, build_scheduler

from train_student_with_csghmc_ensemble import run_epoch


def test(args, cfg, logger, dataloaders, model):
    model.cuda()
    model.eval()
    dataloader = dataloaders["test_loader"]
    
    outputs_list = []
    images_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            wow = images
            images = images.cuda().float()
            ensemble_size = args.ensemble_size
            batch_size = len(images)
            # preprocess batches
            images = images.repeat(ensemble_size, 1, 1, 1)
            # images = images.permute(1,0,2,3,4)
            # images = images.reshape(-1, *images.shape[2:])
            # images_split = torch.split(images, images.size(0) // ensemble_size)
            output = model(images)
            output = output.view(ensemble_size, batch_size, -1).permute(1,0,2)
            # output = output.view(batch_size, ensemble_size, -1)
            probs = torch.nn.Softmax(dim = 2)(output)
            for i in range(batch_size):
                outputs_list.append(probs[i].cpu())
                images_list.append(wow[i].cpu())
                labels_list.append(labels[i].cpu())


    return images_list, outputs_list, labels_list


def main(args, cfg):

    default_setup(cfg, args)
    logger = logging.getLogger("giung2")

    # build dataloaders
    if args.dataset_name == 'cifar10':
        trainset, validset, testset = get_cifar10(args.data_path, split = args.split)
    elif args.dataset_name == 'stl10':
        trainset, validset, testset, unlabeledset = get_stl10(args.data_path, split = args.split)
    elif args.dataset_name == 'cifar100':
        trainset, validset, testset = get_cifar100(args.data_path, split = args.split)
    # FOR OOD
    elif args.dataset_name == 'svhn':
        trainset, validset, testset, extraset = get_svhn(args.data_path, split = args.split)

    dataloaders = {}
    dataloaders["dataloader"] = torch.utils.data.DataLoader(trainset, 
                                          batch_size=args.batch_size, 
                                          num_workers=4,
                                          shuffle=True)

    dataloaders["val_loader"] = torch.utils.data.DataLoader(validset, #was testset 
                                         batch_size=args.batch_size, 
                                         shuffle=False, 
                                         num_workers=4)

    dataloaders["test_loader"] = torch.utils.data.DataLoader(testset, #was testset 
                                         batch_size=args.batch_size, 
                                         shuffle=False, 
                                         num_workers=0)


    log_str = "Build dataloaders:\n"
    log_str += tabulate([
        (
            k,
            len(dataloaders[k].dataset),
            len(dataloaders[k]),
            dataloaders[k].batch_size,
            type(dataloaders[k].sampler).__name__,
        ) for k in dataloaders
    ], headers=["Key", "# Examples", "# Batches", "Batch Size", "Sampler"])
    logger.info(log_str + "\n")


    # build model
    model = get_net(arch=args.distilled_arch, 
                    batch_ensemble=True, 
                    ensemble_size = args.ensemble_size,
                    alpha_initializer = (cfg.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.pop('NAME'), cfg.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.pop('VALUES')),
                    gamma_initializer = (cfg.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.pop('NAME'), cfg.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.pop('VALUES')),
                    use_ensemble_bias = cfg.MODEL.BATCH_ENSEMBLE.USE_ENSEMBLE_BIAS)

    
    load_checkpoint = torch.load(args.model_path)
    print(load_checkpoint['best_acc1'])
    model.load_state_dict(load_checkpoint['model_state_dict'])

    # build teacher models
    _cfg = get_cfg()
    
    # test model
    images, predictions, labels = test(args, cfg, logger, dataloaders, model)

    
    create_dataset(predictions, images, labels, args.save_path)
    # finished
    logger.info("Finished.")


def create_dataset(predictions, images, labels, dataset_path):
    '''
        predictions - 
        dataset - (image,label)
        dataset_path - path to which to save the new dataset
    '''
    predictions_tensor = torch.cat(predictions,0).view(-1, *predictions[0].shape)
    images_tensor = torch.cat(images,0).view(-1, *images[0].shape)
    labels_tensor = torch.tensor(labels)

    print (predictions_tensor.shape)
    print (images_tensor.shape)
    print (labels_tensor.shape)
    new_dataset = TensorDataset(images_tensor, predictions_tensor, labels_tensor) # create your datset
    torch.save(new_dataset, dataset_path)
    


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=None, required=True, metavar="FILE",
                        help="path to config file")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12345", metavar="URL",
                        help="URL for pytorch distributed backend")
    parser.add_argument("--checkpoint-last-only", default=False, action="store_true",
                        help="save 'checkpoint.pth.tar' as the last checkpoint")
    
    # KD settings
    parser.add_argument("--distilled_arch", default=None)
    parser.add_argument("--ensemble_size", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_name", type=str, default='cifar10')
    parser.add_argument("--split", type=float, default=0.8)

    parser.add_argument("--model_path", default=None, type = str, required=True, metavar="FILE")
    parser.add_argument("--kd-alpha", default=0.9, type=float)
    parser.add_argument("--kd-temperature", default=4.0, type=float)
    parser.add_argument("--kd-method-name", default=None, type=str,
                        choices=["gaussian", "ods_l2", "c_ods_l2",])
    parser.add_argument("--kd-method-step-size", default=1.0, type=float,)
    parser.add_argument("--output_dir", type = str)
    parser.add_argument("--save_path", type = str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options at the end of the command.")
    args = parser.parse_args()
    print("Command Line Args:", args)

    # load config file
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file, allow_unsafe=True)
    cfg.merge_from_list(args.opts)

    # check the number of GPUs
    if cfg.NUM_GPUS > torch.cuda.device_count():
        raise AssertionError(
            f"Try to use cfg.NUM_GPUS={cfg.NUM_GPUS}, "
            f"but there exists only {torch.cuda.device_count()} GPUs..."
        )

    if args.distilled_arch is None:
        args.distilled_arch = args.ensemble_arch
    
    
    launch(
        main_func=main,
        num_gpus_per_machine=cfg.NUM_GPUS,
        num_machines=1,
        machine_rank=0,
        dist_url=args.dist_url,
        args=(args, cfg,),
    )
