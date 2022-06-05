import torch
import numpy as np
import torch.nn as nn
import random
import logging
import argparse
from contextlib import contextmanager
import os
import json
from pathlib import Path
import sys
from io import BytesIO
import wandb


def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def get_device(cuda=True, gpus='0'):
    return torch.device("cuda:" + gpus if torch.cuda.is_available() and cuda else "cpu")


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# create folders for saving models and logs
def _init_(out_path, exp_name):
    script_path = os.path.dirname(__file__)
    script_path = '.' if script_path == '' else script_path
    if not os.path.exists(out_path + '/' + exp_name):
        os.makedirs(out_path + '/' + exp_name)
    # save configurations
    os.system('cp -r ' + script_path + '/*.py ' + out_path + '/' + exp_name)


def get_art_dir(args):
    art_dir = Path(args.out_dir)
    art_dir.mkdir(exist_ok=True, parents=True)

    curr = 0
    existing = [
        int(x.as_posix().split('_')[-1])
        for x in art_dir.iterdir() if x.is_dir()
    ]
    if len(existing) > 0:
        curr = max(existing) + 1

    out_dir = art_dir / f"version_{curr}"
    out_dir.mkdir()

    return out_dir


def save_experiment(args, results, return_out_dir=False, save_results=True):
    out_dir = get_art_dir(args)

    json.dump(
        vars(args),
        open(out_dir / "meta.experiment", "w")
    )

    # loss curve
    if save_results:
        json.dump(results, open(out_dir / "results.experiment", "w"))

    if return_out_dir:
        return out_dir


def topk(true, pred, k):
    max_pred = np.argsort(pred, axis=1)[:, -k:]  # take top k
    two_d_true = np.expand_dims(true, 1)  # 1d -> 2d
    two_d_true = np.repeat(two_d_true, k, axis=1)  # repeat along second axis
    return (two_d_true == max_pred).sum()/true.shape[0]


def to_one_hot(y, num_classes, dtype=torch.double):
    # convert a single label into a one-hot vector
    y_output_onehot = torch.zeros((y.shape[0], num_classes), dtype=dtype, device=y.device)
    return y_output_onehot.scatter_(1, y.unsqueeze(1), 1)


def CE_loss(y, y_hat, num_classes, reduction='mean', convert_to_one_hot=True):
    # convert a single label into a one-hot vector
    y_output_onehot = to_one_hot(y, num_classes, dtype=y_hat.dtype) if convert_to_one_hot else y_hat
    if reduction == 'mean':
        return - torch.sum(y_output_onehot * torch.log(y_hat + 1e-12), dim=1).mean()
    return - torch.sum(y_output_onehot * torch.log(y_hat + 1e-12))


def CE_with_probs(y, y_hat, reduction='mean', sum_dim=2):
    # convert a single label into a one-hot vector
    if reduction == 'mean':
        return - torch.sum(y * torch.log(y_hat + 1e-12), dim=sum_dim).mean()
    return - torch.sum(y * torch.log(y_hat + 1e-12))


def calc_metrics(results):
    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])
    avg_loss = np.mean([val['loss'] for val in results.values()])
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def model_save(model, file=None, log_to_wandb=None):
    if file is None:
        file = BytesIO()
    torch.save({'model_state_dict': model.state_dict()}, file)
    if log_to_wandb:
        wandb.save(file.as_posix())

    return file


def model_load(model, file):
    if isinstance(file, BytesIO):
        file.seek(0)

    model.load_state_dict(
        torch.load(file, map_location=lambda storage, location: storage)['model_state_dict']
    )

    return model


def save_data(tensor, file, log_to_wandb=None):
    torch.save(tensor, file)
    if log_to_wandb:
        wandb.save(file.as_posix())


def load_data(file):
    return torch.load(file, map_location=lambda storage, location: storage)


def map_labels(labels):
    new_labels = np.arange(max(labels) + 1)
    original_labels = np.unique(labels)
    orig_to_new = {o: n for o, n in zip(original_labels, new_labels)}
    return np.asarray([orig_to_new[l] for l in labels]).astype(np.long)


def take_subset_of_classes(data, labels, classes):
    targets = np.asarray(labels)
    indices = np.isin(targets, classes)
    new_data, new_labels = data[indices], targets[indices].tolist()
    return new_data, map_labels(new_labels)


class DuplicateToChannels:
    """Duplicate single channel 3 times"""

    def __init__(self):
        pass

    def __call__(self, x):
        return x.repeat((3, 1, 1))


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()