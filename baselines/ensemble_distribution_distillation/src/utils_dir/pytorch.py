"""Utilities for torch"""
import math
import logging
import types
import re
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def to_one_hot(labels, number_of_classes):
    """Labels is a tensor of class indices"""
    return nn.functional.one_hot(labels, number_of_classes)


def torch_settings(seed=1, use_gpu=False):
    """Pytorch settings"""
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    return device


def hard_classification(predicted_distribution):
    """Hard classification from forwards' probability distribution
    """
    class_ind, confidence = tensor_argmax(predicted_distribution)
    return class_ind, confidence


def tensor_argmax(input_tensor):
    value, ind = torch.max(input_tensor, dim=-1)
    return ind, value


def is_nan_or_inf(tensor):
    """NaN or Inf in tensor"""
    return torch.isnan(tensor).sum() > 0 or torch.isinf(tensor).sum() > 0


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):
    """ Cyclical learning rate
    the torch_optim.lr_scheduler.CycleLR does not work with Adam,
    instead I copied this one from here:
    https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
    """

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def adapted_lr(c=0.7):
    # the torch_optim.lr_scheduler.CycleLR does not work with Adam so I copied this one from here:
    # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee

    # Lambda function to calculate the LR
    lr_lambda = lambda it: (it + 1)**(-c)

    return lr_lambda


def positive_linear_asymptote(epsilon=0.0):
    """Transform: R --> R+

    Element-wise map of input_ input to positive real axis

    Asymptotically linear in input for large inputs

    Args:
        epsilon (float): Small positive offset for numerical stability
    """
    return lambda input_: torch.log(1 + torch.exp(input_)) + epsilon


def positive_exponential(epsilon=0.0):
    """Transform: R --> R+

    Element-wise map of input_ input to positive real axis

    Args:
        epsilon (float): Small positive offset for numerical stability
    """
    return lambda input_: torch.exp(input_) + epsilon


def positive_moberg(epsilon=0.0):
    """Transform: R --> R+

    Element-wise map of input_ input to positive real axis
    As used in John Moberg's thesis

    Args:
        epsilon (float): Small positive offset for numerical stability
    """

    return lambda input_: torch.log(1 + torch.exp(input_) + epsilon) + epsilon


LAMBDA_REGEX = re.compile(r"<function ([a-z,_]+).<locals>.<lambda> at .+>")


def human_readable_lambda(lambda_: types.LambdaType):
    raw = lambda_.__str__()
    name = str()
    match = LAMBDA_REGEX.match(raw)
    if match is not None:
        name = match.group(1)
    else:
        LOGGER.error("Could not find lambda name")
        name = "Unknown"
    return name


def gaussian_mixture_moments(mus, sigma_sqs):
    """Estimate moments of a gaussian mixture model

    B - number of observations/samples
    N - number of components in mixture

    Args:
        mus torch.tensor((B, N)): Collection of mu-values
        sigma_sqs torch.tensor((B, N)): Collection of sigma_sq-values
    """

    with torch.no_grad():
        mu = torch.mean(mus, dim=1)
        sigma_sq = torch.mean(sigma_sqs + mus**2, dim=1) - mu**2

    return mu, sigma_sq


def human_readable_arch(layers: torch.nn.modules.container.ModuleList):
    arch = list()
    for layer in layers:
        arch.append(layer.in_features)
    arch.append(layer.out_features)
    return arch
