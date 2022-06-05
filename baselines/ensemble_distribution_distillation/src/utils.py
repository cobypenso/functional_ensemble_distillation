"""Utilities module

TODO: This is now one large beast.
I have started to move utils into separate modules in utils_dir
"""
import sys
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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


def parse_args():
    """Arg parser"""
    parser = argparse.ArgumentParser(description="Ensemble")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=10,
                        help="Number of epochs")
    parser.add_argument("--num_ensemble_members",
                        type=int,
                        default=120,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--retrain",
                        action="store_true",
                        help="Retrain ensemble from scratch")
parser.add_argument("--model_dir",
                        type=Path,
                        default="./models",
                        help="Model directory")
    parser.add_argument("--data_dir",
                        type=str,
                        default="./models",
                        help="dataset directory")
    parser.add_argument("--model_path",
                        type=str,
                        default=None,
                        help="Model save path")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        help="Model save path")
    parser.add_argument("--saved_model",
                        type=_saved_model_path_arg,
                        default=None,
                        help="Path to saved model")
    parser.add_argument("--log_dir",
                        type=Path,
                        default="./logs",
                        help="Logs directory")
    parser.add_argument("--ensemble_dir",
                        type=str,
                        help="Ensemble dir name")
    parser.add_argument("--log_level",
                        type=_log_level_arg,
                        default=logging.INFO,
                        help="Log level")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="Random seed, NB both cuda and cpu")
    parser.add_argument("--gpu",
                        action="store_true",
                        help="Use gpu, if available")
    parser.add_argument("--norm_type",
                        type = str, default = 'bn',
                        help="Use gpu, if available")
    parser.add_argument("--conv_type",
                        type = str, default = 'original',
                        help="original or ws")                   
    parser.add_argument("--distill",
                        action="store_true",
                        help="distill")

    parser.add_argument("--predict",
                        action="store_true",
                        help="predict using the distilled model")
    parser.add_argument("--ood",
                        action="store_true",
                        help="predict ood using the distilled model")
    parser.add_argument("--rep",
                        type=int,
                        default=1,
                        help="Replication number (cifar10 experiments)")
    parser.add_argument("--output_size",
                        type=int,
                        default=10,
                        help="10 - cifar10, 100 - cifar100")
    parser.add_argument("--predictions_save_path",
                        type=str,
                        default='.',
                        help="path for which to save the predictions on the distilled model")
    parser.add_argument("--save_format",
                        type=str,
                        default='h5')

    return parser.parse_args()


def _log_level_arg(arg_string):
    arg_string = arg_string.upper()
    if arg_string == "DEBUG":
        log_level = logging.DEBUG
    elif arg_string == "INFO":
        log_level = logging.INFO
    elif arg_string == "WARNING":
        log_level = logging.WARNING
    elif arg_string == "ERROR":
        log_level = logging.WARNING
    elif arg_string == "CRITICAL":
        log_level = logging.WARNING
    else:
        raise argparse.ArgumentTypeError(
            "Invalid log level: {}".format(arg_string))
    return log_level


def _saved_model_path_arg(arg_string):
    model_path = Path(arg_string)
    if not model_path.exists():
        raise argparse.ArgumentTypeError(
            "Saved model does not exist: {}".format(model_path))
    return model_path


LOG_FORMAT = "%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s"


def setup_logger(log_path=None,
                 logger=None,
                 log_level=logging.INFO,
                 fmt=LOG_FORMAT):
    """Setup for a logger instance.

    Args:
        log_path (str, optional): full path
        logger (logging.Logger, optional): root logger if None
        log_level (logging.LOGLEVEL, optional):
        fmt (str, optional): message format

    """
    logger = logger if logger else logging.getLogger()
    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.setLevel(log_level)
    logger.handlers = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    log_path = Path(log_path)
    if log_path:
        directory = log_path.parent
        directory.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Log at {}".format(log_path))


def hard_classification(predicted_distribution):
    """Hard classification from forwards' probability distribution
    """
    class_ind, confidence = tensor_argmax(predicted_distribution)
    return class_ind, confidence


def tensor_argmax(input_tensor):
    value, ind = torch.max(input_tensor, dim=-1)
    return ind, value


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


def is_nan_or_inf(tensor):
    return torch.isnan(tensor) or torch.isinf(tensor)


def adapted_lr(c=0.7):
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


def generate_order(arr, descending=True):
    """Generate order based on array"""
    sorted_indices = torch.argsort(arr, 0, descending=descending)
    return sorted_indices.reshape((len(arr), ))


def moving_average(arr, window):
    """Moving average"""
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def plot_error_curve(ax,
                     y_true,
                     y_pred,
                     uncert_meas,
                     label,
                     every_nth,
                     window_size=10):
    """Plot errors sorted according to uncertainty measure"""
    error = (y_true[::every_nth] - y_pred[::every_nth])**2
    error, uncert_meas = np.array(error), np.array(uncert_meas[::every_nth])
    sorted_inds = generate_order(uncert_meas)
    sorted_error = error[sorted_inds]
    smooth_error = moving_average(sorted_error, window_size)
    ax.plot(np.arange(len(smooth_error)), smooth_error, label=label)


def plot_sparsification_error(ax, y_true, y_pred, uncert_meas, label,
                              num_partitions):
    """Plot errors sorted according to uncertainty measure"""
    y_true = y_true.reshape((y_true.shape[0], 1))
    y_pred = y_pred.reshape((y_pred.shape[0], 1))
    rel_part_size, sparse_err, sparse_err_oracle = sparsification_error(
        y_true, y_pred, uncert_meas, num_partitions)
    ax.plot(rel_part_size, sparse_err, label=label)
    ax.plot(rel_part_size, sparse_err_oracle, label="Oracle")
    ax.set_xlabel("Fraction of removed points")
    # ax.set_ylabel("\textit{SE}")
    ax.legend()


def sparsification_error(y_true, y_pred, uncert_meas, num_partitions):
    """Plot errors sorted according to uncertainty measure"""

    uncert_order = generate_order(uncert_meas)
    true_order = generate_order((y_true - y_pred)**2)

    rel_part_sizes = list()
    sparse_err = list()
    sparse_err_oracle = list()

    for ind in np.linspace(0, len(y_true), num_partitions, dtype=np.int):
        if ind == len(y_true):
            error_part, error_part_oracle = 0.0, 0.0
            rel_part_size = 1.0
        else:

            y_true_part, y_pred_part = y_true[uncert_order[ind:]], y_pred[
                uncert_order[ind:]]
            error_part = torch.mean((y_true_part - y_pred_part)**2)

            y_true_part_oracle, y_pred_part_oracle = y_true[
                true_order[ind:]], y_pred[true_order[ind:]]
            error_part_oracle = torch.mean(
                (y_true_part_oracle - y_pred_part_oracle)**2)
            rel_part_size = 1 - len(y_true_part) / len(y_true)

        rel_part_sizes.append(rel_part_size)
        sparse_err.append(error_part)
        sparse_err_oracle.append(error_part_oracle)

    rel_part_sizes, sparse_err, sparse_err_oracle = torch.tensor(
        rel_part_sizes), torch.tensor(sparse_err), torch.tensor(
            sparse_err_oracle)
    sparse_err /= sparse_err[0].item()
    sparse_err_oracle /= sparse_err_oracle[0].item()
    return rel_part_sizes, sparse_err, sparse_err_oracle


def ause(y_true, y_pred, uncert_meas, num_partitions):
    y_true = y_true.reshape((y_true.shape[0], 1))
    y_pred = y_pred.reshape((y_pred.shape[0], 1))
    x, y, oracle = sparsification_error(y_true, y_pred, uncert_meas,
                                        num_partitions)
    sparse_err_diff = y - oracle
    return area_under_curve(x=x, y=sparse_err_diff)


def area_under_curve(x, y):
    """Calculate area under curve"""
    return torch.trapz(x=x, y=y)


def plot_uncert(ax,
                inputs,
                targets,
                every_nth,
                mean_mu=None,
                ale=None,
                epi=None):
    """Plot 1D uncertainty"""

    # Plot data
    ax.scatter(inputs, targets, s=3)

    # Plot bounds
    lower_x_bound = np.array([-3, -3])
    upper_x_bound = np.array([3, 3])
    y_bound = np.array([-2, 2])
    ax.plot(lower_x_bound, y_bound, "b--")
    ax.plot(upper_x_bound, y_bound, "b--")

    x = inputs[::every_nth]

    if mean_mu is not None:
        mean_mu = mean_mu[::every_nth]
        ax.plot(x, mean_mu, "r-", label="$\\mu_{avg}(x)$")
    if ale is not None:
        ale = ale[::every_nth]
        ax.errorbar(x,
                    mean_mu,
                    np.sqrt(ale),
                    errorevery=5,
                    color="r",
                    label="$E_w[\\sigma_w^2(x)]$")
        # ax.plot(x, ale, \"g-\", label=\"$E_w[\\sigma_w^2(x)]$\")
    if epi is not None:
        epi = epi[::every_nth]
        ax.fill_between(x,
                        mean_mu + np.sqrt(epi),
                        mean_mu - np.sqrt(epi),
                        facecolor="blue",
                        alpha=0.5,
                        label="var$_w(\\mu_w(x))$")
        # ax.plot(x, np.sqrt(100*epi))
    ax.legend(prop={'size': 20})
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


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


def unpack_results(result):
    """Specific experiment function"""
    ens, dist = result
    ens, ens_header = gen_arr(ens)
    dist, dist_header = gen_arr(dist)
    return (ens, ens_header), (dist, dist_header)


def gen_arr(sub_result):
    """Specific experiment function"""
    (rmse, nll, ause) = sub_result

    return np.column_stack((rmse, nll, ause)), ["rmse", "nll", "ause"]


def csv_result(result, header=False, file=None):
    if file is None:
        file_ens = sys.stdout
        file_dist = sys.stdout
    else:
        file_ens = file.parent / (file.stem + "_ens" + file.suffix)
        file_dist = file.parent / (file.stem + "_dist" + file.suffix)
    delimiter = ";"
    (ens, ens_header), (dist, dist_header) = unpack_results(result)
    np.savetxt(file_ens, ens, delimiter=";", header=delimiter.join(ens_header))
    np.savetxt(file_dist,
               dist,
               delimiter=";",
               header=delimiter.join(dist_header))
