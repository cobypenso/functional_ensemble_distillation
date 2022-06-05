import numpy as np
import torch


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
