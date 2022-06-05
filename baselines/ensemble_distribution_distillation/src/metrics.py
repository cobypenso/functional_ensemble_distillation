"""Metrics"""
import logging
import torch
import numpy as np
import src.utils as utils
import torch.distributions.logistic_normal as torch_logistic_normal

LOGGER = logging.getLogger(__name__)


class Metric:
    """Metric class"""
    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.counter = 0
        self.memory = []

    def __str__(self, decimal_places=3):
        return "{}: {:.3f}".format(self.name, self.mean())

    def update(self, targets, outputs, true_labels = None):
        """Update metric"""
        with torch.no_grad():
            if true_labels is None:
                new_observation = self.function(outputs, targets)
            else:
                new_observation = self.function(outputs, targets, true_labels)
            self.memory.append(new_observation)
        self.counter += 1

    def mean(self):
        """Calculate mean of metric

        Returns: mean (float): NaN if self.memory is empty
        """
        if not self.memory:
            LOGGER.warning("Trying to calculate mean on unpopulated metric.")
        return torch.tensor(self.memory).mean().item()

    def std(self):
        """Calculate std of metric

        Returns: std (float): NaN if len(self.memory) < 2
        """

        if not self.memory:
            LOGGER.warning("Trying to calculate std on unpopulated metric.")
        return torch.tensor(self.memory).std().item()

    def reset(self):
        self.memory = list()
        self.counter = 0


def entropy(predicted_distribution, true_labels=None, correct_nan=False):
    """Entropy

    B = batch size, C = num classes
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values
    The true targets argument is simply there for conformity
    so that the entropy metric functions like any metric.

    Args:
        NOT USED true_labels: torch.tensor((B, C))
        predicted_distribution: torch.tensor((B(, N), C))
        logits: predicted_distribution provided in logits-form (for numerical stability)
        correct_nan: if True, will set 0*log 0 to 0

    Returns:
        entropy(ies): torch.tensor(B,)
    """

    if correct_nan:
        entr = predicted_distribution * torch.log(predicted_distribution)
        entr[torch.isnan(entr)] = 0
        entr = -torch.sum(entr, dim=-1)
    else:
        entr = -torch.sum(
            predicted_distribution * torch.log(predicted_distribution), dim=-1)

    return entr


def uncertainty_separation_parametric(mu, var):
    """Total, epistemic and aleatoric uncertainty

    based on a parametric (normal) variance measure

    M = length of input data x, N = number of distributions

    Args:
        mu: torch.tensor((M, N)): E(y|x) for M x and N distr.
        var: torch.tensor((M, N)): var(y|x) for M x and N distr.

    Returns:
        aleatoric_uncertainty: torch.tensor((M)):
            E_theta[var(y|x, theta)] for M x and N distr.
        epistemic_uncertainty: torch.tensor((M)):
            var_theta[E(y|x, theta)] for M x and N distr.
    """
    epistemic_uncertainty = torch.var(mu, dim=1)
    aleatoric_uncertainty = torch.mean(var, dim=1)
    return aleatoric_uncertainty, epistemic_uncertainty


def uncertainty_separation_variance(predicted_distribution, true_labels):
    """Total, epistemic and aleatoric uncertainty based on a variance measure

    B = batch size, N = num predictions
    Note: if a batch with B samples is given,
    then the output is a tensor with B values
    The true targets argument is simply there for conformity
    so that the entropy metric functions like any metric.

    Args:
        NOT USED true_labels: torch.tensor((B, 1))
        predicted_distribution: torch.tensor((B, N, 2))

    Returns:
        Tuple of uncertainties (relative the maximum uncertainty):
        Total uncertainty: torch.tensor(B,)
        Epistemic uncertainty: torch.tensor(B,)
        Aleatoric uncertainty: torch.tensor(B,)
    """

    true_labels = None
    total_uncertainty = np.var(predicted_distribution[:, :, 0], axis=-1)
    aleatoric_uncertainty = np.mean(predicted_distribution[:, :, 1], axis=-1)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty


def uncertainty_separation_entropy(predicted_distribution,
                                   true_labels=None,
                                   logits=False,
                                   correct_nan=False):
    """Total, epistemic and aleatoric uncertainty based on entropy of categorical distribution

    B = batch size, C = num classes, N = num predictions
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values
    The true targets argument is simply there for conformity
    so that the entropy metric functions like any metric.

    Args:
        NOT USED true_labels: torch.tensor((B, C))
        predicted_distribution: torch.tensor((B, N, C))

    Returns:
        Tuple of uncertainties (relative the maximum uncertainty):
        Total uncertainty: torch.tensor(B,)
        Epistemic uncertainty: torch.tensor(B,)
        Aleatoric uncertainty: torch.tensor(B,)
    """

    if isinstance(predicted_distribution, np.ndarray):
        predicted_distribution = torch.tensor(predicted_distribution)

    max_entropy = torch.log(
        torch.tensor(predicted_distribution.size(-1), dtype=torch.float))

    if logits:
        log_sum_exp_logits = torch.logsumexp(predicted_distribution,
                                             dim=-1,
                                             keepdim=True)
        p = torch.exp(predicted_distribution) / torch.exp(log_sum_exp_logits)
        mean_predicted_distribution = torch.mean(p, dim=1)

        aleatoric_uncertainty = - torch.sum(p * (predicted_distribution - log_sum_exp_logits), dim=[1, 2]) / \
            (predicted_distribution.size(1) * max_entropy)
    else:
        mean_predicted_distribution = torch.mean(predicted_distribution, dim=1)
        aleatoric_uncertainty = torch.mean(
            entropy(predicted_distribution, None), dim=1) / max_entropy

    total_uncertainty = entropy(mean_predicted_distribution,
                                None) / max_entropy
    if correct_nan:
        num_nan = torch.sum(torch.isnan(total_uncertainty)) + torch.sum(
            torch.isnan(aleatoric_uncertainty))
        LOGGER.info("Setting {} nan value(s) to 0.".format(num_nan))
        total_uncertainty[torch.isnan(total_uncertainty)] = 0
        aleatoric_uncertainty[torch.isnan(aleatoric_uncertainty)] = 0

    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty


def accuracy(predicted_distribution, true_labels):
    """ Accuracy
    B = batch size
    K = number of classes

    Args:
        true_labels: torch.tensor(B)
        predicted_distribution: torch.tensor(B, K)

    Returns:
        Accuracy: float
    """
    predicted_labels, _ = utils.tensor_argmax(predicted_distribution)
    number_of_elements = np.prod(true_labels.size())

    if number_of_elements == 0:
        number_of_elements = 1
    return (true_labels == predicted_labels).sum().item() / number_of_elements


def error(predicted_distribution, true_labels):
    """ Error
    B = batch size

    Args:
        true_labels: torch.tensor(B)
        predicted_distribution: torch.tensor(B)

    Returns:
        Error: float
    """
    predicted_labels = utils.tensor_argmax(predicted_distribution)
    number_of_elements = np.prod(true_labels.size())
    if number_of_elements == 0:
        number_of_elements = 1

    return (true_labels != predicted_labels).sum().item() / number_of_elements


def root_mean_squared_error(predictions, targets):
    """ Root mean squared error
    Calls square root on `mean_squared_error` below
    B = Batch size
    N = Sample size
    D = Output dimension

    Args:
        targets: torch.tensor(B, N, D)
        predictions: (torch.tensor(B, D)),
            regression estimate

    Returns:
        Error: float
    """

    rmse = torch.sqrt(mean_squared_error(predictions, targets))
    return rmse


def mean_squared_error(predictions, targets):
    """ Mean squared error
    Replaces squared_error below
    B = Batch size
    N = Sample size
    D = Output dimension

    Args:
        targets: torch.tensor(B, N, D)
        predictions: (torch.tensor(B, D)),
            regression estimate

    Returns:
        Error: float
    """

    B, N, _ = targets.size()
    sum_squared_errors = 0.0
    for n in np.arange(N):
        target = targets[:, n, :]
        sum_squared_errors += ((target - predictions)**2).sum()
    mse = sum_squared_errors / (B * N)
    return mse


def squared_error(predictions, targets):
    """ Error
    B = batch size
    D = output dimension

    Args:
        targets: torch.tensor(B, D)
        predictions: (torch.tensor(B, 2*D)),
            estimated mean and variances of the
            normal distribution of targets arranged as
            [mean_1, ... mean_D, var_1, ..., var_D]

    Returns:
        Error: float
    """

    number_of_elements = targets.size(0)
    if number_of_elements == 0:
        number_of_elements = 1

    return ((targets - predictions[:, :targets.size(-1)])**
            2).sum().item() / number_of_elements


def ece(predicted_distribution, labels):
    """"" Expected Calibration Error
    B = batch size
    D = output dimension

    Args:
        labels: np.ndarray(B,)
        predictions: np.ndarray(B, D)

    Returns:
        Expected Calibration Error: float
    """

    num_samples = labels.shape[0]

    probs_true_labels = np.zeros((predicted_distribution.shape[0], ))  # ??

    confidence = np.max(predicted_distribution, axis=-1)
    predicted_labels = np.argmax(predicted_distribution, axis=-1)

    # Sort the data
    ind = np.argsort(confidence)

    confidence = confidence[ind]
    predicted_labels = predicted_labels[ind]
    labels = labels[ind]

    # Will go for quartiles
    split_values = np.array(
        np.quantile(confidence, q=[0.25, 0.50, 0.75, 1.0], axis=0).tolist())

    num_buckets = split_values.shape[0]
    acc = np.zeros((num_buckets, 1))
    conf = np.zeros((num_buckets, 1))
    bucket_count = np.zeros((num_buckets, 1))

    j = 0
    for i in range(num_buckets):
        while confidence[j] <= split_values[i]:
            acc[i] += predicted_labels[j] == labels[j]
            conf[i] += confidence[j]
            bucket_count[i] += 1
            j += 1

            if j >= confidence.shape[0]:
                break

    acc = acc / bucket_count
    conf = conf / bucket_count

    ece = np.sum((bucket_count / num_samples) * np.abs(acc - conf))

    return ece
