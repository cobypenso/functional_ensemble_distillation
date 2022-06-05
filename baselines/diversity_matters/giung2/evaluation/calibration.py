import torch


__all__ = [
    "evaluate_ece",
    "evaluate_sce",
    "evaluate_tace",
    "evaluate_ace",
]


@torch.no_grad()
def evaluate_ece(confidences: torch.Tensor,
                 true_labels: torch.Tensor,
                 n_bins: int = 15) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.
        n_bins (int): the number of bins used by the histrogram binning.

    Returns:
        ece (float): expected calibration error of predictions.
    """
    # predicted labels and its confidences
    pred_confidences, pred_labels = torch.max(confidences, dim=1)

    # fixed binning (n_bins)
    ticks = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = ticks[:-1]
    bin_uppers = ticks[ 1:]

    # compute ECE across bins
    accuracies = pred_labels.eq(true_labels)
    ece = torch.zeros(1, device=confidences.device)
    avg_accuracies = []
    avg_confidences = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = pred_confidences.gt(
            bin_lower.item()
        ) * pred_confidences.le(
            bin_upper.item()
        )
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = pred_confidences[in_bin].mean()
            ece += torch.abs(
                avg_confidence_in_bin - accuracy_in_bin
            ) * prop_in_bin
            avg_accuracies.append(accuracy_in_bin.item())
            avg_confidences.append(avg_confidence_in_bin.item())
        else:
            avg_accuracies.append(None)
            avg_confidences.append(None)

    return ece.item()


@torch.no_grad()
def evaluate_sce(confidences: torch.Tensor,
                 true_labels: torch.Tensor,
                 n_bins: int = 15) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.
        n_bins (int): the number of bins used by the histrogram binning.
    
    Returns:
        sce (float): static calibration error of predictions.
    """
    ticks = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = ticks[:-1]
    bin_uppers = ticks[ 1:]
    
    n_objects, n_classes = confidences.size()
    sce = torch.zeros(1, device=confidences.device)
    for cur_class in range(n_classes):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            cur_class_conf = confidences[:, cur_class]
            in_bin = cur_class_conf.gt(
                bin_lower.item()
            ) * cur_class_conf.le(
                bin_upper.item()
            )

            bin_acc = true_labels[in_bin].eq(cur_class)
            bin_conf = cur_class_conf[in_bin]

            bin_size = torch.sum(in_bin)
            if bin_size > 0:
                avg_confidence_in_bin = torch.mean(bin_conf.float())
                avg_accuracy_in_bin = torch.mean(bin_acc.float())
                delta = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
                sce += delta * bin_size / (n_objects * n_classes)

    return sce.item()


@torch.no_grad()
def evaluate_tace(confidences: torch.Tensor,
                  true_labels: torch.Tensor,
                  n_bins: int = 15,
                  threshold: float = 1e-3) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.
        n_bins (int): the number of bins used by the histrogram binning.
        threshold (float): the value for thresholding to avoid tiny predictions.
    
    Returns:
        tace (float): thresholded adaptive calibration error of predictions.
    """
    n_objects, n_classes = confidences.size()

    tace = torch.zeros(1, device=confidences.device)
    for cur_class in range(n_classes):
        cur_class_conf = confidences[:, cur_class]

        targets_sorted = true_labels[cur_class_conf.argsort()]
        cur_class_conf_sorted = cur_class_conf.sort()[0]

        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]

        bin_size = len(cur_class_conf_sorted) // n_bins
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins - 1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind
            bin_acc = (targets_sorted[bin_start_ind:bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind:bin_end_ind]
            avg_confidence_in_bin = torch.mean(bin_conf.float())
            avg_accuracy_in_bin = torch.mean(bin_acc.float())
            delta = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            tace += delta * bin_size / (n_objects * n_classes)

    return tace.item()


def evaluate_ace(confidences: torch.Tensor,
                 true_labels: torch.Tensor,
                 n_bins: int = 15) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.
        n_bins (int): the number of bins used by the histrogram binning.
    
    Returns:
        ace (float): adaptive calibration error of predictions.
    """
    ace = evaluate_tace(
        confidences, true_labels,
        n_bins=n_bins, threshold=0.0
    )

    return ace
