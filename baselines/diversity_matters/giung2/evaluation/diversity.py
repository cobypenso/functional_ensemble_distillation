import torch


__all__ = [
    "compute_ent",
    "compute_kld",
]


@torch.no_grad()
def compute_ent(confidences: torch.Tensor, reduction="mean", eps=1e-12) -> torch.Tensor:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        reduction (str): specifies the reduction to apply to the output.
            - none: no reduction will be applied,
            - mean: the sum of the output will be divided by
                    the number of elements in the output.
        eps (float): small value to avoid evaluation of log(0).
    
    Returns:
        ent (Tensor): entropies for given confidences.
            - a tensor of shape [N,] when reduction is "none",
            - a tensor of shape [,] when reduction is "mean".
    """
    assert reduction in [
        "none", "mean",
    ], f"Unknown reduction = \"{reduction}\""

    ent = (confidences * torch.log(eps + confidences)).sum(1).neg() # [N,]

    if reduction == "mean":
        ent = ent.mean() # [,]

    return ent


@torch.no_grad()
def compute_kld(confidences: torch.Tensor, reduction="mean") -> torch.Tensor:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, M, K] of predicted confidences from ensembles.
        reduction (str): specifies the reduction to apply to the output.
            - none: no reduction will be applied,
            - mean: the sum of the output will be divided by
                    the number of elements in the output.
    
    Returns:
        kld (Tensor): KL divergences for given confidences from ensembles.
            - a tensor of shape [N,] when reduction is "none",
            - a tensor of shape [,] when reduction is "mean".
    """
    assert reduction in [
        "none", "mean",
    ], f"Unknown reduction = \"{reduction}\""

    kld = torch.zeros(confidences.size(0), device=confidences.device) # [N,]

    ensemble_size = confidences.size(1)
    if ensemble_size > 1:

        pairs = []
        for i in range(ensemble_size):
            for j in range(ensemble_size):
                pairs.append((i, j))

        for (i, j) in pairs:
            if i == j:
                continue
            kld += torch.nn.functional.kl_div(
                confidences[:, i, :].log(),
                confidences[:, j, :],
                reduction="none", log_target=False,
            ).sum(1) # [N,]

        kld = kld / (ensemble_size * (ensemble_size - 1))

    if reduction == "mean":
        kld = kld.mean() # [,]

    return kld
