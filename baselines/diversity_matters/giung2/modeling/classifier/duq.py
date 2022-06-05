import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from fvcore.common.config import CfgNode


__all__ = [
    "build_duq_classifier",
]


class DUQClassifier(nn.Module):

    def __init__(
            self,
            feature_dim: int,
            num_classes: int,
            centroid_dim: int,
            length_scale: float,
        ) -> None:
        super(DUQClassifier, self).__init__()
        self.feature_dim   = feature_dim
        self.num_classes   = num_classes
        self.centroid_dim = centroid_dim
        self.length_scale  = length_scale

        self.register_parameter(
            "weight", nn.Parameter(
                torch.zeros(self.centroid_dim, self.num_classes, self.feature_dim)
            )
        )
        self.register_buffer(
            "centroids_num", torch.zeros(self.num_classes)
        )
        self.register_buffer(
            "centroids_sum", torch.zeros(self.centroid_dim, self.num_classes)
        )

    @property
    def centroids(self) -> torch.Tensor:
        return self.centroids_sum / self.centroids_num.unsqueeze(0)

    def _rbf(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("ij,mnj->imn", x, self.weight)
        return (
            (z - self.centroids.unsqueeze(0))**2
        ).mean(1).div(2 * self.length_scale**2).mul(-1).exp()

    @torch.no_grad()
    def update_centroids(
            self,
            in_features: torch.Tensor,
            true_labels: torch.Tensor,
            momentum: float = 0.999,
        ) -> None:
        z = torch.einsum("ij,mnj->imn", in_features, self.weight)
        one_hot_labels = F.one_hot(true_labels, self.num_classes).float()
        _centroids_sum = torch.einsum("ijk,ik->jk", z, one_hot_labels)
        _centroids_num = one_hot_labels.sum(0)
        self.centroids_sum.copy_(
            momentum * self.centroids_sum + (1 - momentum) * _centroids_sum
        )
        self.centroids_num.copy_(
            momentum * self.centroids_num + (1 - momentum) * _centroids_num
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:

        outputs = dict()

        # make predictions
        confidences = self._rbf(x)
        outputs["confidences"] = confidences
        outputs["log_confidences"] = confidences.log()
        outputs["logits"] = outputs["log_confidences"]

        return outputs


def build_duq_classifier(cfg: CfgNode) -> nn.Module:

    classifier = DUQClassifier(
        feature_dim  = cfg.MODEL.CLASSIFIER.DUQ_CLASSIFIER.FEATURE_DIM,
        num_classes  = cfg.MODEL.CLASSIFIER.DUQ_CLASSIFIER.NUM_CLASSES,
        centroid_dim = cfg.MODEL.CLASSIFIER.DUQ_CLASSIFIER.CENTROID_DIM,
        length_scale = cfg.MODEL.CLASSIFIER.DUQ_CLASSIFIER.LENGTH_SCALE,
    )

    # initialize weights
    nn.init.kaiming_normal_(classifier.weight, nonlinearity="relu")
    classifier.centroids_num.copy_(classifier.centroids_num + 13)
    nn.init.normal_(classifier.centroids_sum, mean=0.0, std=0.05)

    return classifier
