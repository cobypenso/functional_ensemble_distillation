import torch
import torch.nn as nn
from typing import Dict

from fvcore.common.config import CfgNode
from giung2.layers import *


__all__ = [
    "build_centroid_classifier",
]


class CentroidClassifier(nn.Module):

    def __init__(
            self,
            feature_dim: int,
            num_classes: int,
            **kwargs,
        ) -> None:
        super(CentroidClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feature_dim)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:

        outputs = dict()

        # make predictions
        diff   = torch.unsqueeze(x, dim=1) - torch.unsqueeze(self.centers, dim=0)
        dist   = torch.sum(torch.mul(diff, diff), dim=-1)
        logits = -0.5 * dist

        outputs["logits"] = logits
        outputs["confidences"] = torch.softmax(outputs["logits"], dim=1)
        outputs["log_confidences"] = torch.log_softmax(outputs["logits"], dim=1)

        return outputs


def build_centroid_classifier(cfg: CfgNode) -> nn.Module:

    kwargs = {}

    classifier = CentroidClassifier(
        feature_dim = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.FEATURE_DIM,
        num_classes = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_CLASSES,
        **kwargs
    )

    return classifier
