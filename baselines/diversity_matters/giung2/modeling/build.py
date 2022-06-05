import torch
import torch.nn as nn

from fvcore.common.config import CfgNode
from giung2.modeling.architecture import *
from giung2.modeling.backbone import *
from giung2.modeling.classifier import *


def build_backbone(cfg: CfgNode) -> nn.Module:
    name = cfg.MODEL.BACKBONE.NAME

    if name == "build_resnet_backbone":
        backbone = build_resnet_backbone(cfg)
    elif name == "build_preresnet_backbone":
        backbone = build_preresnet_backbone(cfg)
    else:
        raise NotImplementedError(
            f"Unknown cfg.MODEL.BACKBONE.NAME = \"{name}\""
        )

    return backbone


def build_classifier(cfg: CfgNode) -> nn.Module:
    name = cfg.MODEL.CLASSIFIER.NAME

    if name == "build_softmax_classifier":
        classifier = build_softmax_classifier(cfg)
    elif name == "build_duq_classifier":
        classifier = build_duq_classifier(cfg)
    elif name == "build_centroid_classifier":
        classifier = build_centroid_classifier(cfg)
    else:
        raise NotImplementedError(
            f"Unknown cfg.MODEL.CLASSIFIER.NAME = \"{name}\""
        )

    return classifier


def build_model(cfg: CfgNode) -> nn.Module:
    name = cfg.MODEL.META_ARCHITECTURE.NAME
    
    if name == "ClassificationModelBase":
        model = ClassificationModelBase(
            backbone   = build_backbone(cfg),
            classifier = build_classifier(cfg),
            pixel_mean = cfg.MODEL.PIXEL_MEAN,
            pixel_std  = cfg.MODEL.PIXEL_STD,
        )
    else:
        raise NotImplementedError(
            f"Unknown cfg.MODEL.META_ARCHITECTURE.NAME = \"{name}\""
        )

    return model
