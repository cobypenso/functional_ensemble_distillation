import torch
import torch.nn as nn
from typing import Dict, List
from functools import partial

from fvcore.common.config import CfgNode
from giung2.layers import *


__all__ = [
    "build_resnet_backbone",
]


class IdentityShortcut(nn.Module):

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int,
            expansion: int,
            conv: nn.Module = Conv2d,
            norm: nn.Module = BatchNorm2d,
            relu: nn.Module = ReLU,
            **kwargs
        ) -> None:
        super(IdentityShortcut, self).__init__()
        self.identity = MaxPool2d(kernel_size=1, stride=stride)
        self.pad_size = expansion * planes - in_planes

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.identity(x)
        out = nn.functional.pad(out, (0, 0, 0, 0, 0, self.pad_size), mode="constant", value=0)
        return out


class ProjectionShortcut(nn.Module):

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int,
            expansion: int,
            conv: nn.Module = Conv2d,
            norm: nn.Module = BatchNorm2d,
            relu: nn.Module = ReLU,
            **kwargs
        ) -> None:
        super(ProjectionShortcut, self).__init__()
        self.conv = conv(in_channels=in_planes, out_channels=expansion*planes,
                         kernel_size=1, stride=stride, padding=0, **kwargs)
        self.norm = norm(num_features=expansion*planes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.norm(self.conv(x, **kwargs), **kwargs)
        return out


class FirstBlock(nn.Module):

    def __init__(
            self,
            in_planes: int,
            planes: int,
            conv: nn.Module,
            conv_ksp: List[int],
            norm: nn.Module,
            relu: nn.Module,
            pool: nn.Module,
            pool_ksp: List[int],
            **kwargs
        ) -> None:
        super(FirstBlock, self).__init__()
        self.conv1 = conv(in_channels=in_planes, out_channels=planes,
                          kernel_size=conv_ksp[0], stride=conv_ksp[1], padding=conv_ksp[2], **kwargs)
        self.norm1 = norm(num_features=planes)
        self.relu1 = relu()
        self.pool1 = pool(kernel_size=pool_ksp[0], stride=pool_ksp[1], padding=pool_ksp[2])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.pool1(self.relu1(self.norm1(self.conv1(x, **kwargs), **kwargs), **kwargs), **kwargs)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int,
            shortcut: nn.Module,
            conv: nn.Module = Conv2d,
            norm: nn.Module = BatchNorm2d,
            relu: nn.Module = ReLU,
            **kwargs
        ) -> None:
        super(BasicBlock,self).__init__()
        self.conv1 = conv(in_channels=in_planes, out_channels=planes,
                          kernel_size=3, stride=stride, padding=1, **kwargs)
        self.norm1 = norm(num_features=planes)
        self.relu1 = relu()
        self.conv2 = conv(in_channels=planes, out_channels=self.expansion*planes,
                          kernel_size=3, stride=1, padding=1, **kwargs)
        self.norm2 = norm(num_features=self.expansion*planes)
        self.relu2 = relu()
        if stride != 1  or in_planes != self.expansion * planes:
            self.shortcut = shortcut(
                in_planes, planes, stride, self.expansion, conv, norm, **kwargs
            )
        else:
            self.shortcut = Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.relu1(self.norm1(self.conv1(x,   **kwargs), **kwargs), **kwargs)
        out = self.relu2(self.norm2(self.conv2(out, **kwargs), **kwargs) + self.shortcut(x, **kwargs), **kwargs)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int,
            shortcut: nn.Module,
            conv: nn.Module = Conv2d,
            norm: nn.Module = BatchNorm2d,
            relu: nn.Module = ReLU,
            **kwargs
        ) -> None:
        super(Bottleneck,self).__init__()
        self.conv1 = conv(in_channels=in_planes, out_channels=planes,
                          kernel_size=1, stride=1, padding=0, **kwargs)
        self.norm1 = norm(num_features=planes)
        self.relu1 = relu()
        self.conv2 = conv(in_channels=planes, out_channels=planes,
                          kernel_size=3, stride=stride, padding=1, **kwargs)
        self.norm2 = norm(num_features=planes)
        self.relu2 = relu()
        self.conv3 = conv(in_channels=planes, out_channels=self.expansion*planes,
                          kernel_size=1, stride=1, padding=0, **kwargs)
        self.norm3 = norm(num_features=self.expansion*planes)
        self.relu3 = relu()
        if stride != 1  or in_planes != self.expansion * planes:
            self.shortcut = shortcut(
                in_planes, planes, stride, self.expansion, conv, norm, **kwargs
            )
        else:
            self.shortcut = Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.relu1(self.norm1(self.conv1(x,   **kwargs), **kwargs), **kwargs)
        out = self.relu2(self.norm2(self.conv2(out, **kwargs), **kwargs), **kwargs)
        out = self.relu3(self.norm3(self.conv3(out, **kwargs), **kwargs) + self.shortcut(x, **kwargs), **kwargs)
        return out


class ResNet(nn.Module):

    def __init__(
            self,
            channels: int,
            in_planes: int,
            first_block: nn.Module,
            block: nn.Module,
            shortcut: nn.Module,
            num_blocks: List[int],
            widen_factor: int,
            conv: nn.Module = Conv2d,
            norm: nn.Module = BatchNorm2d,
            relu: nn.Module = ReLU,
            **kwargs
        ) -> None:
        super(ResNet, self).__init__()
        self.channels     = channels
        self.in_planes    = in_planes
        self._in_planes   = in_planes
        self.first_block  = first_block
        self.block        = block
        self.shortcut     = shortcut
        self.num_blocks   = num_blocks
        self.widen_factor = widen_factor
        self.conv         = conv
        self.norm         = norm
        self.relu         = relu

        _layers = [self.first_block(in_planes=self.channels, planes=self.in_planes, **kwargs)]

        _layers += self._make_layer(
            self.in_planes * self.widen_factor, self.num_blocks[0], stride=1, **kwargs
        )
        for idx, num_block in enumerate(self.num_blocks[1:], start=1):
            _layers += self._make_layer(
                self.in_planes * (2 ** idx) * self.widen_factor, num_block, stride=2, **kwargs
            )
        self.layers = nn.Sequential(*_layers)

    def _make_layer(self, planes: int, num_block: int, stride: int, **kwargs) -> List[nn.Module]:
        strides = [stride] + [1] * (num_block - 1)
        _layers = []
        for stride in strides:
            _layers.append(self.block(self._in_planes, planes, stride,
                                      self.shortcut, self.conv, self.norm, self.relu, **kwargs))
            self._in_planes = planes * self.block.expansion
        return _layers

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:

        outputs = dict()

        # intermediate feature maps
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, **kwargs)
            outputs[f"layer{layer_idx}"] = x

        # final feature vector
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        outputs["features"] = x

        return outputs


def build_resnet_backbone(cfg: CfgNode) -> nn.Module:

    # Conv2d layers may be replaced by its variations
    _conv_layers = cfg.MODEL.BACKBONE.RESNET.CONV_LAYERS
    kwargs = {
        "bias": cfg.MODEL.BACKBONE.RESNET.CONV_LAYERS_BIAS,
        "same_padding": cfg.MODEL.BACKBONE.RESNET.CONV_LAYERS_SAME_PADDING,
    }
    if _conv_layers == "Conv2d":
        conv_layers = Conv2d
    elif _conv_layers == "Conv2d_Bezier":
        conv_layers = Conv2d_Bezier
    elif _conv_layers in ["Conv2d_BatchEnsemble", "Conv2d_BatchEnsembleV2",]:
        if cfg.MODEL.BATCH_ENSEMBLE.ENABLED is False:
            raise AssertionError(
                f"Set MODEL.BATCH_ENSEMBLE.ENABLED=True to use {_conv_layers}"
            )
        if _conv_layers == "Conv2d_BatchEnsemble":
            conv_layers = Conv2d_BatchEnsemble
        if _conv_layers == "Conv2d_BatchEnsembleV2":
            conv_layers = Conv2d_BatchEnsembleV2
        kwargs.update({
            "ensemble_size": cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE,
            "use_ensemble_bias": cfg.MODEL.BATCH_ENSEMBLE.USE_ENSEMBLE_BIAS,
            "alpha_initializer": {
                "initializer": cfg.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.NAME,
                "init_values": cfg.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.VALUES,
            },
            "gamma_initializer": {
                "initializer": cfg.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.NAME,
                "init_values": cfg.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.VALUES,
            },
        })
    elif _conv_layers == "Conv2d_Dropout":
        if cfg.MODEL.DROPOUT.ENABLED is False:
            raise AssertionError(
                f"Set MODEL.DROPOUT.ENABLED=True to use {_conv_layers}"
            )
        conv_layers = Conv2d_Dropout
        kwargs.update({
            "drop_p": cfg.MODEL.DROPOUT.DROP_PROBABILITY,
        })
    elif _conv_layers == "Conv2d_SpatialDropout":
        if cfg.MODEL.SPATIAL_DROPOUT.ENABLED is False:
            raise AssertionError(
                f"Set MODEL.SPATIAL_DROPOUT.ENABLED=True to use {_conv_layers}"
            )
        conv_layers = Conv2d_SpatialDropout
        kwargs.update({
            "drop_p": cfg.MODEL.SPATIAL_DROPOUT.DROP_PROBABILITY,
        })
    elif _conv_layers == "Conv2d_DropBlock":
        if cfg.MODEL.DROP_BLOCK.ENABLED is False:
            raise AssertionError(
                f"Set MODEL.DROP_BLOCK.ENABLED=True to use {_conv_layers}"
            )
        conv_layers = Conv2d_DropBlock
        kwargs.update({
            "drop_p": cfg.MODEL.DROP_BLOCK.DROP_PROBABILITY,
            "block_size": cfg.MODEL.DROP_BLOCK.BLOCK_SIZE,
            "use_shared_masks": cfg.MODEL.DROP_BLOCK.USE_SHARED_MASKS,
        })
    else:
        raise NotImplementedError(
            f"Unknown MODEL.BACKBONE.RESNET.CONV_LAYERS: {_conv_layers}"
        )

    # BatchNorm2d layers may be replaced by its variations
    _norm_layers = cfg.MODEL.BACKBONE.RESNET.NORM_LAYERS
    if _norm_layers == "NONE":
        norm_layers = Identity
    elif _norm_layers == "BatchNorm2d":
        norm_layers = BatchNorm2d
    elif _norm_layers == "GroupNorm2d":
        norm_layers = partial(GroupNorm2d, num_groups=cfg.MODEL.BACKBONE.RESNET.IN_PLANES // 2)
    elif _norm_layers == "FilterResponseNorm2d":
        norm_layers = FilterResponseNorm2d
    elif _norm_layers == "FilterResponseNorm2d_Bezier":
        norm_layers = FilterResponseNorm2d_Bezier
    else:
        raise NotImplementedError(
            f"Unknown MODEL.BACKBONE.RESNET.NORM_LAYERS: {_norm_layers}"
        )

    # ReLU layers may be replaced by its variations
    _activations = cfg.MODEL.BACKBONE.RESNET.ACTIVATIONS
    if _activations == "NONE":
        activations = Identity
    elif _activations == "ReLU":
        activations = ReLU
    elif _activations == "SiLU":
        activations = SiLU
    else:
        raise NotImplementedError(
            f"Unknown MODEL.BACKBONE.RESNET.ACTIVATIONS: {_activations}"
        )

    # specify the first block
    first_block = partial(
        FirstBlock,
        conv     = conv_layers,
        conv_ksp = cfg.MODEL.BACKBONE.RESNET.FIRST_BLOCK.CONV_KSP,
        norm     = norm_layers if cfg.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_NORM_LAYER else Identity,
        relu     = activations if cfg.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_ACTIVATION else Identity,
        pool     = MaxPool2d   if cfg.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_POOL_LAYER else Identity,
        pool_ksp = cfg.MODEL.BACKBONE.RESNET.FIRST_BLOCK.POOL_KSP,
    )

    # specify block
    _block = cfg.MODEL.BACKBONE.RESNET.BLOCK
    if _block == "BasicBlock":
        block = BasicBlock
    elif _block == "Bottleneck":
        block = Bottleneck
    else:
        raise NotImplementedError(
            f"Unknown MODEL.BACKBONE.RESNET.BLOCK: {_block}"
        )

    # specify shortcut
    _shortcut = cfg.MODEL.BACKBONE.RESNET.SHORTCUT
    if _shortcut == "IdentityShortcut":
        shortcut = IdentityShortcut
    elif _shortcut == "ProjectionShortcut":
        shortcut = ProjectionShortcut
    else:
        raise NotImplementedError(
            f"Unknown MODEL.BACKBONE.RESNET.SHORTCUT: {_shortcut}"
        )

    # build backbone
    backbone = ResNet(
        channels     = cfg.MODEL.BACKBONE.RESNET.CHANNELS,
        in_planes    = cfg.MODEL.BACKBONE.RESNET.IN_PLANES,
        first_block  = first_block,
        block        = block,
        shortcut     = shortcut,
        num_blocks   = cfg.MODEL.BACKBONE.RESNET.NUM_BLOCKS,
        widen_factor = cfg.MODEL.BACKBONE.RESNET.WIDEN_FACTOR,
        conv         = conv_layers,
        norm         = norm_layers,
        relu         = activations,
        **kwargs
    )

    # initialize weights
    for m in backbone.modules():
        if isinstance(m, Conv2d):
            if isinstance(m.weight, nn.ParameterList):
                for idx in range(len(m.weight)):
                    nn.init.kaiming_normal_(m.weight[idx], mode="fan_out", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    return backbone
