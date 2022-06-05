import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .utils import initialize_tensor


__all__ = [
    "Conv2d",
    "Conv2d_Bezier",
    "Conv2d_BatchEnsemble",
    "Conv2d_BatchEnsembleV2",
    "Conv2d_Dropout",
    "Conv2d_SpatialDropout",
    "Conv2d_DropBlock",
]


class Conv2d(nn.Conv2d):
    
    def __init__(self, *args, **kwargs):
        self.same_padding = kwargs.pop("same_padding", False)
        if self.same_padding:
            kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.same_padding:
            x = self._pad_input(x)
        return self._conv_forward(x, self.weight, self.bias)


class Conv2d_Bezier(Conv2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.ParameterList([self._parameters.pop("weight", None)])
        if self.bias is not None:
            self.bias = nn.ParameterList([self._parameters.pop("bias", None)])

    @torch.no_grad()
    def add_param(self) -> None:
        _p = nn.Parameter(self.weight[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.weight) / len(self.weight))
        self.weight.append(_p)
        if self.bias is not None:
            _p = nn.Parameter(self.bias[-1].detach().clone())
            _p.data.copy_(torch.zeros_like(_p) + sum(self.bias) / len(self.bias))
            self.bias.append(_p)

    def freeze_param(self, index: int) -> None:
        self.weight[index].grad = None
        self.weight[index].requires_grad = False
        if self.bias is not None:
            self.bias[index].grad = None
            self.bias[index].requires_grad = False

    def _sample_parameters(self, λ: float) -> Tuple[torch.Tensor]:
        w = torch.zeros_like(self.weight[0])
        b = torch.zeros_like(self.bias[0]) if self.bias is not None else None

        if len(self.weight) == 1:
            w += self.weight[0]
            if b is not None:
                b += self.bias[0]

        elif len(self.weight) == 2:
            w += (1 - λ) * self.weight[0]
            w += λ * self.weight[1]
            if b is not None:
                b += (1 - λ) * self.bias[0]
                b += λ * self.bias[1]

        elif len(self.weight) == 3:
            w += (1 - λ) * (1 - λ) * self.weight[0]
            w += 2 * (1 - λ) * λ * self.weight[1]
            w += λ * λ * self.weight[2]
            if b is not None:
                b += (1 - λ) * (1 - λ) * self.bias[0]
                b += 2 * (1 - λ) * λ * self.bias[1]
                b += λ * λ * self.bias[2]

        else:
            raise NotImplementedError()

        return w, b

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        kh, kw = self.weight[0].size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        λ = kwargs.pop("bezier_lambda", None)
        weight, bias = self._sample_parameters(λ)
        if self.same_padding:
            x = self._pad_input(x)
        return self._conv_forward(x, weight, bias)


class Conv2d_BatchEnsemble(Conv2d):

    def __init__(self, *args, **kwargs) -> None:
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Conv2d_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_channels)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_channels)
            )
        )
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_channels)
                )
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        initialize_tensor(self.alpha_be, **self.alpha_initializer)
        initialize_tensor(self.gamma_be, **self.gamma_initializer)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, C1, H1, W1 = x.size()
        r_x = x.view(self.ensemble_size, -1, C1, H1, W1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, C1, 1, 1)
        r_x = r_x.view(-1, C1, H1, W1)

        if self.same_padding:
            r_x = self._pad_input(r_x)
        w_r_x = self._conv_forward(r_x, self.weight, self.bias)

        _, C2, H2, W2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, C2, H2, W2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, C2, 1, 1)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, C2, 1, 1)
        s_w_r_x = s_w_r_x.view(-1, C2, H2, W2)

        return s_w_r_x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, ensemble_size={ensemble_size}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            if self.ensemble_bias is None:
                s += ', bias=False, ensemble_bias=False'
            else:
                s += ', bias=False, ensemble_bias=True'
        else:
            if self.ensemble_bias is None:
                s += ', bias=True, ensemble_bias=False'
            else:
                s += ', bias=True, ensemble_bias=True'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Conv2d_BatchEnsembleV2(Conv2d_BatchEnsemble):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, C1, H1, W1 = x.size()
        r_x = x.view(self.ensemble_size, -1, C1, H1, W1)
        r_x = r_x * (1.0 + self.alpha_be.view(self.ensemble_size, 1, C1, 1, 1))
        r_x = r_x.view(-1, C1, H1, W1)

        if self.same_padding:
            r_x = self._pad_input(r_x)
        w_r_x = self._conv_forward(r_x, self.weight, self.bias)

        _, C2, H2, W2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, C2, H2, W2)
        s_w_r_x = s_w_r_x * (1.0 + self.gamma_be.view(self.ensemble_size, 1, C2, 1, 1))
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, C2, 1, 1)
        s_w_r_x = s_w_r_x.view(-1, C2, H2, W2)

        return s_w_r_x


class Conv2d_Dropout(Conv2d):
    
    def __init__(self, *args, **kwargs) -> None:
        drop_p = kwargs.pop("drop_p", None)
        super(Conv2d_Dropout, self).__init__(*args, **kwargs)
        self.drop_p = drop_p

    def _get_masks(self, x: torch.Tensor, seed: int = None) -> torch.Tensor:
        # TODO: handling random seed...
        probs = torch.ones_like(x) * (1.0 - self.drop_p)
        masks = torch.bernoulli(probs)
        return masks

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs.pop("is_drop", False):
            r = self._get_masks(x)
            x = r * x / (1.0 - self.drop_p)
        return super().forward(x, **kwargs)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, drop_p={drop_p}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Conv2d_SpatialDropout(Conv2d):
    
    def __init__(self, *args, **kwargs) -> None:
        drop_p = kwargs.pop("drop_p", None)
        super(Conv2d_SpatialDropout, self).__init__(*args, **kwargs)
        self.drop_p = drop_p

    def _get_masks(self, x: torch.Tensor, seed: int = None) -> torch.Tensor:
        # TODO: handling random seed...
        probs = torch.ones_like(x[:, :, 0, 0]) * (1.0 - self.drop_p)
        masks = torch.bernoulli(probs)[:, :, None, None]
        return masks

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs.pop("is_drop", False):
            r = self._get_masks(x)
            x = r * x / (1.0 - self.drop_p)
        return super().forward(x, **kwargs)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, drop_p={drop_p}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Conv2d_DropBlock(Conv2d):
    
    def __init__(self, *args, **kwargs) -> None:
        drop_p = kwargs.pop("drop_p", None)
        block_size = kwargs.pop("block_size", None)
        use_shared_masks = kwargs.pop("use_shared_masks", None)
        super(Conv2d_DropBlock, self).__init__(*args, **kwargs)
        self.drop_p = drop_p
        self.block_size = block_size
        self.use_shared_masks = use_shared_masks

    def _get_masks(self, x: torch.Tensor, seed: int = None) -> torch.Tensor:
        # TODO: handling random seed...
        
        gamma = self.drop_p / (
            self.block_size ** 2
        ) * (x.size(2) ** 2) / (
            (x.size(2) - self.block_size + 1) ** 2
        )
        
        if self.use_shared_masks:
            probs = torch.ones_like(x[:, 0, :, :]) * gamma
            probs = probs[:, None, :, :]
        else:
            probs = torch.ones_like(x) * gamma

        masks = torch.bernoulli(probs)
        masks = nn.functional.max_pool2d(
            input       = masks,
            kernel_size = self.block_size,
            stride      = 1,
            padding     = self.block_size // 2,
        )
        if self.block_size % 2 == 0:
            masks = masks[:, :, :-1, :-1]
        masks = 1.0 - masks
        return masks

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs.pop("is_drop", False):
            r = self._get_masks(x)
            x = r * x * r.numel() / r.sum()
        return super().forward(x, **kwargs)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, drop_p={drop_p}, block_size={block_size}'
             ', use_shared_masks={use_shared_masks}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
