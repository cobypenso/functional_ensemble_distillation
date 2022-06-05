import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .utils import initialize_tensor


__all__ = [
    "Linear",
    "Linear_Bezier",
    "Linear_BatchEnsemble",
    "Linear_BatchEnsembleV2",
    "Linear_Dropout",
]


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(x)


class Linear_Bezier(Linear):

    def __init__(self, *args, **kwargs) -> None:
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        λ = kwargs.pop("bezier_lambda", None)
        weight, bias = self._sample_parameters(λ)
        return F.linear(x, weight, bias)


class Linear_BatchEnsemble(Linear):

    def __init__(self, *args, **kwargs) -> None:
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Linear_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_features)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_features)
            )
        )
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_features)
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

        _, D1 = x.size()
        r_x = x.view(self.ensemble_size, -1, D1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ensemble_size={}, ensemble_bias={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.ensemble_size, self.ensemble_bias is not None
        )


class Linear_BatchEnsembleV2(Linear_BatchEnsemble):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, D1 = x.size()
        r_x = x.view(self.ensemble_size, -1, D1)
        r_x = r_x * (1.0 + self.alpha_be.view(self.ensemble_size, 1, D1))
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * (1.0 + self.gamma_be.view(self.ensemble_size, 1, D2))
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x


class Linear_Dropout(Linear):
    
    def __init__(self, *args, **kwargs) -> None:
        drop_p = kwargs.pop("drop_p", None)
        super(Linear_Dropout, self).__init__(*args, **kwargs)
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

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, drop_p={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.drop_p
        )
