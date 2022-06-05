import torch
import torch.nn as nn
from typing import Tuple
from .utils import initialize_tensor


__all__ = [
    "BatchNorm2d",
    "GroupNorm2d",
    "FilterResponseNorm2d",
    "FilterResponseNorm2d_Bezier",
]


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class _GroupNorm(nn.GroupNorm):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


def GroupNorm2d(
        num_groups: int,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> nn.Module:
    return _GroupNorm(num_groups, num_features, eps, affine)


class FilterResponseNorm2d(nn.Module):

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-6,
            learnable_eps: bool = False,
            learnable_eps_init: float = 1e-4,
        ) -> None:
        super(FilterResponseNorm2d, self).__init__()
        self.num_features       = num_features
        self.eps                = eps
        self.learnable_eps      = learnable_eps
        self.learnable_eps_init = learnable_eps_init

        self.gamma_frn = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta_frn  = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.tau_frn   = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        if self.learnable_eps:
            self.eps_l_frn = nn.Parameter(torch.Tensor(1))
        else:
            self.register_buffer(
                name="eps_l_frn",
                tensor=torch.zeros(1),
                persistent=False
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.gamma_frn)
        nn.init.zeros_(self.beta_frn)
        nn.init.zeros_(self.tau_frn)
        if self.learnable_eps:
            nn.init.constant_(self.eps_l_frn, self.learnable_eps_init)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def extra_repr(self):
        return '{num_features}, eps={eps}, learnable_eps={learnable_eps}'.format(**self.__dict__)

    def _norm_forward(
            self,
            x: torch.Tensor,
            γ: torch.Tensor,
            β: torch.Tensor,
            τ: torch.Tensor,
            ε: torch.Tensor,
        ) -> torch.Tensor:
        ν2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(ν2 + ε)
        x = γ * x + β
        x = torch.max(x, τ)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self._check_input_dim(x)
        return self._norm_forward(x, self.gamma_frn, self.beta_frn,
                                  self.tau_frn, self.eps + self.eps_l_frn.abs())


class FilterResponseNorm2d_Bezier(FilterResponseNorm2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma_frn = nn.ParameterList([self._parameters.pop("gamma_frn", None)])
        self.beta_frn = nn.ParameterList([self._parameters.pop("beta_frn", None)])
        self.tau_frn = nn.ParameterList([self._parameters.pop("tau_frn", None)])
        if "eps_l_frn" in self._parameters:
            self.eps_l_frn = nn.ParameterList([self._parameters.pop("eps_l_frn", None)])

    @torch.no_grad()
    def add_param(self) -> None:
        _p = nn.Parameter(self.gamma_frn[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.gamma_frn) / len(self.gamma_frn))
        self.gamma_frn.append(_p)
        _p = nn.Parameter(self.beta_frn[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.beta_frn) / len(self.beta_frn))
        self.beta_frn.append(_p)
        _p = nn.Parameter(self.tau_frn[-1].detach().clone())
        _p.data.copy_(torch.zeros_like(_p) + sum(self.tau_frn) / len(self.tau_frn))
        self.tau_frn.append(_p)
        if isinstance(self.eps_l_frn, nn.ParameterList):
            _p = nn.Parameter(self.eps_l_frn[-1].detach().clone())
            _p.data.copy_(torch.zeros_like(_p) + sum(self.eps_l_frn) / len(self.eps_l_frn))
            self.eps_l_frn.append(_p)

    def freeze_param(self, index: int) -> None:
        self.gamma_frn[index].grad = None
        self.gamma_frn[index].requires_grad = False
        self.beta_frn[index].grad = None
        self.beta_frn[index].requires_grad = False
        self.tau_frn[index].grad = None
        self.tau_frn[index].requires_grad = False
        if isinstance(self.eps_l_frn, nn.ParameterList):
            self.eps_l_frn[index].grad = None
            self.eps_l_frn[index].requires_grad = False

    def _sample_parameters(self, λ: float) -> Tuple[torch.Tensor]:
        g = torch.zeros_like(self.gamma_frn[0])
        b = torch.zeros_like(self.beta_frn[0])
        t = torch.zeros_like(self.tau_frn[0])
        e = torch.zeros_like(self.eps_l_frn[0]) if isinstance(self.eps_l_frn, nn.ParameterList) else self.eps_l_frn

        if len(self.gamma_frn) == 1:
            g += self.gamma_frn[0]
            b += self.beta_frn[0]
            t += self.tau_frn[0]
            if isinstance(self.eps_l_frn, nn.ParameterList):
                e += self.eps_l_frn[0]

        elif len(self.gamma_frn) == 2:
            g += (1 - λ) * self.gamma_frn[0]
            g += λ * self.gamma_frn[1]
            b += (1 - λ) * self.beta_frn[0]
            b += λ * self.beta_frn[1]
            t += (1 - λ) * self.tau_frn[0]
            t += λ * self.tau_frn[1]
            if isinstance(self.eps_l_frn, nn.ParameterList):
                e += (1 - λ) * self.eps_l_frn[0]
                e += λ * self.eps_l_frn[1]

        elif len(self.gamma_frn) == 3:
            g += (1 - λ) * (1 - λ) * self.gamma_frn[0]
            g += 2 * (1 - λ) * λ * self.gamma_frn[1]
            g += λ * λ * self.gamma_frn[2]
            b += (1 - λ) * (1 - λ) * self.beta_frn[0]
            b += 2 * (1 - λ) * λ * self.beta_frn[1]
            b += λ * λ * self.beta_frn[2]
            t += (1 - λ) * (1 - λ) * self.tau_frn[0]
            t += 2 * (1 - λ) * λ * self.tau_frn[1]
            t += λ * λ * self.tau_frn[2]
            if isinstance(self.eps_l_frn, nn.ParameterList):
                e += (1 - λ) * (1 - λ) * self.eps_l_frn[0]
                e += 2 * (1 - λ) * λ * self.eps_l_frn[1]
                e += λ * λ * self.eps_l_frn[2]

        else:
            raise NotImplementedError()

        e = e.abs() + self.eps

        return g, b, t, e

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self._check_input_dim(x)
        λ = kwargs.pop("bezier_lambda", None)
        g, b, t, e = self._sample_parameters(λ)
        return self._norm_forward(x, g, b, t, e)
