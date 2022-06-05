import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Tuple


__all__ = [
    "build_sghmc_optimizer",
]


class SGHMC(Optimizer):
    """Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)
    
    Args:
        params (iterable):
            iterable of parameters to optimize or dicts defining parameter groups
        lr (float):
            learning rate
        alpha (float, optional):
            momentum decay factor; alpha=1 is equivalent to SGLD (default: 0.1)
        lr_scale (float, optional):
            scaling factor for numerical stability (default: 1.0)
        weight_decay (float, optional):
            weight decay (L2 penalty) (default: 0.0)
        temperature (float, optional):
            temperature in the posterior (default: 1.0)
    """
    def __init__(self, params, lr, alpha=0.1, lr_scale=1.0, weight_decay=0.0, temperature=1.0) -> None:
        if lr < 0.0:
            raise ValueError("Invalid lr value: {}".format(lr))
        if alpha < 0.0:
            raise ValueError("Inavlid alpha value: {}".format(alpha))
        if lr_scale < 0.0:
            raise ValueError("Invalid lr_scale value: {}".format(lr_scale))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if temperature < 0.0:
            raise ValueError("Invalid temperature value: {}".format(temperature))

        defaults = dict(lr=lr, alpha=alpha, lr_scale=lr_scale,
                        weight_decay=weight_decay, temperature=temperature)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, noise=True):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            alpha        = group["alpha"]
            lr_scale     = group["lr_scale"]
            weight_decay = group["weight_decay"]
            temperature  = group["temperature"]

            # target parameters which need to be updated
            params_with_grad     = []
            d_p_list             = []
            momentum_buffer_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            # update target parameters
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                buf = momentum_buffer_list[i]
                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(1 - alpha).add_(d_p, alpha=-lr)
                if noise:
                    buf.add_(
                        (2.0*lr*alpha*temperature*lr_scale)**0.5 * torch.randn_like(param)
                    )
                
                param.add_(buf)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def build_sghmc_optimizer(model: nn.Module, **kwargs) -> Tuple[Optimizer, List]:

    BASE_LR = kwargs.pop("BASE_LR", None)
    BASE_LR_SCALE = kwargs.pop("BASE_LR_SCALE", None)
    WEIGHT_DECAY = kwargs.pop("WEIGHT_DECAY", None)
    MOMENTUM_DECAY = kwargs.pop("MOMENTUM_DECAY", None)
    TEMPERATURE = kwargs.pop("TEMPERATURE", None)

    _cache = set()
    params = list()
    for module in model.modules():
        
        for module_param_name, value in module.named_parameters(recurse=False):

            if not value.requires_grad:
                continue

            if value in _cache:
                continue
            _cache.add(value)

            schedule_params = dict()
            schedule_params["params"] = [value]
            schedule_params["lr"]           = BASE_LR
            schedule_params["lr_scale"]     = BASE_LR_SCALE
            schedule_params["weight_decay"] = WEIGHT_DECAY
            schedule_params["alpha"]        = MOMENTUM_DECAY
            schedule_params["temperature"]  = TEMPERATURE

            params.append(schedule_params)

    return SGHMC(params, lr=BASE_LR), params
