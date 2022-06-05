import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Tuple


__all__ = [
    "build_sgd_optimizer",
]


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD)
    
    Args:
        params (iterable):
            iterable of parameters to optimize or dicts defining parameter groups
        lr (float):
            learning rate
        momentum (float, optional):
            momentum factor (default: 0.0)
        weight_decay (float, optional):
            weight decay (L2 penalty) (default: 0.0)
        nesterov (bool, optional):
            enables Nesterov momentum (default: False)
        decoupled_weight_decay (bool, optional):
            enabled decoupled weight decay regularization (default: False)
    """
    def __init__(self, params, lr, momentum=0, weight_decay=0,
                 nesterov=False, decoupled_weight_decay=False) -> None:
        if lr < 0.0:
            raise ValueError("Invalid lr value: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, decoupled_weight_decay=decoupled_weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr                     = group['lr']
            momentum               = group['momentum']
            weight_decay           = group['weight_decay']
            nesterov               = group['nesterov']
            decoupled_weight_decay = group['decoupled_weight_decay']

            # target parameters which need to be updated
            params_with_grad     = []
            d_p_list             = []
            momentum_buffer_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # update target parameters
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]

                if weight_decay != 0 and not decoupled_weight_decay:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        
                        buf.mul_(momentum).add_(d_p)
                    if nesterov:
                        d_p = d_p.add(buf)
                    else:
                        d_p = buf

                param.add_(d_p, alpha=-lr)

                if weight_decay != 0 and decoupled_weight_decay:
                    param.add_(weight_decay, alpha=-lr)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def build_sgd_optimizer(model: nn.Module, **kwargs) -> Tuple[Optimizer, List]:

    # basic options 
    BASE_LR                   = kwargs.pop("BASE_LR", None)
    WEIGHT_DECAY              = kwargs.pop("WEIGHT_DECAY", None)
    MOMENTUM                  = kwargs.pop("MOMENTUM", None)
    NESTEROV                  = kwargs.pop("NESTEROV", None)
    DECOUPLED_WEIGHT_DECAY    = kwargs.pop("DECOUPLED_WEIGHT_DECAY", None)

    # options for BatchEnsemble
    SUFFIX_BE                 = kwargs.pop("SUFFIX_BE", tuple())
    BASE_LR_BE                = kwargs.pop("BASE_LR_BE", None)
    WEIGHT_DECAY_BE           = kwargs.pop("WEIGHT_DECAY_BE", None)
    MOMENTUM_BE               = kwargs.pop("MOMENTUM_BE", None)
    NESTEROV_BE               = kwargs.pop("NESTEROV_BE", None)
    DECOUPLED_WEIGHT_DECAY_BE = kwargs.pop("DECOUPLED_WEIGHT_DECAY_BE", None)

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

            if module_param_name.endswith(tuple(SUFFIX_BE)):
                schedule_params["lr"]                     = BASE_LR_BE
                schedule_params["weight_decay"]           = WEIGHT_DECAY_BE
                schedule_params["momentum"]               = MOMENTUM_BE
                schedule_params["nesterov"]               = NESTEROV_BE
                schedule_params["decoupled_weight_decay"] = DECOUPLED_WEIGHT_DECAY_BE

            else:
                schedule_params["lr"]                     = BASE_LR
                schedule_params["weight_decay"]           = WEIGHT_DECAY
                schedule_params["momentum"]               = MOMENTUM
                schedule_params["nesterov"]               = NESTEROV
                schedule_params["decoupled_weight_decay"] = DECOUPLED_WEIGHT_DECAY

            params.append(schedule_params)
    
    return SGD(params, lr=BASE_LR), params
