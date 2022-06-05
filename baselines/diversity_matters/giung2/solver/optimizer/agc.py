"""
https://github.com/vballoli/nfnets-pytorch
"""
import torch
from torch.nn import Module
from torch.optim import Optimizer, SGD
from typing import List, Tuple
from collections import Iterable


__all__ = [
    "AGC",
]


class AGC(torch.optim.Optimizer):

    def __init__(
            self,
            params,
            base_optimizer: torch.optim.Optimizer,
            clipping: float = 1e-2,
            eps: float = 1e-3,
            model: torch.nn.Module = None,
            ignored_params: Iterable = ["classifier.fc",],
        ):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.base_optimizer = base_optimizer

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **base_optimizer.defaults}

        if not isinstance(ignored_params, Iterable):
            ignored_params = [ignored_params]

        if model is not None:
            assert ignored_params not in [
                None, []], "You must specify ignored_params for AGC to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]

            for module_name in ignored_params:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name))
            params = [{"params": list(module.parameters())} for name,
                          module in model.named_modules() if name not in ignored_params]
        
        else:
            params = [{"params": params}]

        self.agc_params = params
        self.eps = eps
        self.clipping = clipping
        
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.agc_params:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = torch.max(
                    self._unitwise_norm(p.detach()),
                    torch.tensor(self.eps).to(p.device),
                )
                grad_norm = self._unitwise_norm(p.grad.detach())
                max_norm = param_norm * self.clipping

                trigger = grad_norm > max_norm

                clipped_grad = p.grad * (
                    max_norm / torch.max(
                        grad_norm, torch.tensor(1e-6).to(grad_norm.device)
                    )
                )
                p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        for group in self.agc_params:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def _unitwise_norm(self, x: torch.Tensor):
        if x.ndim <= 1:
            dim = 0
            keepdim = False
        elif x.ndim in [2, 3]:
            dim = 0
            keepdim = True
        elif x.ndim == 4:
            dim = [1, 2, 3]
            keepdim = True
        else:
            raise ValueError('Wrong input dimensions')

        return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5
