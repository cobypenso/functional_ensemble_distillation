import torch
from torch.optim import Optimizer


__all__ = [
    "SAM",
]


class SAM(Optimizer):
    """Sharpness-Aware Minimization (SAM)
    
    Args:
        base_optimizer (Optimizer):
            optimizer to use with SAM
        rho (float)
    """
    def __init__(self, base_optimizer, rho=0.05) -> None:
        self.base_optimizer = base_optimizer
        self.defaults       = self.base_optimizer.defaults
        self.param_groups   = self.base_optimizer.param_groups
        self.state          = self.base_optimizer.state
        for group in self.param_groups:
            group["rho"] = rho

    @torch.no_grad()
    def first_step(self):

        grad_norm = torch.stack([
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"] if p.grad is not None
        ]).norm(p=2)

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale)

    @torch.no_grad()
    def second_step(self):

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()
