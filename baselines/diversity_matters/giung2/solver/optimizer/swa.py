import torch
from torch.optim import Optimizer


__all__ = [
    "SWA",
]


class SWA(Optimizer):
    """Stochastic Weight Averaging (SWA)
    
    Args:
        base_optimizer (Optimizer):
            optimizer to use with SWA
    """
    def __init__(self, base_optimizer) -> None:
        self.base_optimizer = base_optimizer
        self.defaults       = self.base_optimizer.defaults
        self.param_groups   = self.base_optimizer.param_groups
        self.state          = self.base_optimizer.state
        for group in self.param_groups:
            group["n_avg"] = 0

    @torch.no_grad()
    def step(self, closure=None, sampling=False):
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            
            for p in group["params"]:
                state = self.state[p]
                
                # save current parameters
                state["sgd_buffer"] = p.data
                
                # update SWA solution
                if sampling:
                    if "swa_buffer" not in state:
                        state["swa_buffer"] = torch.zeros_like(state["sgd_buffer"])
                    state["swa_buffer"].add_(
                        state["sgd_buffer"] - state["swa_buffer"],
                        alpha = 1.0 / float(group["n_avg"] + 1)
                    )
            
            if sampling:
                group["n_avg"] += 1
        
        return loss

    @torch.no_grad()
    def load_swa_buffer(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                p.data.copy_(state["swa_buffer"])

    @torch.no_grad()
    def load_sgd_buffer(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                p.data.copy_(state["sgd_buffer"])
