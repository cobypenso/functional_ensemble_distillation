import torch
import torch.nn as nn


__all__ = [
    "ReLU",
    "SiLU",
    "Identity",
    "MaxPool2d",
]


class ReLU(nn.ReLU):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class SiLU(nn.SiLU):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class Identity(nn.Identity):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class MaxPool2d(nn.MaxPool2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)
