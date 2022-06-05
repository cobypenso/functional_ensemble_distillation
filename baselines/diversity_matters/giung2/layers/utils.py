import torch
import torch.nn as nn
from typing import List


def initialize_tensor(
        tensor: torch.Tensor,
        initializer: str,
        init_values: List[float] = [],
    ) -> None:

    if initializer == "zeros":
        nn.init.zeros_(tensor)

    elif initializer == "ones":
        nn.init.ones_(tensor)

    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])

    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])

    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0 * init_values[1] * torch.bernoulli(
                    torch.zeros_like(tensor) + init_values[0]
                ) - init_values[1]
            )

    else:
        raise NotImplementedError(
            f"Unknown initializer: {initializer}"
        )
