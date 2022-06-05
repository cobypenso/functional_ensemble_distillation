import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble
import src.loss as custom_loss


class MeanRegressor(ensemble.EnsembleMember):
    """MeanRegressor
    Network that predicts the mean value (MSE loss)
    Args:
        layer_sizes (list(int)):
        device (torch.Device)
        learning_rate (float)
    """
    def __init__(self,
                 layer_sizes,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        super().__init__(output_size=layer_sizes[-1],
                         loss_function=custom_loss.mse,
                         device=device)

        self.learning_rate = learning_rate

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.Adam(self.parameters(),
                                          lr=self.learning_rate)
        self.to(self.device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        return x

    def transform_logits(self, logits):
        """TODO: Transform to mu and sigma^2"""
        return logits

    def calculate_loss(self, outputs, targets):
        return self.loss(outputs, targets)

    def predict(self, x):
        logits = self.forward(x)
        x = self.transform_logits(logits)

        return x
