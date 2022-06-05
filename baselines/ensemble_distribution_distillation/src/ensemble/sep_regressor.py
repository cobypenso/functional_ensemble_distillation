import itertools
import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble
from src.ensemble import mean_regressor
import src.loss as custom_loss


class SepRegressor(ensemble.EnsembleMember):
    """SepRegressor
    Network that predicts the parameters of a normal distribution

    Args:
        layer_sizes (list(int)): Defines the (equal) subnetworks,
            i.e. the last element (output_size) is D
        device (torch.Device)
        learning_rate (float)
    """
    def __init__(self,
                 layer_sizes,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        # The actual output the output of the combined subnetworks
        super().__init__(output_size=layer_sizes[-1] * 2,
                         target_size=layer_sizes[-1],
                         loss_function=custom_loss.gaussian_neg_log_likelihood,
                         device=device)

        self.mean_only = False
        self.learning_rate = learning_rate
        self.mu_network = mean_regressor.MeanRegressor(
            layer_sizes=layer_sizes,
            device=device,
            learning_rate=self.learning_rate)

        self.sigma_sq_network = mean_regressor.MeanRegressor(
            layer_sizes=layer_sizes,
            device=device,
            learning_rate=self.learning_rate)

        self.optimizer = torch_optim.SGD(itertools.chain(
            self.mu_network.parameters(), self.sigma_sq_network.parameters()),
                                         lr=self.learning_rate,
                                         momentum=0.9)
        self.to(self.device)
        self.active_network = "mu"
        self.switch_active_network(self.active_network)
        self.variance_lower_bound = 0.001

    def forward(self, x):
        mu = self.mu_network.forward(x)
        sigma_sq_logit = self.sigma_sq_network.forward(x)
        logits = torch.cat((mu, sigma_sq_logit), dim=1)

        return logits

    def transform_logits(self, logits):
        # mean = logits[:, :int((self.output_size / 2))]
        # var = torch.exp(logits[:, int((self.output_size / 2)):])

        outputs = logits
        outputs[:, 1] = torch.log(
            1 + torch.exp(outputs[:, 1])) + self.variance_lower_bound

        return outputs

    def switch_active_network(self, network):
        if network == "mu":
            self.active_network = "mu"
            for param in self.mu_network.parameters():
                param.requires_grad = True
            for param in self.sigma_sq_network.parameters():
                param.requires_grad = False

        elif network == "sigma_sq":
            self.active_network = "sigma_sq"
            for param in self.mu_network.parameters():
                param.requires_grad = False
            for param in self.sigma_sq_network.parameters():
                param.requires_grad = True

        elif network == "both":
            self.active_network = "both"
            for param in self.mu_network.parameters():
                param.requires_grad = True
            for param in self.sigma_sq_network.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Inconsistent network")

    def calculate_loss(self, outputs, targets):
        mean = outputs[:, 0].reshape((outputs.size(0), 1))
        var = outputs[:, 1].reshape((outputs.size(0), 1))
        parameters = (mean, var)
        loss = None
        if self.active_network == "mu":
            loss = custom_loss.mse(mean, targets)
        else:
            loss = self.loss(parameters, targets)
        return loss

    def predict(self, x):
        logits = self.forward(x)
        x = self.transform_logits(logits)

        return x

    def _output_to_metric_domain(self, outputs):
        """Transform output for metric calculation
        Output distribution parameters are not necessarily
        exact representation for metrics calculation.
        This helper function can be overloaded to massage the output
        into the correct shape

        Extracts mean value
        """
        return outputs[:, 0]
