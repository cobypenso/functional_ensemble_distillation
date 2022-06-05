import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
import src.utils_dir.pytorch as torch_utils


class Model(distilled_network.DistilledNet):
    def __init__(self,
                 layer_sizes,
                 target_dim,
                 teacher,
                 variance_transform=torch_utils.positive_linear_asymptote(),
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(teacher=teacher,
                         loss_function=custom_loss.norm_inv_wish_nll,
                         device=device)

        self.variance_transform = variance_transform
        self.target_dim = target_dim
        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)

        self.to(self.device)

    def forward(self, x):
        """Estimate distribution parameters
        """

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        mu = x[:, :self.target_dim]
        scale = self.variance_transform(x[:, self.target_dim:(self.target_dim +
                                                              1)])
        psi = self.variance_transform(x[:, (self.target_dim +
                                            1):(2 * self.target_dim + 1)])
        # Degrees of freedom should be larger than D - 1
        # Temporary way of defining that
        nu = self.variance_transform(
            x[:, (2 * self.target_dim + 1):]) + (self.target_dim - 1)

        if any(map(torch_utils.is_nan_or_inf, [mu, scale, psi, nu])):
            self._log.debug("mu: {}\nscale: {}\npsi: {}\nnu: {}".format(
                mu, scale, psi, nu))
            if torch_utils.is_nan_or_inf(mu):
                self._log.error("Got NaN in tensor {}: {}".format("mu", mu))
            elif torch_utils.is_nan_or_inf(scale):
                self._log.error("Got NaN in tensor {}: {}".format(
                    "scale", scale))
            elif torch_utils.is_nan_or_inf(psi):
                self._log.error("Got NaN in tensor {}: {}".format("psi", psi))
            elif torch_utils.is_nan_or_inf(nu):
                self._log.error("Got NaN in tensor {}: {}".format("nu", nu))
            else:
                self._log.error("Incosistent Nan/Inf check")

            raise ValueError("Got NaN")

        return mu, scale, psi, nu

    def _generate_teacher_predictions(self, inputs):
        """Generate teacher predictions"""

        logits = self.teacher.predict(inputs).to(self.device)

        return logits

    def predict(self, input_):
        """Predict parameters
        Wrapper function for the forward function.
        """
        return torch.cat(self.forward(input_), dim=-1)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs,
                         (teacher_predictions[:, :, :self.target_dim],
                          teacher_predictions[:, :, self.target_dim:]))
