import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
from src import utils
import src.utils_dir.pytorch as torch_utils


class Model(distilled_network.DistilledNet):
    def __init__(self,
                 layer_sizes,
                 teacher,
                 variance_transform=utils.positive_linear_asymptote(),
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001,
                 scale_teacher_logits=False):

        super().__init__(teacher=teacher,
                         loss_function=custom_loss.gaussian_neg_log_likelihood,
                         device=device)

        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate
        self.scale_teacher_logits = scale_teacher_logits

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.variance_transform = variance_transform
        self.optimizer = torch_optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

        self.to(self.device)

    def info(self):
        """Get model settings"""
        return {
            "name":
            "gauss_logits",
            "layer_sizes":
            torch_utils.human_readable_arch(self.layers),
            "loss_function":
            self.loss.__name__,
            "variance_transform":
            torch_utils.human_readable_lambda(self.variance_transform)
        }

    def forward(self, x):
        """Estimate parameters of distribution
        """

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        mid = int(x.shape[-1] / 2)
        mean = x[:, :mid]
        var_z = x[:, mid:]

        var = self.variance_transform(var_z)
        if torch.isinf(var).sum() > 0:
            raise ValueError("Got NaN")

        return mean, var

    def _generate_teacher_predictions(self, inputs):
        """Generate teacher predictions"""

        logits = self.teacher.get_logits(inputs).to(self.device)

        return logits

    def predict(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        samples = self.predict_logits(input_)

        return nn.Softmax(dim=-1)(samples)

    def predict_logits(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if num_samples is None:
            num_samples = 50

        mean, var = self.forward(input_)

        samples = torch.zeros(
            [input_.size(0), num_samples,
             int(self.output_size / 2)])
        for i in range(input_.size(0)):

            rv = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mean[i, :], covariance_matrix=torch.diag(var[i, :]))

            samples[i, :, :] = rv.rsample([num_samples])

        if self.scale_teacher_logits:
            samples = torch.cat(
                (samples, torch.zeros(samples.size(0), num_samples, 1)))

        return samples

    def _learning_rate_condition(self, epoch=None):
        """Evaluate condition for increasing learning rate
        Defaults to never increasing. I.e. returns False
        """

        return True

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """

        return self.loss(outputs, teacher_predictions)

    def mean_expected_value(self, outputs, teacher_predictions):
        exp_value = outputs[0]

        return torch.mean(exp_value, dim=0)

    def mean_variance(self, outputs, teacher_predictions):
        variance = outputs[1]

        return torch.mean(variance, dim=0)
