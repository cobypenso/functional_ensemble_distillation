import torch
import torch.optim as torch_optim
import torch.nn as nn
import src.distilled.distilled_network as distilled_network
import src.loss as custom_loss
import src.utils as utils


class Model(distilled_network.DistilledNet):
    """Simple regressor model
    Network that predicts the parameters of a normal distribution
    """
    def __init__(self,
                 layer_sizes,
                 loss_function,
                 teacher,
                 device=torch.device("cpu"),
                 variance_transform=utils.positive_linear_asymptote()):

        super().__init__(teacher=teacher,
                         loss_function=loss_function,
                         device=device)

        self.variance_transform = variance_transform
        self._log.info("Using variance transform: {}".format(
            self.variance_transform.__name__))

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = None
        self.to(self.device)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        logits = self.layers[-1](x)
        return self.transform_logits(logits)

    def transform_logits(self, logits):
        """Transform logits for the simple regressor

        Maps half of the "logits" from unbounded to positive real numbers.
        TODO: Only works for one-dim output.

        Args:
            logits (torch.Tensor(B, D)):
        """

        mean = logits[:, :1]
        var = self.variance_transform(logits[:, 1:])

        return (mean, var)

    def calculate_loss(self, outputs, targets, _labels):
        return self.loss(outputs, targets)

    def predict(self, x):
        x = x.to(self.device)
        logits = self.forward(x)
        mean, var = self.transform_logits(logits)
        output = torch.stack([mean, var], dim=1).reshape(
            (mean.size(0), self.output_size * 2))

        return output

    def _generate_teacher_predictions(self, inputs):
        """Generate teacher predictions"""

        predictions = self.teacher.predict(inputs)
        mean, var = predictions[:, :, 0].to(
            self.device), predictions[:, :, 1].to(self.device)
        return mean, var

    def _output_to_metric_domain(self, metric, outputs):
        """Transform output for metric calculation

        Extract expected value parameter from outputs
        """
        metric_output = None
        if metric.name is not None:
            if metric.name == "MSE":
                metric_output = outputs[0]
            elif metric.name == "RMSE":
                metric_output = outputs[0]
            else:
                self._log.error(
                    "Metric transform not implemented for: {}".format(
                        metric.name))
        else:
            self._log.error(
                "Metric: {} has no 'name' attribute".format(metric))

        return metric_output
