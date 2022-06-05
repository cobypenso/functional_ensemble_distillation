import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
import torch.nn.functional as F


class CifarResnetLogits(distilled_network.DistilledNet):
    """CifarResnetDirichlet
    Network that predicts parameters for Gaussian distribution over teacher output logits,
    or (if mixture_distillation=True) predicts the mean of the teacher output
    Args:
        teacher (Ensemble)
        block (resnet_utils.BasicBlock/resnet_utils.Bottleneck)
        num_blocks (vector(int))
        device (torch.Device)
        learning_rate (float)
        mixture_distillation (boolean)
        temp (float)
    """
    def __init__(self,
                 teacher,
                 block,
                 num_blocks,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 mixture_distillation=False,
                 temp=2.5):

        if mixture_distillation:
            loss_fun = custom_loss.cross_entropy_soft_targets
        else:
            loss_fun = custom_loss.gaussian_neg_log_likelihood

        super().__init__(teacher=teacher,
                         loss_function=loss_fun,
                         device=device)

        self.learning_rate = learning_rate
        self.mixture_distillation = mixture_distillation

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if self.mixture_distillation:
            self.output_size = 10
            self.temp = temp
        else:
            self.output_size = 27

        self.linear = nn.Linear(512 * block.expansion, self.output_size)

        # Ad-hoc fix zero variance.
        self.variance_lower_bound = 0.0
        if self.variance_lower_bound > 0.0:
            self._log.warning("Non-zero variance lower bound set ({})".format(
                self.variance_lower_bound))

        self.optimizer = torch_optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

        self.to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, softmax_transform=True, return_raw=False, comp_fix=False):
        """Estimate parameters of distribution
        """

        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.mixture_distillation:
            if softmax_transform:
                out = torch.exp(out / self.temp) / torch.sum(torch.exp(out / self.temp), dim=-1, keepdim=True)
            return out

        else:
            split = int(out.shape[-1] / 3)
            mean = out[:, :split]
            var_z = out[:, split:(2 * split)]
            const = nn.ReLU()(out[:, (2 * split):])

            if comp_fix:
                var = torch.zeros(var_z.size()).to(self.device)
                var[var_z > 10] = var_z[var_z > 10] + const[var_z > 10] + self.variance_lower_bound
                var[var_z <= 10] = torch.log(1 + torch.exp(var_z[var_z <= 10])) + const[
                    var_z <= 10] + self.variance_lower_bound
            else:
                var = torch.log(1 + torch.exp(var_z)) + const + self.variance_lower_bound

            if return_raw:
                return mean, var, out
            else:
                return mean, var

    def _generate_teacher_predictions(self, inputs):
        """Generate teacher predictions"""

        if self.mixture_distillation:
            logits = self.teacher.get_logits(inputs)
            predictions = torch.exp(logits / self.temp) / torch.sum(torch.exp(logits / self.temp), dim=-1, keepdim=True)
            mean_predictions = torch.mean(predictions, dim=1)

            return mean_predictions

        else:

            logits = self.teacher.get_logits(inputs)
            scaled_logits = logits - torch.stack([logits[:, :, -1]], axis=-1)

            return scaled_logits[:, :, :-1]

    def predict(self, input_, num_samples=None, return_raw_data=False, return_logits=False, comp_fix=False):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if self.mixture_distillation:
            return self.forward(input_)

        else:

            samples = self.predict_logits(input_, num_samples, return_raw_data, comp_fix)

            output = []
            if return_raw_data:
                samples, mean, var, raw_output = samples
                output.append(mean)
                output.append(var)
                output.append(raw_output)

            if return_logits:
                output.append(samples)

            if isinstance(samples, list):
                output.append(nn.Softmax(dim=-1)(samples[0]))
            else:
                output.append(nn.Softmax(dim=-1)(samples))

            return output

    def predict_logits(self, input_, num_samples=None, return_raw_data=False, comp_fix=False):

        if isinstance(input_, list) or isinstance(input_, tuple):
            input_ = input_[0]

        if self.mixture_distillation:
            return self.forward(input_, softmax_transform=False)

        else:
            if num_samples is None:
                num_samples = 100

            if return_raw_data:
                mean, var, raw_output = self.forward(input_, return_raw=True, comp_fix=comp_fix)
            else:
                mean, var = self.forward(input_, comp_fix=comp_fix)

            samples = torch.zeros(
                [input_.size(0), num_samples,
                 int(self.output_size / 3)])

            for i in range(input_.size(0)):
                rv = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=mean[i, :], covariance_matrix=torch.diag(var[i, :]))
                samples[i, :, :] = rv.rsample([num_samples])

            samples = torch.cat((samples, torch.zeros(samples.size(0), num_samples, 1)), dim=-1)

            output = [samples]
            if return_raw_data:
                output.append(mean)
                output.append(var)
                output.append(raw_output)

            return output

    def _learning_rate_condition(self, epoch):
        if epoch%20 == 0:
            return True
        else:
            return False

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        return self.loss(outputs, teacher_predictions)

    def eval_mode(self, train=False, temp=None):
        # Setting layers to eval mode

        if train:
            self.conv1.train()
            self.bn1.train()
            self.layer1.train()
            self.layer2.train()
            self.layer3.train()
            self.layer4.train()
            self.linear.train()

            if self.mixture_distillation:
                if temp is not None:
                    self.temp = temp
                else:
                    self.temp = 2.5

        else:
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            self.linear.eval()

            if self.mixture_distillation:
                self.temp = 1




