import torch
import torch.nn as nn
import torch.optim as torch_optim
import torch.nn.functional as F

import src.loss as custom_loss
from src.distilled import distilled_network
from src import metrics

def norm_layer(channels, type, **kwargs):
    if type == 'bn':
        return nn.BatchNorm2d(channels)
    elif type == 'gn':
        num_groups=32 # replace with kwargs['groups']
        return nn.GroupNorm(num_groups, channels)
    else:
        raise Exception('Unsupported norm layer type')

class Conv2d_WN(nn.Conv2d):
    # https://github.com/joe-siyuan-qiao/pytorch-classification.git
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WN, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
                        
                        
class CifarResnetDirichlet(distilled_network.DistilledNet):
    """CifarResnetDirichlet
    Network that predicts parameters for Dirichlet distribution over teacher output
    Args:
        teacher (Ensemble)
        block (resnet_utils.BasicBlock/resnet_utils.Bottleneck)
        num_blocks (vector(int))
        device (torch.Device)
        learning_rate (float)
        temp (float)
    """
    def __init__(self,
                 teacher,
                 block,
                 num_blocks,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 temp=10,
                 output_size = 10,
                 conv_type = 'original',
                 norm_type = 'bn'):

        super().__init__(teacher=teacher,
                         loss_function=custom_loss.dirichlet_nll,
                         device=device)

        self.learning_rate = learning_rate
        self.temp = temp

        self.in_planes = 64
        self.conv_layer = Conv2d_WN if conv_type=='ws' else nn.Conv2d
        self.norm_type = norm_type
        
        self.conv1 = self.conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64, self.norm_type)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, conv_type = conv_type)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, conv_type = conv_type)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, conv_type = conv_type)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, conv_type = conv_type)

        self.output_size = output_size
        self.linear = nn.Linear(512 * block.expansion, self.output_size)

        self.optimizer = torch_optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride, conv_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv_type, self.norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Estimate parameters of distribution
        """
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0].float()
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        out = torch.exp(self.linear(out) / self.temp)

        return out

    def _generate_teacher_predictions(self, inputs, gamma=1e-4):
        """Generate teacher predictions"""

        logits = self.teacher.get_logits(inputs)
        predictions = torch.exp(logits / self.temp) / torch.sum(torch.exp(logits / self.temp), dim=-1, keepdim=True)

        # "Central smoothing"
        predictions = (1 - gamma) * predictions + gamma * (1 / self.output_size)

        return predictions

    def predict(self, input_, num_samples=None, return_params=False):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if isinstance(input_, list) or isinstance(input_, tuple):
            input_ = input_[0]

        if num_samples is None:
            num_samples = 100

        alphas = self.forward(input_.float())

        samples = torch.zeros(
            [input_.size(0), num_samples, self.output_size])

        for i in range(input_.size(0)):
            
            rv = torch.distributions.dirichlet.Dirichlet(concentration=alphas[i, :], validate_args = False)
            samples[i, :, :] = rv.rsample([num_samples])

        if return_params:
            return alphas, samples

        else:
            return samples

    def _learning_rate_condition(self, epoch):
        if epoch%20 == 0:
            return True
        else:
            return False
            
    def _temperature_anneling(self, temp_factor=0.95):

        if self.temp > 1:
            self.temp = temp_factor * self.temp

        if self.temp < 1:
            self.temp = 1

    def _temp_annealing_schedule(self, epoch=None):
        if epoch >= 50:
            return True
        else:
            return False

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs, teacher_predictions)

    def calculate_acc(self, outputs, teacher_predictions, labels=None):
        if labels is None:
            return 
        return (torch.argmax(torch.mean(outputs), dim = -1) == labels).sum()


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

            if temp is None:
                self.temp = 10
            else:
                self.temp = temp

        else:
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            self.linear.eval()
            self.temp = 1

    def eval(self, loader):
        self.eval_mode()
        counter = 0
        model_acc = 0

        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs[0].to(self.device), labels.to(self.device)

            predicted_distribution = self.predict(inputs.float()).mean(axis=1)
            model_acc += metrics.accuracy(predicted_distribution.to(self.device), labels.long())
            counter += 1

        self.eval_mode(train=True)
        return (model_acc / counter)
        