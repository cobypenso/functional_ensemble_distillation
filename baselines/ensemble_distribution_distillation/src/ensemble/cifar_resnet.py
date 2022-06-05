import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as torch_optim
from src.ensemble import ensemble
from src.experiments.cifar10 import resnet_utils


class ResNet(ensemble.EnsembleMember):
    """ResNet
    Resnet network adapted to Cifar10, adapted from https://github.com/kuangliu/pytorch-cifar

    Args:
        block (resnet_utils.BasicBlock/resnet_utils.Bottleneck)
        num_blocks (vector(int))
        device (torch.Device)
        learning_rate (float)
        num_classes (int)
    """

    def __init__(self, block, num_blocks, device=torch.device('cpu'), learning_rate=0.001, num_classes=10):
        super().__init__(output_size=num_classes, loss_function=nn.CrossEntropyLoss(), device=device)
        self.learning_rate = learning_rate

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.optimizer = torch_optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # Added functions
    def transform_logits(self, logits):
        return logits

    def calculate_loss(self, outputs, labels):
        return self.loss(outputs, labels.long())

    def predict(self, x, t=1):
        x = self.forward(x)
        x = (nn.Softmax(dim=-1))(x)

        return x

    def _learning_rate_condition(self, epoch):
        step_epochs = [80, 120, 160, 180]
        if epoch in step_epochs:
            return True
        else:
            return False

    def eval_mode(self, train=False):
        # Setting layers to eval mode

        if train:
            # Not sure this is the way to go
            self.conv1.train()
            self.bn1.train()
            self.layer1.train()
            self.layer2.train()
            self.layer3.train()
            self.layer4.train()
            self.linear.train()
        else:
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            self.linear.eval()


def ResNet18():
    return ResNet(resnet_utils.BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(resnet_utils.BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(resnet_utils.Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(resnet_utils.Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(resnet_utils.Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())