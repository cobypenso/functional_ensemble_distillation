"""Resnet-blocks for cifar10, from https://github.com/kuangliu/pytorch-cifar"""

import torch.nn as nn
import torch.nn.functional as F

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_type = 'original', norm_type = 'bn'):
        super(BasicBlock, self).__init__()
        
        self.conv_layer = Conv2d_WN if conv_type == 'ws' else nn.Conv2d
        self.norm_type = norm_type
        
        self.conv1 = self.conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes, self.norm_type)
        self.conv2 = self.conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes, self.norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                self.conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes, self.norm_type)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, conv_type = 'original', norm_type = 'bn'):
        super(Bottleneck, self).__init__()
        
        self.conv_layer = Conv2d_WN if conv_type == 'ws' else nn.Conv2d
        self.norm_type = norm_type
        
        self.conv1 = self.conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, self.norm_type)
        self.conv2 = self.conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, self.norm_type)
        self.conv3 = self.conv_layer(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(self.expansion*planes, self.norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                self.conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes, self.norm_type)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
