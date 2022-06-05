'''
This file includs different models
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import norm
from utils import nonlin, initialize_tensor



def get_net(arch, batch_ensemble = False, concentration = False, **kwargs):
    '''
    Get Model
    @param arch - [str] - arch of the model
    @param concentration - For Dirichlet baseline support - https://arxiv.org/abs/1905.12194
    @param batch_ensemble - For BatchEnsemble model support - https://arxiv.org/abs/2110.14149
    @param kwargs - the rest, passed through to Model constructor.
    '''
    print('==> Building model..')
    
    # Batch norm based resnet
    if arch == 'resnet18_bn_cifar10':
        net = ResNet18(norm_type='bn', num_classes = 10, batch_ensemble = batch_ensemble, **kwargs)
    elif arch == 'resnet18_bn_stl10':
        net = ResNet18(norm_type='bn', num_classes = 10, batch_ensemble = batch_ensemble, **kwargs)
    elif arch == 'resnet18_bn_cifar100':
        net = ResNet18(norm_type='bn', num_classes = 100, batch_ensemble = batch_ensemble, **kwargs)

    # Group norm based resnet
    elif arch == 'resnet18_gn':
        net = ResNet18(norm_type='gn', batch_ensemble = batch_ensemble, **kwargs)
    elif arch == 'resnet18_gn_ws_cifar10' or arch == 'resnet18_gn_ws_stl10':
        net = ResNet18(norm_type='gn', ws = True, num_classes = 10, batch_ensemble = batch_ensemble, concentration = concentration, **kwargs)
    elif arch == 'resnet18_gn_ws_cifar100':
        net = ResNet18(norm_type='gn', ws = True, num_classes = 100, batch_ensemble = batch_ensemble, concentration = concentration, **kwargs)

    else:
        raise Exception(arch + " architecture is not supported")

    return net
    

###################################################
#################       ResNet      ###############
###################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type = 'bn', ws = False, batch_ensemble=False, **kwargs):
        super(BasicBlock, self).__init__()
        conv_layer = Conv2d_WN if ws else nn.Conv2d
        linear_layer = nn.Linear
        if batch_ensemble:
            conv_layer = Conv2d_BatchEnsemble
            linear_layer = Linear_BatchEnsemble

        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn1 = norm_layer(planes, norm_type)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = norm_layer(planes, norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, **kwargs),
                norm_layer(self.expansion*planes, norm_type)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type = 'bn', ws = False, batch_ensemble = False, **kwargs):
        super(Bottleneck, self).__init__()
        conv_layer = Conv2d_WN if ws else nn.Conv2d
        linear_layer = nn.Linear
        if batch_ensemble:
            conv_layer = Conv2d_BatchEnsemble
            linear_layer = Linear_BatchEnsemble

        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False, **kwargs)
        self.bn1 = norm_layer(planes, norm_type) 
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn2 = norm_layer(planes, norm_type)
        self.conv3 = conv_layer(planes, self.expansion*planes, kernel_size=1, bias=False, **kwargs)
        self.bn3 = norm_layer(self.expansion*planes, norm_type) 

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, **kwargs),
                norm_layer(self.expansion*planes, norm_type) 
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels = 3, 
                 num_classes=10, add_noise = False, noise_std = 0.1, 
                 norm_type = 'bn', ws = False, learnable_noise = False,
                 batch_ensemble = False, concentration = False, **kwargs):
        super(ResNet, self).__init__()
        conv_layer = Conv2d_WN if ws else nn.Conv2d
        linear_layer = nn.Linear
        if batch_ensemble:
            linear_layer = Linear_BatchEnsemble
            conv_layer = Conv2d_BatchEnsemble
        
        self.concentration = concentration
        self.in_planes = 64
        self.in_channels = in_channels
        self.conv1 = conv_layer(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn1 = norm_layer(64, norm_type)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_type=norm_type, ws = ws, batch_ensemble = batch_ensemble, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_type=norm_type, ws = ws, batch_ensemble = batch_ensemble, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_type=norm_type, ws = ws, batch_ensemble = batch_ensemble, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_type=norm_type, ws = ws, batch_ensemble = batch_ensemble, **kwargs)
        if concentration:
            self.linear1 = linear_layer(512*block.expansion, 64, **kwargs)
            self.linear2 = linear_layer(64, 1, **kwargs)
        else:
            self.linear = linear_layer(512*block.expansion, num_classes, **kwargs)
        self.noise = add_noise
        self.noise_std = noise_std
        self.learnable_noise = learnable_noise
        if self.learnable_noise:
            self.noise_factor = nn.Parameter(torch.ones((5)))
        else:
            self.noise_factor = torch.ones(5)

    def _make_layer(self, block, planes, num_blocks, stride, norm_type='bn', ws = False, batch_ensemble = False, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_type, ws, batch_ensemble, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        # noise insertion #
        if self.noise:
            out = out + self.noise_factor[0] * torch.normal(mean = torch.zeros_like(out), 
                                   std = self.noise_std * torch.ones_like(out))
        out = self.layer1(out)
        # noise insertion #
        if self.noise:
            out = out + self.noise_factor[1] * torch.normal(mean = torch.zeros_like(out), 
                                   std = self.noise_std * torch.ones_like(out))
        out = self.layer2(out)
        # noise insertion #
        if self.noise:
            out = out + self.noise_factor[2] * torch.normal(mean = torch.zeros_like(out), 
                                   std = self.noise_std * torch.ones_like(out))
        out = self.layer3(out)
        # noise insertion #
        if self.noise:
            out = out + self.noise_factor[3] *  torch.normal(mean = torch.zeros_like(out), 
                                   std = self.noise_std * torch.ones_like(out))
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        # noise insertion #
        if self.noise:
            out = out + self.noise_factor[4] *  torch.normal(mean = torch.zeros_like(out), 
                                   std = self.noise_std * torch.ones_like(out))
        
        if self.concentration: # In order to support AMT baseline -  self.concentration=False by default
            out = self.linear2(self.linear1(out))
        else:
            out = self.linear(out)
        
        return out

def ResNet18(in_channels = 3, num_classes=10, add_noise = False, noise_std = 0.1, 
             norm_type = 'bn', ws = False, learnable_noise = False, batch_ensemble = False, concentration = False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, 
                  in_channels=in_channels, add_noise = add_noise, 
                  noise_std = noise_std, norm_type = norm_type, 
                  ws = ws, learnable_noise = learnable_noise,
                  batch_ensemble = batch_ensemble, concentration = concentration, **kwargs)

def ResNet34(in_channels = 3, num_classes=10, add_noise = False, noise_std = 0.1, 
             norm_type = 'bn', ws = False, learnable_noise = False, batch_ensemble = False, concentration = False, **kwargs):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, 
                  in_channels=in_channels, add_noise = add_noise, 
                  noise_std = noise_std, norm_type = norm_type, 
                  ws = ws, learnable_noise = learnable_noise,
                  batch_ensemble = batch_ensemble, concentration = concentration, **kwargs)


###################################################
################### Building Blocks ###############
###################################################

class Conv2d_BatchEnsemble(nn.Conv2d):
    # https://github.com/cs-giung/giung2/tree/main/projects/Diversity-Matters
    def __init__(self, *args, **kwargs) -> None:
        self.same_padding =  True
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Conv2d_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_channels)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_channels)
            )
        )
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_channels)
                )
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        initialize_tensor(self.alpha_be, self.alpha_initializer[0], self.alpha_initializer[1])
        initialize_tensor(self.gamma_be, self.gamma_initializer[0], self.gamma_initializer[1])
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, C1, H1, W1 = x.size()
        r_x = x.view(self.ensemble_size, -1, C1, H1, W1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, C1, 1, 1)
        r_x = r_x.view(-1, C1, H1, W1)

        # if self.same_padding:
        #     r_x = self._pad_input(r_x)
        # w_r_x = self._conv_forward(r_x, self.weight, self.bias)
        w_r_x = super().forward(r_x)

        _, C2, H2, W2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, C2, H2, W2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, C2, 1, 1)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, C2, 1, 1)
        s_w_r_x = s_w_r_x.view(-1, C2, H2, W2)

        return s_w_r_x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, ensemble_size={ensemble_size}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            if self.ensemble_bias is None:
                s += ', bias=False, ensemble_bias=False'
            else:
                s += ', bias=False, ensemble_bias=True'
        else:
            if self.ensemble_bias is None:
                s += ', bias=True, ensemble_bias=False'
            else:
                s += ', bias=True, ensemble_bias=True'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Conv2d_BatchEnsembleV2(Conv2d_BatchEnsemble):
    # https://github.com/cs-giung/giung2/tree/main/projects/Diversity-Matters
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, C1, H1, W1 = x.size()
        r_x = x.view(self.ensemble_size, -1, C1, H1, W1)
        r_x = r_x * (1.0 + self.alpha_be.view(self.ensemble_size, 1, C1, 1, 1))
        r_x = r_x.view(-1, C1, H1, W1)

        # if self.same_padding:
        #     r_x = self._pad_input(r_x)
        # w_r_x = self._conv_forward(r_x, self.weight, self.bias)
        w_r_x = super().forward(r_x)

        _, C2, H2, W2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, C2, H2, W2)
        s_w_r_x = s_w_r_x * (1.0 + self.gamma_be.view(self.ensemble_size, 1, C2, 1, 1))
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, C2, 1, 1)
        s_w_r_x = s_w_r_x.view(-1, C2, H2, W2)

        return s_w_r_x


class Linear_BatchEnsemble(nn.Linear):
    # https://github.com/cs-giung/giung2/tree/main/projects/Diversity-Matters
    def __init__(self, *args, **kwargs) -> None:
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Linear_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_features)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_features)
            )
        )
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_features)
                )
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        initialize_tensor(self.alpha_be, self.alpha_initializer[0], self.alpha_initializer[1])
        initialize_tensor(self.gamma_be, self.gamma_initializer[0], self.gamma_initializer[1])
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, D1 = x.size()
        r_x = x.view(self.ensemble_size, -1, D1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ensemble_size={}, ensemble_bias={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.ensemble_size, self.ensemble_bias is not None
        )


class Linear_BatchEnsembleV2(Linear_BatchEnsemble):
    # https://github.com/cs-giung/giung2/tree/main/projects/Diversity-Matters
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, D1 = x.size()
        r_x = x.view(self.ensemble_size, -1, D1)
        r_x = r_x * (1.0 + self.alpha_be.view(self.ensemble_size, 1, D1))
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * (1.0 + self.gamma_be.view(self.ensemble_size, 1, D2))
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x


def norm_layer(channels, type, **kwargs):
    if type == 'bn':
        return nn.BatchNorm2d(channels)
    elif type == 'gn':
        num_groups=32 # replace with kwargs['groups']
        return nn.GroupNorm(num_groups, channels)
    elif type == 'frn':
        return FilterResponseNorm2d(channels)
    else:
        raise Exception('Unsupported norm layer type')


class Conv2d_WN(nn.Conv2d):
    '''
        Conv2d with Weight Standardization
    '''
    # https://github.com/joe-siyuan-qiao/pytorch-classification.git
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
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
