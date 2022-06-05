import torch
import torch.nn as nn
from models import *

'''
This file contains generators which will be used to learn the
predictive distribution
'''

def get_generator(args):
    '''
        Use cifar10_cat_gn_with_noise, cifar100_cat_gn_with_noise, stl10_cat_gn_with_noise
    '''

    if args.gen == 'cifar10_cat_gn_with_noise' or args.gen == 'stl10_cat_gn_with_noise':
        generator = ConditionalResNet18_cat(add_noise = True, 
                                            noise_std = args.noise_std, 
                                            learnable_noise = args.learn_noise, 
                                            norm_type ='gn', 
                                            ws = True)
    elif args.gen == 'cifar100_cat_gn_with_noise':
        generator = ConditionalResNet18_cat(add_noise = True, 
                                            noise_std = args.noise_std, 
                                            learnable_noise = args.learn_noise, 
                                            norm_type ='gn', 
                                            ws = True,
                                            num_classes = 100)
    elif args.gen == 'cifar100_cat_gn_with_noise_resnet34':
        generator = ConditionalResNet34_cat(add_noise = True, 
                                            noise_std = args.noise_std, 
                                            learnable_noise = args.learn_noise, 
                                            norm_type ='gn', 
                                            ws = True,
                                            num_classes = 100)

    if args.gen_ckpt:
        generator.load_state_dict(torch.load(args.gen_ckpt, map_location=torch.device('cpu')))
        
    return generator


################################################################################################################################

class ConditionalResNet18_cat(nn.Module):
    def __init__(self, add_noise = False, noise_std = 0.1, modes = 0, norm_type ='bn', ws = False, learnable_noise = False, num_classes = 10):
        super().__init__()
        self.model = ResNet18(in_channels=6, 
                              add_noise = add_noise, 
                              noise_std = noise_std, 
                              norm_type = norm_type, 
                              ws = ws, 
                              learnable_noise = learnable_noise,
                              num_classes = num_classes)
        self.learnable_noise = learnable_noise 

        if modes >= 1:
            self.stds = nn.Parameter(torch.Tensor((modes)))
            self.means = nn.Parameter(torch.arange(-1, 1, 2 / modes))
            nn.init.ones_(self.stds)
        
        if self.learnable_noise:
            self.latent_std_factor = nn.Parameter(torch.ones(1))
        else:
            self.latent_std_factor = torch.ones(1)
        
        if torch.cuda.is_available():
            self.cuda()
            self.latent_std_factor = self.latent_std_factor.cuda()
            
    def forward(self, z, condition_var):
        # -- z [B:3:32:32] -- #
        # -- condition_var [B:3:32:32] -- #
        z = self.latent_std_factor * z
        x = torch.cat([condition_var,z], dim=1)
        # -- x [B:6,32,32]
        out = self.model(x)
        return out

################################################################################################################################

class ConditionalResNet34_cat(nn.Module):
    def __init__(self, add_noise = False, noise_std = 0.1, modes = 0, norm_type ='bn', ws = False, learnable_noise = False, num_classes = 10):
        super().__init__()
        self.model = ResNet34(in_channels=6, 
                              add_noise = add_noise, 
                              noise_std = noise_std, 
                              norm_type = norm_type, 
                              ws = ws, 
                              learnable_noise = learnable_noise,
                              num_classes = num_classes)
        self.learnable_noise = learnable_noise 

        if modes >= 1:
            self.stds = nn.Parameter(torch.Tensor((modes)))
            self.means = nn.Parameter(torch.arange(-1, 1, 2 / modes))
            nn.init.ones_(self.stds)
        
        if self.learnable_noise:
            self.latent_std_factor = nn.Parameter(torch.ones(1))
        else:
            self.latent_std_factor = torch.ones(1)
        
        if torch.cuda.is_available():
            self.cuda()
            self.latent_std_factor = self.latent_std_factor.cuda()
            
    def forward(self, z, condition_var):
        # -- z [B:3:32:32] -- #
        # -- condition_var [B:3:32:32] -- #
        z = self.latent_std_factor * z
        x = torch.cat([condition_var,z], dim=1)
        # -- x [B:6,32,32]
        out = self.model(x)
        return out
