import torch
import torch.nn as nn

class MMD(nn.Module):
    def __init__(self, bandwidth_range = [2.0], kernel = 'multiscale', device='cuda', return_matrix = False):
        super(MMD, self).__init__()    
        self.bandwidth_range = bandwidth_range
        self.device = device
        self.kernel = kernel
        self.return_matrix = return_matrix

    def forward(self, x, y):
        """
        Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        
        XX, YY, XY = (torch.zeros(xx.shape, device = self.device),
                    torch.zeros(xx.shape, device = self.device),
                    torch.zeros(xx.shape, device = self.device))
                    
        if self.kernel == "multiscale":
            
            bandwidth_range = self.bandwidth_range
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        if self.kernel == "rbf":
        
            bandwidth_range = self.bandwidth_range
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)
        
        if self.kernel == 'linear':
            XX += xx
            YY += yy
            XY += zz
            
        if self.return_matrix:
            return torch.mean(XX + YY - 2. * XY), XX, XY, YY
            
        return torch.mean(XX + YY - 2. * XY)