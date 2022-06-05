import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

'''
All Functions and Modules in this file, responsible for calibration analysis
and temperature scaling (Method for better calibration)

Based on https://github.com/gpleiss/temperature_scaling/ - temperature scaling repo.
(https://arxiv.org/abs/1706.04599)
'''

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, plot = False, softmax = False, ax = None):
        if softmax:
            softmaxes = logits
        else:
            softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)

        confidence_in_bins = []
        accuracy_in_bins = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                confidence_in_bins.append(avg_confidence_in_bin)
                accuracy_in_bins.append(accuracy_in_bin)
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            else:
                confidence_in_bins.append(0)
                accuracy_in_bins.append(0)
        if plot:
            x = (self.bin_lowers + self.bin_uppers) / 2
            y = accuracy_in_bins
            ax.bar(x=x, height=x, width=self.bin_uppers[0], align='center', color='r', edgecolor='black', linewidth=1)
            ax.bar(x=x,height=y, width=self.bin_uppers[0], align='center', edgecolor='black', linewidth=1)
            ax.plot([0,1], [0,1], linestyle = "--", color="gray")
            ax.set_xlabel('Condifence')
            ax.set_ylabel('Accuracy')
            return ece, ax

        return ece

def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    if temperature.shape[0] == logits.shape[0]:
        temperature = temperature[...,None,None].expand(logits.size(0), logits.size(1), logits.size(2))
    elif len(logits.size()) == 2:
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    elif len(logits.size())== 3:
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1), logits.size(2))
    return logits / temperature

def set_temperature(labels, logits, temperature, device):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader

    Note: Optimization over the temperature parameter is done using SGD
    """
    #logits = logits.detach()
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = _ECELoss().to(device)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits.view(-1,10), labels.view(-1)).item()
    before_temperature_ece = ece_criterion(logits.view(-1,10), labels.view(-1)).item()
    print('Before temperature - NLL: %.8f, ECE: %.8f' % (before_temperature_nll, before_temperature_ece))
    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.SGD([temperature], lr=0.0001, momentum=0.9)

    for i in range(100):
        loss = nll_criterion(temperature_scale(logits.view(-1,10), temperature), labels.view(-1))
        loss.backward()
        optimizer.step()

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(temperature_scale(logits.view(-1,10), temperature), labels.view(-1)).item()
    after_temperature_ece = ece_criterion(temperature_scale(logits.view(-1,10), temperature), labels.view(-1)).item()
    print('Optimal temperature: %.8f' % temperature.item())
    print('After temperature - NLL: %.8f, ECE: %.8f' % (after_temperature_nll, after_temperature_ece))
    return temperature

def plot_calibration(labels, logits, temperature, filename="calibration"):
    '''
    Plot calibration(Reliability diagram) given true labels and logits.

    @param: labels
    @param: logits
    @param: temperature
    @param: filename - filename (withoud .pdf postfix) for which the plot will be saved to.

    2 Options of usage:
        1. If temperature is given, then plotting is
           done before & after temperature scaling
        2. If temeperature is None then plot calibration without tmp scale.
    '''
    ece_criterion = _ECELoss()
    if temperature is None:
        _,plt = ece_criterion(logits.view(-1,10), labels.view(-1), plot = True)
        plt.savefig(filename+".pdf")
    else:
        _,plt = ece_criterion(logits.view(-1,10), labels.view(-1), plot = True)
        plt.savefig(filename+"_before_tmp.pdf")
        _,plt = ece_criterion(temperature_scale(logits.view(-1,10), temperature), labels.view(-1), plot = True)
        plt.savefig(filename+"_after_tmp.pdf")
