import torch.nn.functional as F
import torch.nn as nn

def mse_loss(output, target):
    return nn.MSELoss()(output, target.float())

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    return nn.BCELoss()(output, target.float())
