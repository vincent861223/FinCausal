import torch.nn.functional as F
import torch.nn as nn
import torch

def crossEntropy_loss(output, target):
    weight = torch.tensor([0.2, 0.8]).to(output.device)
    return nn.CrossEntropyLoss(weight=weight)(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    return nn.BCELoss()(output, target.float())

def f1_loss(output, target):
    return F1_Loss()(output, target)

class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        target2 = target.unsqueeze(1)
            #print(target2)
        target = torch.zeros(target.size(0), 2).cuda().scatter_(1, target2, 1)

        softmax =  nn.Softmax()

        probas = softmax(output)
        #probas = nn.Sigmoid()(output)
        target = target.type(torch.cuda.FloatTensor)
        TP = (probas * target).sum(dim=1)
        precision = TP / (probas.sum(dim=1) + self.epsilon)
        recall = TP / (target.sum(dim=1) + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

