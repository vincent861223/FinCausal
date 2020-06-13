import torch
from sklearn.metrics import precision_recall_fscore_support


def accuracy(score, position):
    with torch.no_grad():
        correct = 0
        for key in score.keys():
            pred = torch.max(score[key], dim=-1)[1]
            correct += torch.sum(pred == position[key]).item()
    return correct / (len(position['cause_start']) * 4)

def precision():
    with torch.no_grad():
        pred = output.round()


def f1_score(output, target):
    with torch.no_grad():
        pred = output.round()
        precision, recall, f1, _ = precision_recall_fscore_support(target.cpu(), pred.cpu(),  average='weighted')
    return f1



def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
