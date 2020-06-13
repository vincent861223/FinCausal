import torch
from sklearn.metrics import precision_recall_fscore_support


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.max(output, dim=-1)[1] 
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def f1_score(output, target):
    with torch.no_grad():
        pred = torch.max(output, dim=-1)[1] 
        precision, recall, f1, _ = precision_recall_fscore_support(target.cpu(), pred.cpu(), labels=[0, 1], average='weighted')
    return f1



def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
