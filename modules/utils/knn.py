import torch
import torch.nn as nn
import torch.nn.functional as F

def _knn(similarity, target, topk=1):
    correct = (target.expand_as(similarity) == target.expand_as(similarity).t())
    _, pred_label = similarity.topk(topk, dim=1)
    result = 0
    for i in range(pred_label.size(0)):
        correct_i = correct[i][pred_label[i]].sum() / float(topk)
        result += correct_i.item()

    return result


def KNN(pred:torch.Tensor, target, topk=500, l2norm=True):
    with torch.no_grad():
        if len(pred.size()) != 2:
            pred = pred.mean(axis=[2, 3])
        assert len(pred.size()) == 2
        assert len(target.size()) == 1

        if l2norm:
            pred = F.normalize(pred, dim=1)

        out = torch.matmul(pred, pred.t())
        return _knn(out, target, topk)