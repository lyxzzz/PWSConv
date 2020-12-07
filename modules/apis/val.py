import torch
import torch.nn as nn
import time
import mmcv
from ..utils import KNN

def evaluate_cls(model, device, data_loader, datasize, epoch=0, logger=None):
    model.eval()
    correct = None
    total = 0
    prog_bar = mmcv.ProgressBar(datasize)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, targets)

            total += targets.size(0)
            if correct is None:
                correct = {}
                for k in outputs['accuracy']:
                    correct[k] = outputs['accuracy'][k].item()
            else:
                for k in outputs['accuracy']:
                    correct[k] += outputs['accuracy'][k].item()

            for _ in range(targets.size(0)):
                prog_bar.update()

    for k in correct:
        correct[k] = round(correct[k] / float(total) * 100.0, 4)
    
    logger.record_eval(epoch, correct)

def evaluate_knn(model, device, data_loader, datasize, config, epoch=0, logger=None):
    model.eval()
    total = 0

    knn_config = config.get('knn', {})
    if "topk_percent" not in knn_config:
        topk = int(datasize / config['num_class'] * 0.2)
    else:
        topk = int(datasize / config['num_class'] * knn_config.pop('topk_percent'))
    knn_config['topk'] = topk

    prog_bar = mmcv.ProgressBar(datasize)

    total_outputs = []
    total_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model.forward_knn(inputs)
            total_outputs.append(outputs)
            total_targets.append(targets)

            for _ in range(targets.size(0)):
                prog_bar.update()

    print('==> Calculating KNN..')
    total_outputs = torch.cat(total_outputs, dim=0)
    total_targets = torch.cat(total_targets, dim=0)
    total_acc = KNN(total_outputs, total_targets, **knn_config) / datasize
    correct = {
        f"KNN-{topk}":total_acc * 100.0
    }
    logger.record_eval(epoch, correct)