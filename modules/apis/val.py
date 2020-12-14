import torch
import torch.nn as nn
import numpy as np
import time
import mmcv
from ..utils import KNN

class evaluate_cls(object):
    def __init__(self, model, device, data_loader, datasize, logger=None):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.logger = logger

    def __call__(self, epoch=0):
        self.model.eval()

        correct = None
        total = 0
        prog_bar = mmcv.ProgressBar(self.datasize)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs, targets)

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
        
        self.logger.record_eval(epoch, correct)

class evaluate_knn(object):
    def __init__(self, model, device, data_loader, datasize, config, logger):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger
        self.outputs_container = None
        self.targets_container = np.zeros([1, datasize], dtype=np.int64)
        
        self.knn_config = config.get('knn', {})
        if "topk" not in self.knn_config:
            if "topk_percent" not in self.knn_config:
                self.topk = int(datasize / config['num_class'] * 0.2)
            else:
                self.topk = int(datasize / config['num_class'] * self.knn_config.pop('topk_percent'))
            self.knn_config['topk'] = topk
        else:
            self.topk = self.knn_config['topk']
    
    def __call__(self, epoch=0):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(self.datasize)

        sample_idx = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                batchsize = targets.size(0)

                outputs = self.model.forward_knn(inputs)
                    
                if self.outputs_container is None:
                    self.outputs_container = np.zeros([self.datasize, outputs.size(1)], dtype=np.float32)
                
                self.outputs_container[sample_idx:sample_idx+batchsize] = outputs.cpu().numpy()
                self.targets_container[0, sample_idx:sample_idx+batchsize] = targets.cpu().numpy()

                sample_idx += batchsize
                for _ in range(batchsize):
                    prog_bar.update()

        print('==> Calculating KNN..')
        total_acc = KNN(self.outputs_container, self.targets_container, **self.knn_config)
        correct = {
            f"KNN-{self.topk}":total_acc * 100.0
        }
        self.logger.record_eval(epoch, correct)