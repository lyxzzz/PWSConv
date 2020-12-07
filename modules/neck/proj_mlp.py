import torch
import torch.nn as nn
import torchvision
from mmcv.cnn import (normal_init, constant_init, kaiming_init)

class Proj_MLP(nn.Module):
    def __init__(self, hidden_layer=3, in_channels=2048, hidden_channels=2048, out_channels=2048):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        layer_list = [
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)]

        if hidden_layer != 2:
            layer_list += [
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True)]
        
        layer_list += [
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels)]

        self.layer = nn.Sequential(*layer_list)
    
    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming']
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x 