import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_norm_layer, constant_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcv.runner import load_checkpoint

class ReluNeck(nn.Module):

    def __init__(self, in_channels, frozen_state=False, norm_cfg=None, avgpool=False):
        super(ReluNeck, self).__init__()
        self.norm_cfg = norm_cfg
        self.frozen_state = frozen_state
        if self.norm_cfg is not None:
            self.norm = build_norm_layer(self.norm_cfg, in_channels)[1]
        else:
            self.norm = None

        self.relu = nn.ReLU(inplace=False)
        if avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = None
            
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_state and self.norm is not None:
            self.norm.eval()
            for param in self.norm.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)

    def train(self, mode=True):
        super(ReluNeck, self).train(mode)
        self._freeze_stages()

    def forward(self, x):
        if self.norm:
            x = self.norm(x)
        
        outs = self.relu(x)
        if self.avgpool is not None:
            outs = self.avgpool(outs).view(outs.size(0), -1)
        return outs

