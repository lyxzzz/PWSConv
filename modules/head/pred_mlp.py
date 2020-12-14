import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn import (normal_init, constant_init, kaiming_init)

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()
    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class Pred_MLP(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048): # bottleneck structure
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=False),

            nn.Linear(hidden_channels, out_channels)
        )

    # def init_weights(self, init_linear='normal', std=0.01, bias=0.):
    #     assert init_linear in ['normal', 'kaiming']
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             if init_linear == 'normal':
    #                 normal_init(m, std=std, bias=bias)
    #             else:
    #                 kaiming_init(m, mode='fan_in', nonlinearity='relu')
    #         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
    #                             nn.GroupNorm, nn.SyncBatchNorm)):
    #             if m.weight is not None:
    #                 nn.init.constant_(m.weight, 1)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z1:torch.Tensor, z2:torch.Tensor):
        p1 = self.layer(z1)
        p2 = self.layer(z2)
        
        losses = {}
        losses['loss'] = D(p1, z2) / 2 + D(p2, z1) / 2

        losses['output_std'] = torch.sqrt(torch.var(F.normalize(z1, dim=1)))
        
        return losses 
