import torch.nn as nn

from .pws_layer import PWSConv, pws_init
from mmcv.cnn import build_norm_layer


def build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=0, padding=0, norm_cfg=None):
    convtype = conv_cfg.pop('type')
    norm_layer = None
    if convtype == 'pws':
        conv_layer = PWSConv(in_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding=padding, bias=True, **conv_cfg)
    else:
        conv_layer = nn.Conv2d(in_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding=padding, bias=(norm_cfg is None))
        
        if norm_cfg is not None:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
    conv_cfg['type'] = convtype
    return conv_layer, norm_layer