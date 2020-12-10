import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import (build_norm_layer,
                      constant_init, kaiming_init)

from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcv.runner import load_checkpoint

from ..utils import PWSConv, pws_init, build_conv_layer


class BasicBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1, self.norm1 = build_conv_layer(conv_cfg, in_channels, self.mid_channels,
                                        kernel_size=3, stride=stride, padding=1,
                                        norm_cfg=norm_cfg)

        if "initalpha" in conv_cfg:
            initalpha = conv_cfg["initalpha"]
            conv_cfg["initalpha"] = False

        self.conv2, self.norm2 = build_conv_layer(conv_cfg, self.mid_channels, out_channels,
                                        kernel_size=3, stride=1, padding=1, 
                                        norm_cfg=norm_cfg)
        if "initalpha" in conv_cfg and initalpha:
            conv_cfg["initalpha"] = True

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            x = self.relu(x)
            
            out = self.conv1(x)
            if self.norm1 is not None:
                out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            if self.norm2 is not None:
                out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        return out


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.conv1, self.norm1 = build_conv_layer(conv_cfg, in_channels, self.mid_channels,
                                        kernel_size=1, stride=self.conv1_stride, padding=0,
                                        norm_cfg=norm_cfg)
        
        if "initalpha" in conv_cfg:
            initalpha = conv_cfg["initalpha"]
            conv_cfg["initalpha"] = False

        self.conv2, self.norm2 = build_conv_layer(conv_cfg, self.mid_channels, self.mid_channels,
                                        kernel_size=3, stride=self.conv2_stride, padding=1, 
                                        norm_cfg=norm_cfg)

        self.conv3, self.norm3 = build_conv_layer(conv_cfg, self.mid_channels, out_channels,
                                        kernel_size=1, stride=1, padding=0,
                                        norm_cfg=norm_cfg)
        
        if "initalpha" in conv_cfg and initalpha:
            conv_cfg["initalpha"] = True

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            x = self.relu(x)

            out = self.conv1(x)
            if self.norm1 is not None:
                out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            if self.norm2 is not None:
                out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            if self.norm3 is not None:
                out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out
        out = _inner_forward(x)

        return out


def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            
            downsample_conv, downsample_norm = build_conv_layer(conv_cfg, in_channels, out_channels,
                                                                kernel_size=1, stride=conv_stride, padding=0,
                                                                norm_cfg=norm_cfg)
            downsample.append(downsample_conv)
            if downsample_norm is not None:
                downsample.append(downsample_norm)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 out_indices=(3, ),
                 style='pytorch',
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                style=self.style,
                avg_down=self.avg_down,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1, self.norm1 = build_conv_layer(self.conv_cfg, in_channels, stem_channels,
                                                kernel_size=7, stride=2, padding=3, 
                                                norm_cfg=self.norm_cfg)
        if self.norm1 is None:
            self.norm1 = build_norm_layer(self.norm_cfg, stem_channels)[1]
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.norm1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, PWSConv):
                pws_init(m)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)
        
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

class ResNet_Cifar(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_Cifar, self).__init__(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1, self.norm1 = build_conv_layer(self.conv_cfg, in_channels, stem_channels,
                                                kernel_size=3, stride=1, padding=1, 
                                                norm_cfg=self.norm_cfg)
        if self.norm1 is None:
            self.norm1 = build_norm_layer(self.norm_cfg, stem_channels)[1]
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = None

