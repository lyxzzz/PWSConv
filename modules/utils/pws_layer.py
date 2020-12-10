import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import math

def pws_init(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode=module.mode, nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)
    if module.alpha is not None:
        nn.init.constant_(module.alpha, 1)

class PWSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, initalpha=False, equiv=False, gamma=1e-5, mode='fan_out', bias=True, activation=None):
        super(PWSConv, self).__init__()

        if isinstance(kernel_size, list):
            assert kernel_size[0] == kernel_size[1]
        
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.kernel = None
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None
        self.alpha = Parameter(torch.Tensor(out_channels, 1, 1, 1))

        self.stride = stride
        self.padding = padding
        self.mode = mode
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma

        self.initalpha = initalpha
        if initalpha:
            self._forwardfunc = self.__first_forward
        else:
            self._forwardfunc = self.__normal_forward

        receptive_field_size = self.weight[0][0].numel()
        fan_in = in_channels * receptive_field_size
        fan_out = out_channels * receptive_field_size
        fan = fan_in if mode == 'fan_in' else fan_out

        self.mode_shape = [1, 2, 3]
        self.norm_shape = [in_channels, kernel_size, kernel_size]

        self.constval = math.sqrt(2/float(fan))
        if equiv:
            self.gamma = 2 / float(fan) * self.gamma

        # self.kernel_norm = nn.LayerNorm([in_channels, kernel_size, kernel_size], eps=gamma, elementwise_affine=False)
        self.activation = activation

        pws_init(self)

    def __cal_weight(self):
        ex = self.weight.mean(axis=self.mode_shape, keepdim=True)
        var = self.weight.var(axis=self.mode_shape, keepdim=True) + self.gamma

        weight = self.constval * self.alpha * (self.weight - ex) / var.sqrt()
        return weight

    def __normal_forward(self, x):
        if self.training:
            kernel = self.constval * self.alpha * F.layer_norm(self.weight, self.norm_shape, weight=None, bias=None, eps=self.gamma)
        else:
            kernel = self.kernel
        output = F.conv2d(x, kernel, bias=self.bias, padding=self.padding, stride=self.stride)

        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def __first_forward(self, x):
        assert self.training
        self._forwardfunc = self.__normal_forward
        
        kernel = self.constval * self.alpha * F.layer_norm(self.weight, self.norm_shape, weight=None, bias=None, eps=self.gamma)
        output = F.conv2d(x, kernel, bias=self.bias, padding=self.padding, stride=self.stride)

        outputvar = output.var()
        nn.init.constant_(self.alpha, torch.sqrt(2 / outputvar))

        kernel = self.constval * self.alpha * F.layer_norm(self.weight, self.norm_shape, weight=None, bias=None, eps=self.gamma)
        output = F.conv2d(x, kernel, bias=self.bias, padding=self.padding, stride=self.stride)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def train(self, mode=True):
        super(PWSConv, self).train(mode)
        if not mode:
            self.kernel = self.constval * self.alpha * F.layer_norm(self.weight, self.norm_shape, weight=None, bias=None, eps=self.gamma)

    def forward(self, x):
        return self._forwardfunc(x)

    def extra_repr(self):
        pws_str = 'in_channels={}, out_channels={}, stride={}, kernel_size={}, mode={}, gamma={}, initalpha={}'.format(
            self.in_channels, self.out_channels, self.stride, self.kernel_size, self.mode, self.gamma, self.initalpha)
        return pws_str