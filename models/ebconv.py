import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t

import math
import numpy as np
from functools import partial
from typing import Union

from bnn import BConfig, bconfig
from bnn.layers.helpers import copy_paramters


class BinarySoftActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return (input == input.max(dim=1, keepdim=True)
                [0]).view_as(input).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input


class EBConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        num_experts: int = 1,
        activation=torch.sigmoid,
        use_only_first: bool = False,
        use_se: bool = True,
        bconfig: BConfig = None
    ) -> None:
        super(EBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.num_experts = num_experts
        self.bconfig = bconfig
        self.use_se = use_se
        self.use_only_first = use_only_first

        self.weight = nn.Parameter(
            torch.Tensor(
                num_experts,
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(num_experts, self.out_channels)
            )
        else:
            self.register_parameter('bias', None)

        # ebcond head
        self.fc = nn.Linear(in_channels, num_experts)
        self.activation = activation

        # se head
        if self.use_se:
            self.se_fc = nn.Sequential(
                nn.Linear(in_channels, out_channels // 8, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 8, out_channels, bias=False),
                nn.Sigmoid()
            )

        if bconfig is not None:
            self.activation_pre_process = bconfig.activation_pre_process()
            self.activation_post_process = bconfig.activation_post_process(
                self, shape=[1, out_channels, 1, 1])
            self.weight_pre_process = bconfig.weight_pre_process()

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in = np.prod(self.weight.shape[2:])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], a=-bound, b=bound)

    def forward(self, x):
        B, C, H, W = x.size()

        # Compute the expert selection
        avg_x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gate_x = self.activation(self.fc(avg_x))
        # WTA function with Identity
        gate_x = BinarySoftActivation.apply(gate_x)

        # Supress the expert selection temporarily, select expert 0 always.
        if (self.bconfig is None or isinstance(
                self.activation_pre_process,
                nn.Identity)) or self.use_only_first:
            gate_x = gate_x * torch.zeros_like(gate_x)
            gate_x[:, 0] = gate_x[:, 0] + torch.ones_like(gate_x[:, 0])

        base_weight = self.weight
        weight = torch.matmul(
            gate_x,
            base_weight.view(self.num_experts, -1)
        ).view(B * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)

        bias = None
        if self.bias is not None:
            bias = torch.matmul(gate_x, self.bias).flatten()

        # Binarize the weights and the input features
        if self.bconfig is not None:
            weight = self.weight_pre_process(weight)
            x = self.activation_pre_process(x)

        x = x.view(1, B * C, H, W)
        out = F.conv2d(
            x, weight, bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups * B
        )
        out = out.permute([1, 0, 2, 3]).view(
            B, self.out_channels, out.shape[-2], out.shape[-1])

        # Apply learnable alpha if set
        if self.bconfig is not None:
            out = self.activation_post_process(out, x)

        if self.use_se:
            scaling = self.se_fc(avg_x)  # Use feature pre-binarization
            scaling = scaling.view(B, scaling.size(1), 1, 1)
            out = out.mul(scaling.expand_as(out))

        return out

    @classmethod
    def from_module(
            cls,
            mod: nn.Module,
            bconfig: BConfig = None,
            update: bool = False):
        if not bconfig:
            assert hasattr(
                mod, 'bconfig'), 'The input modele requires a predifined bconfig'
            assert mod.bconfig, 'The input model bconfig is invalid'
            bconfig = mod.bconfig
        bnn_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            num_experts=mod.num_experts,
            activation=mod.activation,
            use_only_first=mod.use_only_first,
            use_se=mod.use_se,
            bconfig=bconfig)
        bnn_conv.weight = mod.weight
        bnn_conv.bias = mod.bias

        if update:
            copy_paramters(mod, bnn_conv, bconfig)

        return bnn_conv
