import torch
from torch import nn
import torch.nn.functional as F

import math
import numpy as np


class BConv2d(nn.Conv2d):
    def __init__(self, *args, binary=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary = binary

        self.alpha = nn.Parameter(torch.ones(1, self.out_channels, 1, 1))

    def forward(self, input):
        weight = self.weight
        if self.binary:
            input = torch.sign(input)
            weight = torch.sign(weight)

        out = F.conv2d(
            input, weight, self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        out = out.mul(self.alpha.expand_as(out))
        return out


class EBConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            num_experts=1,
            binary=False,
            activation=torch.sigmoid,
            use_se=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.num_experts = num_experts
        self.binary = binary
        self.use_se = use_se

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

        self.fc = nn.Linear(in_channels, num_experts)
        self.activation = activation

        self.alpha2 = nn.Parameter(torch.ones(1, out_channels, 1, 1))

        if use_se:
            self.se_fc = nn.Sequential(
                nn.Linear(in_channels, out_channels // 8, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 8, out_channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        B, C, H, W = x.size()
        avg_x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        gate_x = self.activation(self.fc(avg_x))
        if not self.binary:
            gate_x.fill_(0)
            gate_x[:, 0].fill_(1)

        base_weight = self.weight

        if self.binary:
            base_weight = torch.sign(base_weight)

        if self.binary:
            x_s = torch.argmax(gate_x, dim=1)
            weight = base_weight[x_s].squeeze().view(
                B * self.out_channels,
                self.in_channels // self.groups,
                self.kernel_size,
                self.kernel_size)
        else:
            # Note this is done for compatibility with the rest of the code, a
            # normal conv can be used. This is the case for the downsample
            # region
            weight = torch.matmul(
                gate_x,
                base_weight.view(self.num_experts, -1)
            ).view(B * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)

        bias = None
        if self.bias is not None:
            bias = torch.matmul(gate_x, self.bias).flatten()

        if self.binary:
            x = torch.sign(x)

            assert((x == -1).sum() + (x == 1).sum() == x.nelement())
            assert((weight == -1).sum() +
                   (weight == 1).sum() == weight.nelement())

        x = x.view(1, B * C, H, W)
        out = F.conv2d(
            x, weight, bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups * B
        )
        out = out.permute([1, 0, 2, 3]).view(
            B, self.out_channels, out.shape[-2], out.shape[-1])
        out = torch.mul(out, self.alpha2)

        if self.use_se:
            scaling = self.se_fc(avg_x)
            scaling = scaling.view(scaling.size(0), scaling.size(1), 1, 1)
            out = out.mul(scaling.expand_as(out))

        return out


def conv3x3(
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        dilation=1,
        num_experts=1,
        binary=False,
        activation=torch.sigmoid,
        use_se=False):
    """3x3 convolution with padding"""
    if groups < 1:
        groups = in_planes // 64
    if num_experts > 1:
        return EBConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
            num_experts=num_experts,
            binary=binary,
            activation=activation,
            use_se=use_se)
    else:
        if binary:
            return BConv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation,
                binary=binary)
        else:
            return nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, num_experts=1, binary=False,
            activation=torch.sigmoid, use_se=False):
    """1x1 convolution"""
    if num_experts > 1:
        return EBConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                        bias=False, num_experts=num_experts, binary=binary,
                        activation=activation, use_se=use_se)
    else:
        if binary:
            return BConv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                binary=binary)
        else:
            return nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_experts=1,
                 binary=False, activation=torch.sigmoid, use_se=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            dilation=dilation,
            groups=groups,
            num_experts=num_experts,
            binar=binary,
            activation=activation,
            use_se=use_se)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            planes,
            planes,
            dilation=dilation,
            groups=groups,
            num_experts=num_experts,
            binary=binary,
            activation=activation,
            use_se=use_se)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, None


class PreBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_experts=1,
                 binary=False, activation=torch.sigmoid, use_se=False):
        super(PreBasicBlock, self).__init__()
        self.binary = binary
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            dilation=dilation,
            groups=groups,
            num_experts=num_experts,
            binary=binary,
            activation=activation,
            use_se=use_se)

        self.bn1 = norm_layer(inplanes, eps=1e-4)
        self.conv2 = conv3x3(
            planes,
            planes,
            dilation=dilation,
            groups=groups,
            num_experts=num_experts,
            binary=binary,
            activation=activation,
            use_se=use_se)
        self.bn2 = norm_layer(planes, eps=1e-4)

        self.activation1 = nn.PReLU(num_parameters=planes)
        self.activation2 = nn.PReLU(num_parameters=planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.bn1(x)
        out1 = self.conv1(out1)
        out1 = self.activation1(out1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out1 += identity

        out2 = self.bn2(out1)
        out2 = self.conv2(out2)
        out2 = self.activation2(out2)

        out2 += out1

        return out2


class PreBasicBlockDeep(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_experts=1,
                 binary=False, activation=torch.sigmoid, use_se=False):
        super(PreBasicBlockDeep, self).__init__()
        self.binary = binary
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            dilation=dilation,
            groups=groups,
            num_experts=num_experts,
            binary=binary,
            activation=activation,
            use_se=use_se)
        self.bn1 = norm_layer(inplanes, eps=1e-4)

        self.conv2 = conv3x3(
            planes,
            planes,
            dilation=dilation,
            groups=groups,
            num_experts=num_experts,
            binary=binary,
            activation=activation,
            use_se=use_se)
        self.bn2 = norm_layer(planes, eps=1e-4)

        self.conv3 = conv1x1(
            planes,
            planes,
            num_experts=num_experts,
            binary=binary,
            activation=activation,
            use_se=use_se)
        self.bn3 = norm_layer(planes, eps=1e-4)

        self.activation1 = nn.PReLU(num_parameters=planes)
        self.activation2 = nn.PReLU(num_parameters=planes)
        self.activation3 = nn.PReLU(num_parameters=planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.bn1(x)
        out1 = self.conv1(out1)
        out1 = self.activation1(out1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out1 += identity

        out2 = self.bn2(out1)
        out2 = self.conv2(out2)
        out2 = self.activation2(out2)

        out2 += out1

        out3 = self.bn3(out2)
        out3 = self.conv3(out3)
        out3 = self.activation3(out3)

        out3 += out2

        return out3


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            groups=[
                1,
                1,
                1,
                1],
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            num_experts=1,
            binary=False,
            activation=torch.sigmoid,
            use_se=False,
            expansion=[
                1,
                1,
                1,
                1],
        add_g_layer=False,
            decompose_downsample=-1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_experts = num_experts
        self.binary = binary
        self.activation = activation
        self.use_se = use_se
        self.add_g_layer = add_g_layer
        self.decompose_downsample = decompose_downsample

        print(layers)
        print(expansion)
        print(groups)
        print(f'Num classes = {num_classes}')

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        print(f'Dil: {replace_stride_with_dilation}')
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                self.inplanes // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            norm_layer(
                self.inplanes // 2),
            nn.ReLU(
                inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                self.inplanes // 2,
                self.inplanes // 4,
                1,
                1,
                bias=False),
            norm_layer(
                self.inplanes // 4),
            nn.ReLU())
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                self.inplanes // 4,
                self.inplanes // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            norm_layer(
                self.inplanes // 2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, 1, 1, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(
            64 * expansion[0]), layers[0], groups=self.groups[0])
        self.layer2 = self._make_layer(block,
                                       int(128 * expansion[1]),
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       groups=self.groups[1])
        self.layer3 = self._make_layer(block,
                                       int(256 * expansion[2]),
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       groups=self.groups[2])
        self.layer4 = self._make_layer(block,
                                       int(512 * expansion[3]),
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       groups=self.groups[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            int(512 * expansion[3]) * block.expansion, num_classes)

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            dilate=False,
            groups=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        current_block = block
        if self.add_g_layer and groups > 1 and self.binary:
            current_block = PreBasicBlockDeep

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * current_block.expansion:
            if self.binary:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False),
                    conv1x1(
                        self.inplanes,
                        self.inplanes //
                        self.decompose_downsample,
                        1,
                        num_experts=self.num_experts),
                    norm_layer(
                        self.inplanes //
                        self.decompose_downsample),
                    nn.PReLU(
                        self.inplanes //
                        self.decompose_downsample),
                    conv1x1(
                        self.inplanes //
                        self.decompose_downsample,
                        planes *
                        current_block.expansion,
                        1,
                        num_experts=self.num_experts),
                    norm_layer(
                        planes *
                        current_block.expansion),
                    nn.PReLU(
                        planes *
                        current_block.expansion))
            else:
                downsample = nn.Sequential(
                    norm_layer(
                        self.inplanes,
                        eps=1e-4),
                    conv1x1(
                        self.inplanes,
                        planes *
                        current_block.expansion,
                        stride,
                        num_experts=self.num_experts,
                        binary=self.binary,
                        activation=self.activation,
                        use_se=self.use_se),
                )

        layers = []
        layers.append(
            current_block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                num_experts=self.num_experts,
                binary=self.binary,
                activation=self.activation,
                use_se=self.use_se))
        self.inplanes = planes * current_block.expansion
        for _ in range(1, blocks):
            layers.append(
                current_block(
                    self.inplanes,
                    planes,
                    groups=groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    num_experts=self.num_experts,
                    binary=self.binary,
                    activation=self.activation,
                    use_se=self.use_se))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = torch.cat(
            [
                self.conv2_2(self.conv2_1(x)),
                self.maxpool(x)
            ],
            dim=1
        )
        x = self.conv3(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_avg = self.avgpool(x)
        x = torch.flatten(x_avg, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet_generic(structure=[2, 2, 2, 2], block_type=None, **kwargs):
    if block_type is None:
        if kwargs['binary']:
            block = PreBasicBlock
        else:
            block = BasicBlock
    else:
        block = eval(block_type)

    return ResNet(block, structure, **kwargs)
