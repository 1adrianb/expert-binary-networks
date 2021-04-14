from typing import Tuple, Type, Union, List, Optional, Callable, Any

import torch
from torch.functional import Tensor
import torch.nn as nn
from bnn.models.resnet import ResNet, _resnet

from .ebconv import EBConv2d


def conv3x3(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        num_experts: int = 4,
        ebconv_activation=torch.sigmoid,
        use_only_first: bool = False,
        use_se: bool = True) -> nn.Module:
    """3x3 convolution with padding"""
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
            activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
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


def conv1x1(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        num_experts: int = 4,
        ebconv_activation=torch.sigmoid,
        use_only_first: bool = False,
        use_se: bool = True) -> nn.Module:
    """1x1 convolution"""
    if num_experts > 1:
        return EBConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            bias=False,
            num_experts=num_experts,
            activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
    else:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            bias=False)


class EBBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation=nn.ReLU,
                 num_experts: int = 4,
                 ebconv_activation=torch.sigmoid,
                 use_only_first: bool = False,
                 use_se: bool = True
                 ) -> None:
        super(EBBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            groups=groups,
            num_experts=num_experts,
            ebconv_activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(
            planes,
            planes,
            groups=groups,
            num_experts=num_experts,
            ebconv_activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
        self.bn2 = norm_layer(planes)
        self.act1 = activation(
            inpace=True) if activation == nn.ReLU else activation(
            num_parameters=planes)
        self.act2 = activation(
            inpace=True) if activation == nn.ReLU else activation(
            num_parameters=planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.bn1(x)
        out1 = self.conv1(out1)
        out1 = self.act1(out1)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out1 += identity

        out2 = self.bn2(out1)
        out2 = self.conv2(out2)
        out2 = self.act2(out2)

        out2 += out1

        return out2


class EBDeepBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation=nn.ReLU,
                 num_experts: int = 4,
                 ebconv_activation=torch.sigmoid,
                 use_only_first: bool = False,
                 use_se: bool = True
                 ) -> None:
        super(EBDeepBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            groups=groups,
            num_experts=num_experts,
            ebconv_activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(
            planes,
            planes,
            groups=groups,
            num_experts=num_experts,
            ebconv_activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(
            planes,
            planes,
            num_experts=num_experts,
            ebconv_activation=ebconv_activation,
            use_only_first=use_only_first,
            use_se=use_se)
        self.bn3 = norm_layer(planes)
        self.act1 = activation(
            inpace=True) if activation == nn.ReLU else activation(
            num_parameters=planes)
        self.act2 = activation(
            inpace=True) if activation == nn.ReLU else activation(
            num_parameters=planes)
        self.act3 = activation(
            inpace=True) if activation == nn.ReLU else activation(
            num_parameters=planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.bn1(x)
        out1 = self.conv1(out1)
        out1 = self.act1(out1)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out1 += identity

        out2 = self.bn2(out1)
        out2 = self.conv2(out2)
        out2 = self.act2(out2)

        out2 += out1

        out3 = self.bn3(out2)
        out3 = self.conv3(out3)
        out3 = self.act3(out3)

        out3 += out2

        return out3


class EBResNet(ResNet):

    def __init__(
        self,
        block: Type[Union[EBBasicBlock, EBDeepBasicBlock]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: List[int] = [1, 1, 1, 1],
        expansion: List[int] = [1, 1, 1, 1],
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
        stem_type: str = 'basic',
        num_experts: int = 4,
        ebconv_activation=torch.sigmoid,
        use_only_first: bool = False,
        use_se: bool = True,
        downsample_ratio: int = 1
    ) -> None:
        self.expansion = expansion
        self.num_experts = num_experts
        self.ebconv_activation = ebconv_activation
        self.use_only_first = use_only_first
        self.use_se = use_se
        self.downsample_ratio = downsample_ratio
        self._counter = 0

        super(EBResNet, self).__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            activation=activation,
            stem_type=stem_type
        )

    def _make_layer(self,
                    block: Type[Union[EBDeepBasicBlock,
                                      EBDeepBasicBlock]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        planes = int(planes * self.expansion[self._counter])
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.downsample_ratio == 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False),
                    conv1x1(
                        self.inplanes,
                        planes *
                        block.expansion,
                        num_experts=self.num_experts,
                        stride=1),
                    norm_layer(
                        planes *
                        block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False),
                    conv1x1(
                        self.inplanes,
                        self.inplanes //
                        self.downsample_ratio,
                        stride=1,
                        num_experts=self.num_experts),
                    norm_layer(
                        self.inplanes //
                        self.downsample_ratio),
                    nn.PReLU(
                        self.inplanes //
                        self.downsample_ratio),
                    conv1x1(
                        self.inplanes //
                        self.downsample_ratio,
                        planes *
                        block.expansion,
                        stride=1,
                        num_experts=self.num_experts),
                    norm_layer(
                        planes *
                        block.expansion),
                    nn.PReLU(
                        planes *
                        block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups[self._counter],
                self.base_width,
                previous_dilation,
                norm_layer,
                activation=self._activation,
                num_experts=self.num_experts,
                ebconv_activation=self.ebconv_activation,
                use_only_first=self.use_only_first,
                use_se=self.use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups[self._counter],
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=self._activation,
                    num_experts=self.num_experts,
                    ebconv_activation=self.ebconv_activation,
                    use_only_first=self.use_only_first,
                    use_se=self.use_se))
        self._counter += 1
        self.outplanes = planes
        return nn.Sequential(*layers)

    def _forward_impl(self,
                      x: torch.Tensor) -> Tuple[torch.Tensor,
                                                List[torch.Tensor]]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.stem_type == 'basic':
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

        all_outs = []
        x = self.layer1(x)
        all_outs.append(x)
        x = self.layer2(x)
        all_outs.append(x)
        x = self.layer3(x)
        all_outs.append(x)
        x = self.layer4(x)
        all_outs.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, all_outs


def resnet_generic(block_type: Optional[Type[Union[EBBasicBlock, EBDeepBasicBlock]]]
                   = EBDeepBasicBlock, structure=[2, 2, 2, 2], **kwargs: Any) -> EBResNet:
    return EBResNet(block_type, layers=structure, **kwargs)
