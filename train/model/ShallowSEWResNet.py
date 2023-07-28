import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate

__all__ = ['ShallowSEWResNet', 
           'sew_resnet8', 'sew_resnet14', 'wide_sew_resnet8', 'wide_sew_resnet14', 
           'sew_resnext11', 'sew_resnext20', 'sew_resnext11_inverted', 'sew_resnext20_inverted']


def sew_function(x, y, cnf):
    if cnf == "ADD":
        return x + y
    elif cnf == "AND":
        return x * y
    elif cnf == "OR":
        return x + y - x * y
    else:
        raise NotImplementedError


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, inverted=False, downsample=None, cnf=None):
        super(BasicBlock, self).__init__()
        self.conv1 = layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(out_planes)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.conv2 = layer.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(out_planes)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.cnf = cnf

    def forward(self, x):
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))

        if self.downsample is not None:
            x = self.downsample_sn(self.downsample(x))

        out = sew_function(out, x, self.cnf)

        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, inverted=False, downsample=None, cnf=None):
        super(BottleneckBlock, self).__init__()
        if inverted:
            hidden_planes = in_planes * 6
        else:
            hidden_planes = in_planes
        if in_planes == out_planes:
            hidden_planes = hidden_planes // 2
        
        self.conv1 = layer.Conv2d(in_planes, hidden_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = layer.BatchNorm2d(hidden_planes)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.conv2 = layer.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = layer.BatchNorm2d(hidden_planes)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.conv3 = layer.Conv2d(hidden_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = layer.BatchNorm2d(out_planes)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.cnf = cnf

    def forward(self, x):
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        out = self.sn3(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            x = self.downsample_sn(self.downsample(x))

        out = sew_function(out, x, self.cnf)

        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class ShallowSEWResNet(nn.Module):
    def __init__(self, layers_depth, block, num_classes=1000, widen_factor=1, inverted=False, cnf=None):
        super(ShallowSEWResNet, self).__init__()
        assert (isinstance(layers_depth, list) and len(layers_depth) == 3)
        nChannels = [64, 128 * widen_factor, 256 * widen_factor, 512 * widen_factor]
        self.inverted = inverted

        self.conv1 = layer.Conv2d(3, nChannels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = layer.BatchNorm2d(nChannels[0])
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_layer(layers_depth[0], nChannels[0], nChannels[1], block, 2, cnf)
        self.block2 = self._make_layer(layers_depth[1], nChannels[1], nChannels[2], block, 2, cnf)
        self.block3 = self._make_layer(layers_depth[2], nChannels[2], nChannels[3], block, 2, cnf)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, n_layer, in_planes, out_planes, block, stride, cnf=None):
        layers = []
        downsample = nn.Sequential(
            layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
            layer.BatchNorm2d(out_planes)
        )
        layers.append(block(in_planes, out_planes, stride, self.inverted, downsample, cnf))
        for i in range(1, n_layer):
            layers.append(block(out_planes, out_planes, 1, inverted=self.inverted, cnf=cnf))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.maxpool(out)
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.avgpool(out)
        if self.avgpool.step_mode == 's':
            out = torch.flatten(out, 1)
        elif self.avgpool.step_mode == 'm':
            out = torch.flatten(out, 2)
        out = self.fc(out)

        return out


def sew_resnet8(**kwargs):
    return ShallowSEWResNet([1, 1, 1], BasicBlock, **kwargs)


def sew_resnet14(**kwargs):
    return ShallowSEWResNet([2, 2, 2], BasicBlock, **kwargs)


def wide_sew_resnet8(**kwargs):
    return ShallowSEWResNet([1, 1, 1], BasicBlock, widen_factor=4, **kwargs)


def wide_sew_resnet14(**kwargs):
    return ShallowSEWResNet([2, 2, 2], BasicBlock, widen_factor=4, **kwargs)


def sew_resnext11(**kwargs):
    return ShallowSEWResNet([1, 1, 1], BottleneckBlock, **kwargs)


def sew_resnext20(**kwargs):
    return ShallowSEWResNet([2, 2, 2], BottleneckBlock, **kwargs)


def sew_resnext11_inverted(**kwargs):
    return ShallowSEWResNet([1, 1, 1], BottleneckBlock, inverted=True, **kwargs)


def sew_resnext20_inverted(**kwargs):
    return ShallowSEWResNet([2, 2, 2], BottleneckBlock, inverted=True, **kwargs)
