import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate


__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101', 'sew_resnet152']


def sew_function(x, y, cnf):
    if cnf == "ADD":
        return x + y
    elif cnf == "AND":
        return x * y
    elif cnf == "OR":
        return x + y - x * y
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, cnf=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.stride = stride
        self.cnf = cnf
    
    def forward(self, x):
        identity = x
        
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        
        out = sew_function(out, identity, self.cnf)
        
        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, cnf=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.stride = stride
        self.cnf = cnf
    
    def forward(self, x):
        identity = x
        
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        out = self.sn3(self.bn3(self.conv3(out)))
        
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        
        out = sew_function(out, identity, self.cnf)
        
        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class SEWResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_groups=64, 
                 norm_layer=None, cnf=None, zero_init_residual=False):
        super(SEWResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_groups
        self.conv1 = layer.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.in_planes)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cnf=cnf)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cnf=cnf)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cnf=cnf)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride=1, cnf=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, self._norm_layer, cnf))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer, cnf=cnf))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)
        
        return x

        
def sew_resnet18(**kwargs):
    return SEWResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return SEWResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return SEWResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101(**kwargs):
    return SEWResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152(**kwargs):
    return SEWResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
