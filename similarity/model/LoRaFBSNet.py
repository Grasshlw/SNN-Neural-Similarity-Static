import torch
import torch.nn as nn
from spikingjelly.activation_based import base, layer, neuron, surrogate


__all__ = ['LoRaFBSNet', 'ConvRecurrentContainer', 'lorafb_snet18', 'lorafb_snet34', 'lorafb_snet50', 'lorafb_snet101', 'lorafb_snet152']


class ConvRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module, in_channels, out_channels, stride, step_mode="s"):
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, "step_mode") or sub_module.step_mode == "s"
        self.sub_module_out_channels = out_channels
        self.sub_module = sub_module
        
        if stride > 1:
            self.dwconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, 
                                             padding=1, output_padding=1, groups=out_channels, bias=False)
            self.bn1 = layer.BatchNorm2d(out_channels)
            self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        else:
            self.dwconv = None
        self.pwconv = nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = layer.BatchNorm2d(in_channels)
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.register_memory("y", None)
    
    def single_step_forward(self, x):
        if self.y is None:
            if self.dwconv is None:
                self.y = torch.zeros(x.size(0), self.sub_module_out_channels, x.size(2), x.size(3)).to(x)
            else:
                h = (x.size(2) + 2 - (3 - 1) - 1) // 2 + 1
                w = (x.size(3) + 2 - (3 - 1) - 1) // 2 + 1
                self.y = torch.zeros(x.size(0), self.sub_module_out_channels, h, w).to(x)
        if self.dwconv is None:
            out = self.y
        else:
            out = self.sn1(self.bn1(self.dwconv(self.y)))
        out = torch.cat((x, out), dim=1)
        out = self.bn(self.pwconv(out))
        out = self.sn(out)
        self.y = self.sub_module(out)
        
        return self.y


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
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
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
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
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


class LoRaFBSNet(nn.Module):
    def __init__(self, block, layers, num_classes=101, groups=1, width_per_groups=64, 
                 norm_layer=None, cnf=None, zero_init_residual=False):
        super(LoRaFBSNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_groups
        self.conv1 = layer.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.in_planes)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(block, 64, layers[0], cnf=cnf)
        self.recurrent_layer1 = ConvRecurrentContainer(layer1, in_channels=64, out_channels=64 * block.expansion, stride=1)
        layer2 = self._make_layer(block, 128, layers[1], stride=2, cnf=cnf)
        self.recurrent_layer2 = ConvRecurrentContainer(layer2, in_channels=64 * block.expansion, out_channels=128 * block.expansion, stride=2)
        layer3 = self._make_layer(block, 256, layers[2], stride=2, cnf=cnf)
        self.recurrent_layer3 = ConvRecurrentContainer(layer3, in_channels=128 * block.expansion, out_channels=256 * block.expansion, stride=2)
        layer4 = self._make_layer(block, 512, layers[3], stride=2, cnf=cnf)
        self.recurrent_layer4 = ConvRecurrentContainer(layer4, in_channels=256 * block.expansion, out_channels=512 * block.expansion, stride=2)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, (layer.Conv2d, nn.Conv2d, nn.ConvTranspose2d)):
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
        
        x = self.recurrent_layer1(x)
        x = self.recurrent_layer2(x)
        x = self.recurrent_layer3(x)
        x = self.recurrent_layer4(x)
        
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)
        
        return x

        
def lorafb_snet18(**kwargs):
    return LoRaFBSNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def lorafb_snet34(**kwargs):
    return LoRaFBSNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def lorafb_snet50(**kwargs):
    return LoRaFBSNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def lorafb_snet101(**kwargs):
    return LoRaFBSNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def lorafb_snet152(**kwargs):
    return LoRaFBSNet(Bottleneck, [3, 8, 36, 3], **kwargs)
