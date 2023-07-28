import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate


__all__ = ['SpikingMobileNet', 'spikingmobilenet']


def sew_function(x, y, cnf):
    if cnf == "ADD":
        return x + y
    elif cnf == "AND":
        return x * y
    elif cnf == "OR":
        return x + y - x * y
    else:
        raise NotImplementedError
        

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, inverted=1, cnf=None):
        super(BottleneckBlock, self).__init__()
        self.residual = in_planes == out_planes and stride == 1
        hidden_planes = in_planes * inverted
        modulelist = []
        if inverted != 1:
            modulelist.append(layer.Conv2d(in_planes, hidden_planes, kernel_size=1, stride=1, bias=False))
            modulelist.append(layer.BatchNorm2d(hidden_planes))
            modulelist.append(neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
        
        modulelist.append(layer.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=stride, padding=1, groups=hidden_planes, bias=False))
        modulelist.append(layer.BatchNorm2d(hidden_planes))
        modulelist.append(neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
        
        modulelist.append(layer.Conv2d(hidden_planes, out_planes, kernel_size=1, stride=1, bias=False))
        modulelist.append(layer.BatchNorm2d(out_planes))
        modulelist.append(neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
        self.layers = nn.Sequential(*modulelist)
        
        self.cnf = cnf

    def forward(self, x):
        out = self.layers(x)
        
        if self.residual:
            out = sew_function(out, x, self.cnf)

        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class SpikingMobileNet(nn.Module):
    def __init__(self, layers_config, num_classes=1000, cnf=None):
        super(SpikingMobileNet, self).__init__()
        self.conv1 = layer.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        modulelist = []
        in_planes = 32
        for config in layers_config:
            layers = []
            layers.append(
                BottleneckBlock(
                    in_planes=in_planes, 
                    out_planes=config[0], 
                    stride=config[1], 
                    inverted=config[2],
                    cnf=cnf
                )
            )
            in_planes = config[0]
            for i in range(1, config[3]):
                layers.append(
                    BottleneckBlock(
                        in_planes=in_planes, 
                        out_planes=config[0], 
                        stride=1, 
                        inverted=config[2],
                        cnf=cnf
                    )
                )
            layers = nn.Sequential(*layers)
            modulelist.append(layers)
        
        modulelist.append(layer.Conv2d(in_planes, 1280, kernel_size=1, stride=1, bias=False))
        modulelist.append(layer.BatchNorm2d(1280))
        modulelist.append(neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
        self.layers = nn.Sequential(*modulelist)
        
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.layers(out)

        out = self.avgpool(out)
        if self.avgpool.step_mode == 's':
            out = torch.flatten(out, 1)
        elif self.avgpool.step_mode == 'm':
            out = torch.flatten(out, 2)
        out = self.fc(out)

        return out

    
def spikingmobilenet(**kwargs):
    return SpikingMobileNet(
        [[16, 1, 1, 1],
         [24, 2, 6, 2],
         [32, 2, 6, 3],
         [64, 2, 6, 4],
         [96, 1, 6, 3],
         [160, 2, 6, 3],
         [320, 1, 6, 1],
        ],
        **kwargs
    )
