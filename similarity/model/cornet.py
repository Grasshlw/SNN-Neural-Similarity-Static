## Reproduce from https://github.com/dicarlolab/CORnet.git

from collections import OrderedDict
import torch
from torch import nn
import math


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


class CORnet_Z(nn.Module):
    
    def __init__(self, init_weights=True):
        super().__init__()
        self.V1 = CORblock_Z(3, 64, kernel_size=7, stride=2)
        self.V2 = CORblock_Z(64, 128)
        self.V4 = CORblock_Z(128, 256)
        self.IT = CORblock_Z(256, 512)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ]))

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.V1(x)
        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)
        x = self.decoder(x)
        
        return x


class CORblock_RT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape]).to(self.conv_input.weight.device)
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state


class CORnet_RT(nn.Module):

    def __init__(self, times=5):
        super().__init__()
        self.times = times

        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_RT(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_RT(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_RT(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000))
        ]))

    def forward(self, inp, n_layer=6):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                this_inp = outputs['inp']
            else:  # at t=0 there is no input yet to V2 and up
                this_inp = None
            new_output, new_state = getattr(self, block)(this_inp, batch_size=len(outputs['inp']))
            outputs[block] = new_output
            states[block] = new_state

        for t in range(1, self.times):
            new_outputs = {'inp': inp}
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                prev_state = states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                new_outputs[block] = new_output
                states[block] = new_state
            outputs = new_outputs

        if n_layer == 1:
            out = outputs['V1']
        elif n_layer == 2:
            out = outputs['V2']
        elif n_layer == 3:
            out = outputs['V4']
        elif n_layer == 4:
            out = outputs['IT']
        elif n_layer == 5:
            out = self.decoder.avgpool(outputs['IT'])
            out = self.decoder.flatten(out)
        else:
            out = self.decoder(outputs['IT'])
        return out


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnet_S(nn.Module):
    
    def __init__(self, init_weights=True):
        super().__init__()
        self.V1 = nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))
        self.V2 = CORblock_S(64, 128, times=2)
        self.V4 = CORblock_S(128, 256, times=4)
        self.IT = CORblock_S(256, 512, times=2)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ]))

        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        # weight initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # nn.Linear is missing here because I originally forgot 
            # to add it during the training of this network
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.V1(x)
        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)
        x = self.decoder(x)

        return x


def cornet_z(checkpoint_path=None):
    model = CORnet_Z(init_weights=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        key = k[7:]
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)
    return model


def cornet_rt(checkpoint_path=None):
    model = CORnet_RT()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        key = k[7:]
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)
    return model


def cornet_s(checkpoint_path=None):
    model = CORnet_S(init_weights=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        key = k[7:]
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)
    return model
