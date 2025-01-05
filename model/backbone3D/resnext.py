import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32, 
                 pretrain_path=None):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        
        # Modified first conv layer for HD resolution
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),  # Reduced temporal stride
            padding=(3, 3, 3),
            bias=False)
        
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Modified pooling for HD resolution
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), 
                                  stride=(1, 2, 2),  # Reduced temporal stride
                                  padding=(0, 1, 1))
        
        # Modified layers with adjusted stride for HD resolution
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                     cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=(1, 2, 2))
        
        # Adaptive pooling for variable input sizes
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pretrain_path = pretrain_path
        
        # Add input size validation
        self.input_size = (1080, 1920)  # height, width

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), 
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input validation
        if x.size(-2) != 1080 or x.size(-1) != 1920:
            raise ValueError(f"Input spatial dimensions must be 1080x1920, got {x.size(-2)}x{x.size(-1)}")
            
        # Memory optimization warning
        if x.size(0) > 2:  # Batch size > 2
            print("Warning: Large batch size with HD resolution may cause OOM issues")
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        return x

    def load_pretrain(self):
        if self.pretrain_path is None:
            return
            
        state_dict = self.state_dict()
        pretrain_state_dict = torch.load(self.pretrain_path)

        for param_name, value in pretrain_state_dict['state_dict'].items():
            param_name = param_name.split('.', 1)[1]
            if param_name not in state_dict:
                continue
            state_dict[param_name] = value
            
        self.load_state_dict(state_dict)
        print("backbone3D : resnext pretrained loaded!", flush=True)


def resnext50(**kwargs):
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(config):
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 
                    pretrain_path=config['BACKBONE3D']['RESNEXT']['PRETRAIN']['ver_101'])
    # Initialize with weights suitable for HD resolution
    if model.pretrain_path is None:
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    return model


def resnext152(**kwargs):
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
