import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from qtorch.quant import Quantizer, quantizer
from qtorch import FloatingPoint
import math

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, quant, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential() # shortcut
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        out = self.quant(self.conv1(self.quant(x)))
        out = F.relu(self.bn1(out)) #

        out = self.bn2(self.quant(self.conv2(self.quant(out))))

        out += self.shortcut(x)
        out = F.relu(out)

        return out 

class Bottleneck(nn.Module):
    expansion = 4 

    def __init__(self, in_planes, planes, quant, stride=1):
        super(Bottleneck, self).__init__()        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        out = self.quant(self.conv1(self.quant(x)))
        out = F.relu(self.bn1(out))

        out = self.quant(self.conv2(self.quant(out)))
        out = F.relu(self.bn2(out))

        out = self.bn3(self.quant(self.conv3(self.quant(out))))
        out += self.shortcut(x)
        out = F.relu(out)

        return out 

class ResNet(nn.Module):

    def __init__(self, block, quant, num_classes=10, depth=20):

        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, quant)
        self.layer2 = self._make_layer(block, 32, n, quant, stride=2)
        self.layer3 = self._make_layer(block, 64, n, quant, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.quant = quant()
        IBM_half = FloatingPoint(exp=6, man=9) # 设置对应的浮点数据类型
        self.quant_half = Quantizer(IBM_half, IBM_half, "nearest", "nearest")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1):
        layers = list()
        layers.append(block(self.inplanes, planes, quant , stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_half(x)
        x = self.conv1(x)
        x = self.quant(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)
        x = self.quant(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.quant_half(x)

        return x


def resnet20(quant):
    return ResNet(BasicBlock, quant, depth=20)

def resnet32(quant):
    return ResNet(BasicBlock, quant, depth=32)

def resnet44(quant):
    return ResNet(BasicBlock, quant, depth=44)

def resnet56(quant):
    return ResNet(BasicBlock, quant, depth=56)

def resnet110(quant):
    return ResNet(BasicBlock, quant, depth=110)