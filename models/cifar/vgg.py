
"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms
from qtorch import FloatingPoint
from qtorch.quant import Quantizer

__all__ = ["vgg16", "vgg19", "vgg16bn", "vgg19bn"]


def make_layers(cfg, quant, batch_norm=False):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != "N"
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            if use_quant:
                layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)


cfg = {
    16: [
        "64",
        "64",
        "M",
        "128",
        "128",
        "M",
        "256",
        "256",
        "256",
        "M",
        "512",
        "512",
        "512",
        "M",
        "512",
        "512",
        "512",
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, quant=None, num_classes=10, depth=16, batch_norm=False):

        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], quant, batch_norm)
        IBM_half = FloatingPoint(exp=6, man=9)
        quant_half = lambda: Quantizer(IBM_half, IBM_half, "nearest", "nearest")
        self.classifier = nn.Sequential(
            quant_half(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Linear(512, num_classes),
            quant_half(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

def vgg16(quant):
    return VGG(quant, depth=16)

def vgg16bn(quant):
    return VGG(quant, depth=16, batch_norm=True)

def vgg19(quant):
    return VGG(quant, depth=19)

def vgg19bn(quant):
    return VGG(quant, depth=19, batch_norm=True)