import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np
from layers import *


class SSD(nn.Module):
    def __init__(self, class_num):
        super(SSD, self).__init__()
        self.layers = nn.ModuleList(vgg() + add_extras())          # all layers
        self.loc, self.conf = nn.ModuleList(predictor(self.layers, class_num))
        self.predictor = [22, 34, 38, 42, 46, 50]   # 6 conv predictor
        self.priors = PriorBox()

    def forward(self, x):
        loc = []
        conf = []
        output = []

        for num, layer in enumerate(self.layers):
            x = layer(x)
            if num in self.predictor:
                if num == 22:
                    x = nn.LayerNorm(x)
                output.append(x)

        for o, l, c in zip(output, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W)
            # 这里的排列是将其改为 $(N, H, W, C)$
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 同理, 最终的shape为(两维):[batch, num_boxes*num_classes]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        outputs = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return outputs


def vgg():
    layers = []
    vgg16 = models.vgg16(pretrained=True).features
    for name, layer in vgg16._modules.items():
        if name == '30':               # 一直用到conv5_3
            break
        layers.append(layer)
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras():
    extra8_1 = nn.Conv2d(1024, 256, 1, 1, 0)
    extra8_2 = nn.Conv2d(256, 512, 3, 2, 1)
    extra9_1 = nn.Conv2d(512, 128, 1, 1, 0)
    extra9_2 = nn.Conv2d(128, 256, 3, 2, 1)
    extra10_1 = nn.Conv2d(256, 128, 1, 1, 0)
    extra10_2 = nn.Conv2d(128, 256, 3, 1, 0)
    extra11_1 = nn.Conv2d(256, 128, 1, 1, 0)
    extra11_2 = nn.Conv2d(128, 256, 3, 1, 0)
    return [extra8_1, nn.ReLU(inplace=True), extra8_2, nn.ReLU(inplace=True),
            extra9_1, nn.ReLU(inplace=True), extra9_2, nn.ReLU(inplace=True),
            extra10_1, nn.ReLU(inplace=True), extra10_2, nn.ReLU(inplace=True),
            extra11_1, nn.ReLU(inplace=True), extra11_2, nn.ReLU(inplace=True)]


def predictor(layers, class_num):
    loc_layers = []
    conf_layers = []
    predictor_info = {22: 4, 34: 6, 38: 6, 42: 6, 46: 4, 50: 4}    # predictor的层数与box数量

    for layer in predictor_info:
        # 定义6个坐标预测层, 输出的通道数就是每个像素点上会产生的 default box 的数量 * 4(坐标)
        loc_layers += [nn.Conv2d(layers[layer].out_channels, predictor_info[layer] * 4, 3, 1, 1)]
        # 定义分类层, 输出的通道数不一样对应每一个像素点上的每一个default box上每一个分类的分数
        conf_layers += [nn.Conv2d(layers[layer].out_channels, predictor_info[layer] * class_num, 3, 1, 1)]
    return loc_layers, conf_layers


def build_ssd():
    pass


a = SSD(class_num=21)
