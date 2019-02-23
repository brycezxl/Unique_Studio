from __future__ import division
from math import sqrt as sqrt
import torch


class PriorBox(object):
    """生成每个feature map的每个像素点上的4/6个default boxes"""
    def __init__(self):
        super(PriorBox, self).__init__()
        cfg = {
            'num_classes': 21,
            'lr_steps': (80000, 100000, 120000),
            'max_iter': 120000,
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_dim': 300,
            'steps': [8, 16, 32, 64, 100, 300],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'VOC',
        }
        self.image_size = cfg["min_dim"]
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        self.feature_maps = cfg["feature_maps"]

    def forward(self):
        box = []

        for k, f in enumerate(self.feature_maps):
            for i in range(f):     # 产生每一个点（i, j)
                for j in range(f):
                    f_k = self.image_size / self.steps[k]     # 当前feature map大小
                    cx = (j + 0.5) / f_k       # 列
                    cy = (i + 0.5) / f_k       # 行

                    # 第一个ratio为1的box
                    s_k = self.min_sizes[k] / self.image_size
                    box += [cx, cy, s_k, s_k]

                    # 第二个 sqrt(s_k * s_k+1)
                    s_k = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    box += [cx, cy, s_k, s_k]

                    # 其余
                    for ar in self.aspect_ratios[k]:
                        box += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                        box += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(box).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)  # 每个元素的夹紧到区间 [min,max] 内
        return output
