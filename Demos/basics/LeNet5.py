#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : LeNet5.py
@Ref     : https://blog.csdn.net/defi_wang/article/details/107589456
@Author  : Gavin Gao
@Date    : 6/26/22 6:29 PM 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.C1 = nn.Conv2d(1, 6, 5)
        self.C3 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.F5 = nn.Linear(16 * 5 * 5, 120)
        self.F6 = nn.Linear(120, 84)
        self.OUTPUT = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.C1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.C3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.F5(x))
        x = F.relu(self.F6(x))
        x = self.OUTPUT(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = LeNet5()
print(net)
