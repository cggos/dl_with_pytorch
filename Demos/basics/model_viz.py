#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Ref
    * https://zhuanlan.zhihu.com/p/232348083
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

from Demos.dogs_vs_cats import network_model


class ModelTest(nn.Module):
    def __init__(self):
        super(ModelTest, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 10, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        return x


if __name__ == "__main__":
    model0 = ModelTest()
    model1 = network_model.Net()

    data_in = torch.rand(1, 3, 4, 4)
    out0 = model0(data_in)

    print(f'model0 out:\n {out0}')

    method = 0
    if method == 0:
        torch.save(model0, "modelviz.out.pt")  # netron
    elif method == 1:
        g = make_dot(out0)  # graphviz
        g.render('modelviz.out', view=True)
        # g.view()
