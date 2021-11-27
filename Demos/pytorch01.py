#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

x = torch.Tensor(5, 3)  # 构造一个未初始化的5*3的矩阵
x = torch.rand(5, 3)  # 构造一个随机初始化的矩阵
y = torch.rand(5, 3)
x + y  # 语法一
torch.add(x, y)  # 语法二

# 任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
# 例如：x.copy_(y), x.t_(), 这俩都会改变x的值
result = torch.Tensor(5, 3)  # 语法一
torch.add(x, y, out=result)  # 语法二
y.add_(x)

# 此处演示当修改numpy数组之后,与之相关联的tensor也会相应的被修改
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print("a: {}".format(a))
print("b: {}".format(b))

# 将numpy的Array转换为torch的Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print("a: {}".format(a))
print("b: {}".format(b))

# 另外除了CharTensor之外，所有的tensor都可以在CPU运算和GPU预算之间相互转换
# 使用CUDA函数来将Tensor移动到GPU上，当CUDA可用时会进行GPU的运算
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = x + y
    print("z: {}".format(z))

ts = time.time()
a = torch.rand(10000, 10000)
b = torch.rand(10000, 10000)
a.matmul(b)
te = time.time()
print('time cost: ', te - ts, 's')

ts = time.time()
a = a.cuda()
b = b.cuda()
a.matmul(b)
te = time.time()
print('time cost: ', te - ts, 's')
