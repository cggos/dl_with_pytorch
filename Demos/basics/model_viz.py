import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

from Demos.dogs_vs_cats.dogs_vs_cats import network_model


class ModelViz(nn.Module):
    def __init__(self):
        super(ModelViz, self).__init__()
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
    model = network_model.ModelViz()

    out = model(torch.rand(1, 3, 4, 4))

    print(f'out:\n {out}')

    method = 0
    if method == 0:
        torch.save(network_model.Net(), "../modelviz.pt")  # 生成一个pt文件，然后打开 netron 进行可视化
    elif method == 1:
        g = make_dot(out)  # 使用graphviz进行可视化
        # g.render('modelviz', view=True)
        g.view()
