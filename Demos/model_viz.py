import torch
from torchviz import make_dot

import network_model

if __name__ == "__main__":
    model = network_model.ModelViz()

    out = model(torch.rand(1, 3, 4, 4))

    print(f'out:\n {out}')

    method = 0
    if method == 0:
        torch.save(network_model.Net(), "modelviz.pt")  # 生成一个pt文件，然后打开 netron 进行可视化
    elif method == 1:
        g = make_dot(out)  # 使用graphviz进行可视化
        # g.render('modelviz', view=True)
        g.view()
