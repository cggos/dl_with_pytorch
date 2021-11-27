import os
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

from Demos.common import utils

if __name__ == "__main__":
    torch.manual_seed(1)

    path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../images/lena.bmp")
    img = Image.open(path_img).convert('RGB')  # 0~255

    img_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = img_transform(img)
    img_tensor.unsqueeze_(dim=0)  # C*H*W to B*C*H*W

    # pooling
    maxpool_layer = nn.MaxPool2d((5, 5), stride=(2, 2))  # input:(i, o, size) weights:(o, i , h, w)
    img_pool = maxpool_layer(img_tensor)

    print("池化前尺寸:{}\n池化后尺寸:{}".format(img_tensor.shape, img_pool.shape))

    img_raw = utils.transform_invert(img_tensor.squeeze(), img_transform)
    img_pool = utils.transform_invert(img_pool[0, 0:3, ...], img_transform)

    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(img_pool)
    plt.show()
