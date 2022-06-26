#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Ref
    * https://blog.csdn.net/u011922698/article/details/123268070
    * https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Detect_Persons_From_Image_YOLOv5.ipynb
"""

import base64
from io import BytesIO

import cv2
import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # channels=3, classes=10, force_reload=True

model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

# one img
img = 'https://ultralytics.com/images/zidane.jpg'

# or batch imgs
for f in ['zidane.jpg', 'bus.jpg']:
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)
img1 = Image.open('zidane.jpg')  # PIL image
img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]

# inference, includes NMS
results = model(imgs)  # img or imgs, custom inference size=320

# img predictions
box_tensor = results.xyxy[0]
box_pandas = results.pandas().xyxy[0]
# box_json = box_pandas.to_json(orient="record")

print(box_pandas)

results.print()
# results.render()  # updates results.imgs with boxes and labels
# for img in results.imgs:
#     buffered = BytesIO()
#     img_base64 = Image.fromarray(img)
#     img_base64.save(buffered, format="JPEG")
#     print(base64.b64encode(buffered.getvalue()).decode('utf-8'))  # base64 encoded image with results
results.show()
