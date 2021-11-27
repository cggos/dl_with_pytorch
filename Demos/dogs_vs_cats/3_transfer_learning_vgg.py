#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import gc
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms, datasets

from Demos.common import utils
import network_model


class LayerActivations():
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

    def remove(self):
        self.hook.remove()


class MyDataset:
    def __init__(self, feat, labels):
        self.conv_feat = feat
        self.labels = labels

    def __len__(self):
        return len(self.conv_feat)

    def __getitem__(self, idx):
        return self.conv_feat[idx], self.labels[idx]


def data_gen(conv_feat, labels, batch_size=64, shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(conv_feat))
        conv_feat = conv_feat[index]
        labels = labels[index]
    for idx in range(0, len(conv_feat), batch_size):
        yield conv_feat[idx:idx + batch_size], labels[idx:idx + batch_size]


def preconvfeat(dataset, model):
    conv_features = []
    labels_list = []
    for data in dataset:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)
        conv_features.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])

    return conv_features, labels_list


def cnn_imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


def main():
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.memory_summary(device=None, abbreviated=False)

    stats = torch.cuda.memory_stats(device=torch.cuda.current_device())

    print(f'stats:\n{stats}')
    print(f'max: {torch.cuda.max_memory_allocated(device=torch.cuda.current_device())}')

    print("Load data into PyTorch tensors\n")

    simple_transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train = datasets.ImageFolder('/tmp/dogs-vs-cats/train/', simple_transform)
    valid = datasets.ImageFolder('/tmp/dogs-vs-cats/valid/', simple_transform)

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=3, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=32, num_workers=3)

    vgg = models.vgg16(pretrained=True)
    vgg = vgg.cuda()

    # print(f'vgg:\n{vgg}')

    print("Freeze layers")

    vgg.classifier[6].out_features = 2
    for param in vgg.features.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 10):
        epoch_loss, epoch_accuracy = network_model.fit_vgg(vgg, optimizer, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = network_model.fit_vgg(vgg, optimizer, valid_data_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    print("Adjusting dropout")

    for layer in vgg.classifier.children():
        if type(layer) == nn.Dropout:
            layer.p = 0.2

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 3):
        epoch_loss, epoch_accuracy = network_model.fit_vgg(vgg, optimizer, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = network_model.fit_vgg(vgg, optimizer, valid_data_loader,
                                                                   phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    print("Data augmentation")

    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
    train = datasets.ImageFolder('/tmp/dogs-vs-cats/train/', train_transform)
    valid = datasets.ImageFolder('/tmp/dogs-vs-cats/valid/', simple_transform)

    train_data_loader = DataLoader(train, batch_size=32, num_workers=3, shuffle=True)
    valid_data_loader = DataLoader(valid, batch_size=32, num_workers=3, shuffle=True)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 3):
        epoch_loss, epoch_accuracy = network_model.fit_vgg(vgg, optimizer, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = network_model.fit_vgg(vgg, optimizer, valid_data_loader,
                                                                   phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    print("Calculating preconvoluted features")

    vgg = models.vgg16(pretrained=True)
    vgg = vgg.cuda()

    features = vgg.features

    for param in features.parameters(): param.requires_grad = False

    train_data_loader = DataLoader(train, batch_size=32, num_workers=3, shuffle=False)
    valid_data_loader = DataLoader(valid, batch_size=32, num_workers=3, shuffle=False)

    conv_feat_train, labels_train = preconvfeat(train_data_loader, features)
    conv_feat_val, labels_val = preconvfeat(valid_data_loader, features)

    train_feat_dataset = MyDataset(conv_feat_train, labels_train)
    val_feat_dataset = MyDataset(conv_feat_val, labels_val)

    train_feat_loader = DataLoader(train_feat_dataset, batch_size=64, shuffle=True)
    val_feat_loader = DataLoader(val_feat_dataset, batch_size=64, shuffle=True)

    optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 20):
        epoch_loss, epoch_accuracy = network_model.fit_numpy(vgg.classifier, optimizer, train_feat_loader,
                                                             phase='training')
        val_epoch_loss, val_epoch_accuracy = network_model.fit_numpy(vgg.classifier, optimizer, val_feat_loader,
                                                                     phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    print("Visualizing intermediate CNN layers")

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=3, shuffle=False)
    img, label = next(iter(train_data_loader))

    img_tmp = utils.transform_invert(img[5], simple_transform)
    plt.imshow(img_tmp)
    plt.show()

    img = img[5][None]

    vgg = models.vgg16(pretrained=True).cuda()

    conv_out = LayerActivations(vgg.features, 0)
    o = vgg(Variable(img.cuda()))
    conv_out.remove()
    act = conv_out.features

    fig = plt.figure(figsize=(20, 50))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(30):
        ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(act[0][i])
    plt.show()

    conv_out = LayerActivations(vgg.features, 1)
    o = vgg(Variable(img.cuda()))
    conv_out.remove()
    act = conv_out.features

    fig = plt.figure(figsize=(20, 50))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(30):
        ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(act[0][i])
    plt.show()

    conv_out = LayerActivations(vgg.features, 1)
    o = vgg(Variable(img.cuda()))
    conv_out.remove()
    act = conv_out.features

    fig = plt.figure(figsize=(20, 50))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(30):
        ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(act[0][i])
    plt.show()

    print("Visualizing weights")

    vgg = models.vgg16(pretrained=True).cuda()
    vgg.state_dict().keys()
    cnn_weights = vgg.state_dict()['features.0.weight'].cpu()

    fig = plt.figure(figsize=(30, 30))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(30):
        ax = fig.add_subplot(12, 6, i + 1, xticks=[], yticks=[])
        cnn_imshow(cnn_weights[i])
    plt.show()

    cnn_weights.shape


if __name__ == "__main__":
    main()
