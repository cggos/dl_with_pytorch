#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import zipfile
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import utils
import network_model


def main():
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s \n' % torch.cuda.is_available())

    data_root = '/tmp/dogs-vs-cats'

    train_zip = os.path.join(data_root, 'train.zip')
    if not os.path.exists(train_zip):
        os.system("mkdir -p {}".format(data_root))
        local_zip = os.path.join("/dev_sdb/", 'datasets/dogs-vs-cats.zip')
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall(data_root)
        zip_ref.close()

    print("Create validation data set")

    train_dir = os.path.join(data_root, 'train/')
    valid_dir = os.path.join(data_root, 'valid/')
    if os.path.exists(train_dir):
        os.system("rm -fr {}".format(train_dir))
    if os.path.exists(valid_dir):
        os.system("rm -fr {}".format(valid_dir))
    os.mkdir(valid_dir)

    zip_ref = zipfile.ZipFile(train_zip, 'r')
    zip_ref.extractall(data_root)
    zip_ref.close()

    for t in ['train', 'valid']:
        for folder in ['dog/', 'cat/']:
            os.mkdir(os.path.join(data_root, t, folder))

    files = glob(os.path.join(data_root, '*/*.jpg'))
    no_of_images = len(files)
    print(f'Total no of images {no_of_images}')

    shuffle = np.random.permutation(no_of_images)

    for i in shuffle[:2000]:
        # shutil.copyfile(files[i],'../chapter3/dogsandcats/valid/')
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(data_root, 'valid', folder, image))

    for i in shuffle[2000:]:
        # shutil.copyfile(files[i],'../chapter3/dogsandcats/valid/')
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(data_root, 'train', folder, image))

    print("Load data into PyTorch tensors\n")

    simple_transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train = datasets.ImageFolder('/tmp/dogs-vs-cats/train/', simple_transform)
    valid = datasets.ImageFolder('/tmp/dogs-vs-cats/valid/', simple_transform)

    print(train.classes)  # Category determined by the name of the division folder
    print(train.class_to_idx)  # The index is 0,1 according to the order.
    print(train.imgs[:5])  # Returns the path of the image obtained from all folders and their categories

    utils.imshow(train[50][0])
    plt.show()

    print("\nCreate data generators\n")

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=3, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=32, num_workers=3)
    dataset_sizes = {'train': len(train_data_loader.dataset), 'valid': len(valid_data_loader.dataset)}
    data_loaders = {'train': train_data_loader, 'valid': valid_data_loader}

    print("Train and valid network model\n")

    train01(dataset_sizes, data_loaders)

    # train02(train_data_loader, valid_data_loader)


def train01(dataset_sizes, data_loaders):
    model_ft = network_model.create_model()
    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    network_model.train_model(model_ft,
                              nn.CrossEntropyLoss(),
                              optimizer_ft,
                              exp_lr_scheduler,
                              dataset_sizes, data_loaders, num_epochs=5)


def train02(train_data_loader, valid_data_loader):
    model = network_model.Net()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    train_loss_list, train_accuracy_list = [], []
    valid_loss_list, valid_accuracy_list = [], []
    num_epochs = 10
    for epoch in range(1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss, train_accuracy = network_model.fit(model, optimizer, train_data_loader, phase='training')
        valid_loss, valid_accuracy = network_model.fit(model, optimizer, valid_data_loader, phase='validation')
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)
    plt.figure()
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, 'bo', label='training loss')
    plt.plot(range(1, len(valid_loss_list) + 1), valid_loss_list, 'r', label='validation loss')
    plt.legend()
    plt.figure()
    plt.plot(range(1, len(train_accuracy_list) + 1), train_accuracy_list, 'bo', label='train accuracy')
    plt.plot(range(1, len(valid_accuracy_list) + 1), valid_accuracy_list, 'r', label='val accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
