#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import zipfile
from glob import glob
import numpy as np

if __name__ == "__main__":
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

    np.random.seed()

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
