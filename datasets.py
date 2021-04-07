# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午11:06
# @Author  : red0orange
# @File    : datasets.py
# @Content : 数据加载
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import random


class Dataset(data.Dataset):
    def __init__(self, root, train=True, test=False, img_size=(224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 transforms=None):
        random.seed(521)
        self.img_size = img_size
        self.data = []
        if test:
            imgs_path = os.path.join(root, 'test')
        else:
            imgs_path = os.path.join(root, 'train')
        classes = [i for i in os.listdir(imgs_path) if i.isnumeric()]
        for cls_path, cls in [(os.path.join(imgs_path, i), int(i)) for i in classes]:
            self.data.extend([(os.path.join(cls_path, i), cls) for i in os.listdir(cls_path)])
        random.shuffle(self.data)
        # tmp_data = []
        # for img_path,label in self.data:
        #     img = Image.open(img_path)
        #     if len(np.array(img).shape) == 3 and np.array(img).shape[2] == 3:
        #         tmp_data.append((img_path,label))
        #     else:
        #         os.remove(img_path)
        # self.data = tmp_data
        imgs_len = len(self.data)
        if not test and train:
            self.data = self.data[:int(0.9 * imgs_len)]  # 训练集
        elif not test and not train:
            self.data = self.data[int(0.9 * imgs_len):]  # 验证集
        normalize = T.Normalize(mean=mean,
                                std=std)
        if transforms:
            self.transforms = transforms
        else:
            if test or not train:
                self.transforms = T.Compose([T.ToTensor(), normalize])
            else:
                self.transforms = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path)
        img = img.resize(self.img_size, Image.ANTIALIAS)
        img_tensor = self.transforms(img)
        return img_path, img_tensor, label
