import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10
import torch



class Generate_Pseduo_Label_Dataset(data.Dataset):

    def __init__(self, data, targets, weak_transform=None, strong_transform=None, target_transform=None):

        self.data = data
        self.targets = targets
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.weak_transform is not None and self.strong_transform is not None:
            weka_img = self.weak_transform(img)
            strong_img = self.strong_transform(img)
        else:
            raise RuntimeError("Transform Lacking in Pseduo Label Dataset Generating")
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return weka_img, strong_img, targets

    def __len__(self):
        return len(self.data)



class Pseduo_Label_Dataset(data.Dataset):

    def __init__(self, data, targets, target_transform=None):

        self.data = data
        self.targets = targets
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index],self.targets[index]

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)

