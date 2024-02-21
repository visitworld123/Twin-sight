import logging
import copy
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10
import torch
import torchvision.transforms as transforms

from data_preprocessing.utils.utils import Cutout
from data_preprocessing.randaugment import RandAugment


class FSSL_Dataset(data.Dataset):

    def __init__(self, data, targets, ulb, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.ulb = ulb
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        self.strong_transform = copy.deepcopy(transform)
        self.strong_transform.transforms.insert(0, RandAugment(3, 5, self.dataset))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.array(img))
        if self.transform is not None:
            sample_transformed = self.transform(img)
            if self.dataset in ['fmnist','SVHN']:
                strong_sample_transformed = self.transform(img)
            else:
                strong_sample_transformed = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return ( sample_transformed, strong_sample_transformed, target) if not self.ulb else (
            sample_transformed, strong_sample_transformed)
    def __len__(self):
        return len(self.data)