import copy
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from randaugment import *


data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'fmnist': ((0.2860,), (0.3530,)),
              'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687))}



def transform_pseudo_label(dataset):
    if dataset in ['cifar10','cifar100']:
        weak_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
        strong_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10, dataset=dataset),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
    if dataset in ['SVHN']:
        weak_trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
        strong_trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10,  dataset=dataset),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
    if dataset in ['fmnist']:
        weak_trans = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
        strong_trans = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10,  dataset=dataset),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
    return weak_trans, strong_trans



class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        input = {'data': input['data'], 'target': input['target']}
        return input

    def __len__(self):
        return self.size


def transform_pseudo_label_fedConsis(dataset):
    if dataset in ['cifar10','cifar100']:
        weak_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
        strong_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10, dataset=dataset),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
    if dataset in ['SVHN']:
        weak_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
        strong_trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10,  dataset=dataset),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
    if dataset in ['fmnist']:
        weak_trans = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
        strong_trans = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10,  dataset=dataset),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[dataset])
            ])
    return weak_trans, strong_trans

