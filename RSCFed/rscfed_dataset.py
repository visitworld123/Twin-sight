# encoding: utf-8
"""
Read images and corresponding labels.
"""
import torch
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
from transform_lib import *
N_CLASSES = 10



class CheXpertDataset(data.Dataset):

    def __init__(self, data, targets, is_labeled, transform=None, target_transform=None):

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.is_labeled = is_labeled


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.is_labeled == True:
            if self.transform is not None:
                image = self.transform(img)
                return image, torch.FloatTensor(targets)
        else:
            if self.transform is not None:
                img1 = self.transform(img)
                img2 = self.transform(img)

                return img1, img2, targets
    def __len__(self):
        return len(self.data)


# class CheXpertDataset(Dataset):
#     def __init__(self, dataset_type, data_np, label_np, pre_w, pre_h, lab_trans=None, un_trans_wk=None, data_idxs=None,
#                  is_labeled=False,
#                  is_testing=False):
#         """
#         Args:
#             data_dir: path to image directory.
#             csv_file: path to the file containing images
#                 with corresponding labels.
#             transform: optional transform to be applied on a sample.
#         """
#         super(CheXpertDataset, self).__init__()
#
#         self.images = data_np
#         self.labels = label_np
#         self.is_labeled = is_labeled
#         self.dataset_type = dataset_type
#         self.is_testing = is_testing
#
#         self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
#         if not is_testing:
#             if is_labeled == True:
#                 self.transform = lab_trans
#             else:
#                 self.data_idxs = data_idxs
#                 self.weak_trans = un_trans_wk
#         else:
#             self.transform = lab_trans
#
#         print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index: the index of item
#         Returns:
#             image and its labels
#         """
#         if self.dataset_type == 'skin':
#             img_path = self.images[index]
#             image = Image.open(img_path).convert('RGB')
#         else:
#
#
#         image_resized = self.resize(image)
#         label = self.labels[index]
#
#         if not self.is_testing:
#             if self.is_labeled == True:
#                 if self.transform is not None:
#                     image = self.transform(image_resized).squeeze()
#                     # image=image[:,:224,:224]
#                     return index, image, torch.FloatTensor([label])
#             else:
#                 if self.weak_trans and self.data_idxs is not None:
#                     weak_aug = self.weak_trans(image_resized)
#                     idx_in_all = self.data_idxs[index]
#
#                     for idx in range(len(weak_aug)):
#                         weak_aug[idx] = weak_aug[idx].squeeze()
#                     return index, weak_aug, torch.FloatTensor([label])
#         else:
#             image = self.transform(image_resized)
#             return index, image, torch.FloatTensor([label])
#             # return index, weak_aug, strong_aug, torch.FloatTensor([label])
#
#     def __len__(self):
#         return len(self.labels)
#
#
# class TransformTwice:
#     def __init__(self, transform):
#         self.transform = transform
#
#     def __call__(self, inp):
#         out1 = self.transform(inp)
#         out2 = self.transform(inp)
#         return [out1, out2]


def get_dataloader(data_np, label_np, dataset_type, train_bs, is_labeled=None,
                   is_testing=False):
    if dataset_type == 'fmnist':
        train_ori_transform = transforms.Compose([])
        train_ori_transform.transforms.append(transforms.Resize(32))
        train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_ori_transform.transforms.append(transforms.ToTensor())
        trans = train_ori_transform
    elif dataset_type == 'SVHN':
        trans = SimCLRTransform(32, False)
        normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                         std=[0.19803012, 0.20101562, 0.19703614])
        trans.train_transform.transforms.append(normalize)
    elif dataset_type == 'cifar100':
        trans = SimCLRTransform(32, False)
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        trans.train_transform.transforms.append(normalize)
    elif dataset_type == 'cifar10':
        trans = SimCLRTransform(32, False)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        trans.train_transform.transforms.append(normalize)

    if not is_testing:
        if is_labeled:
            ds = CheXpertDataset(data_np, label_np,
                                         is_labeled=True,
                                         transform=trans)
        else:

            ds = CheXpertDataset(data_np, label_np,
                                         is_labeled=False,
                                         transform=trans)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=0)
    else:
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        ds = CheXpertDataset( data_np, label_np,  is_labeled=True,transform=test_trans)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=False, num_workers=0)

    return dl, ds