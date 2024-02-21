import logging
import random
import math
import functools
import os
import copy
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    SVHN,
    FashionMNIST,
    MNIST,
)


from .cifar10.datasets import CIFAR10_truncated_WO_reload
from .cifar100.datasets import CIFAR100_truncated_WO_reload
from .SVHN.datasets import SVHN_truncated_WO_reload
from .FashionMNIST.datasets import FashionMNIST_truncated_WO_reload

from .cifar10.datasets import data_transforms_cifar10
from .cifar100.datasets import data_transforms_cifar100
from .SVHN.datasets import data_transforms_SVHN
from .FashionMNIST.datasets import data_transforms_fmnist


from data_preprocessing.utils.stats import record_net_data_stats

from data_preprocessing.FSSL_dataset import FSSL_Dataset

NORMAL_DATASET_LIST = ["cifar10", "cifar100", "SVHN",
                        "mnist", "fmnist"]



class Data_Loader(object):

    full_data_obj_dict = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "SVHN": SVHN,
        "fmnist": FashionMNIST,

    }
    sub_data_obj_dict = {
        "cifar10": CIFAR10_truncated_WO_reload,
        "cifar100": CIFAR100_truncated_WO_reload,
        "SVHN": SVHN_truncated_WO_reload,
        "fmnist": FashionMNIST_truncated_WO_reload,
    }

    transform_dict = {
        "cifar10": data_transforms_cifar10,
        "cifar100": data_transforms_cifar100,
        "SVHN": data_transforms_SVHN,
        "fmnist": data_transforms_fmnist,
    }

    num_classes_dict = {
        "cifar10": 10,
        "cifar100": 100,
        "SVHN": 10,
        "fmnist": 10,
    }


    image_resolution_dict = {
        "cifar10": 32,
        "cifar100": 32,
        "SVHN": 32,
        "fmnist": 32,
    }
    dataset_mean_dict = {
        'cifar10': [0.4914, 0.4822, 0.4465],
        'cifar100': [0.5071, 0.4865, 0.4409],
        'SVHN': [0.4377, 0.4438, 0.4728],
        'fmnist': [0.5,]

    }

    dataset_std_dict = {
        'cifar10': [0.2470, 0.2435, 0.2616],
        'cifar100': [0.2673, 0.2564, 0.2762],
        'SVHN': [0.1980, 0.2010, 0.1970],
        'fmnist':[0.5,]
    }

    def __init__(self, args=None, process_id=0, mode="centralized", task="centralized",
                data_efficient_load=True, dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="hetero", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default", other_params={}):

        # less use this.
        self.args = args

        # For partition
        self.process_id = process_id
        self.mode = mode
        self.task = task
        self.data_efficient_load = data_efficient_load # Loading mode, for standalone usage.
        self.dirichlet_balance = dirichlet_balance
        self.dirichlet_min_p = dirichlet_min_p

        self.dataset = dataset
        self.datadir = datadir
        self.partition_method = partition_method
        self.partition_alpha = partition_alpha
        self.client_number = client_number
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_sampler = data_sampler

        self.augmentation = augmentation
        self.other_params = other_params

        # For image
        self.resize = resize

        self.init_dataset_obj()




    def load_data(self):
        self.federated_standalone_split() # 主要是联邦单机,本code中的主要加载数据方式
        self.other_params["train_cls_local_counts_dict"] = self.train_cls_local_counts_dict
        self.other_params["client_dataidx_map"] = self.client_dataidx_map
        self.other_params["labeled_client_idx"] = self.labeled_client_idx
        self.other_params["unlabeled_client_idx"] = self.unlabeled_client_idx
        self.other_params["local_labeled_ds_PC_dict"] = self.local_labeled_ds_PC_dict
        self.other_params["local_unlabeled_ds_PC_dict"] = self.local_unlabeled_ds_PC_dict
        self.other_params["local_labeled_ds_PD_dict"] = self.local_labeled_ds_PD_dict
        self.other_params["local_unlabeled_ds_PD_dict"] = self.local_unlabeled_ds_PD_dict
        self.other_params["local_ds_pesudolabel_PD_dict"] = self.local_ds_pesudolabel_PD_dict
        
    
        return self.train_data_global_num, self.test_data_global_num, self.train_data_global_dl, self.test_data_global_dl, \
               self.train_data_local_num_dict, self.train_data_local_ori_dict, self.train_targets_local_ori_dict,\
               self.class_num, self.other_params

    def partial_labeled_data(self, label_ratio, client_data_idx_map):
        labeled_data_idx_map = {}
        unlabled_data_idx_map = {}
        for client_idx in client_data_idx_map.keys():
            labeled_data_num = math.ceil( len(client_data_idx_map[client_idx]) * label_ratio)
            labeled_data_idx = np.random.choice(client_data_idx_map[client_idx], labeled_data_num, replace=False)
            unlabel_data_idx = np.setdiff1d(client_data_idx_map[client_idx], labeled_data_idx)

            labeled_data_idx_map[client_idx] = labeled_data_idx
            unlabled_data_idx_map[client_idx] = unlabel_data_idx

        return labeled_data_idx_map, unlabled_data_idx_map
# got it获得数据集名称，transform的函数名 ，class_num
    def init_dataset_obj(self):
        self.full_data_obj = Data_Loader.full_data_obj_dict[self.dataset]
        self.sub_data_obj = Data_Loader.sub_data_obj_dict[self.dataset]
        logging.info(f"dataset augmentation: {self.augmentation}, resize: {self.resize}")
        self.transform_func = Data_Loader.transform_dict[self.dataset]  # 生成transform的function
        self.class_num = Data_Loader.num_classes_dict[self.dataset]
        self.image_resolution = Data_Loader.image_resolution_dict[self.dataset]


# got it 获得 train_transform 和test_transform
    def get_transform(self, resize, augmentation, dataset_type, image_resolution=32):
        MEAN, STD, train_transform, test_transform = \
            self.transform_func(
                resize=resize, augmentation=augmentation, dataset_type=dataset_type, image_resolution=image_resolution)
        # if self.args.Contrastive == "SimCLR":
        return MEAN, STD, train_transform, test_transform




# got it 加载完整数据集
    def load_full_data(self):
        # For cifar10, cifar100, SVHN, FMNIST
        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "full_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        if self.dataset == "SVHN":
            train_ds = self.full_data_obj(self.datadir,  "train", download=True, transform=train_transform, target_transform=None)
            test_ds = self.full_data_obj(self.datadir,  "test", download=True, transform=test_transform, target_transform=None)
            train_ds.data = train_ds.data.transpose((0,2,3,1))
            # test_ds.data =  test_ds.data.transpose((0,2,3,1))
            logging.info(os.getcwd())
        else:
            train_ds = self.full_data_obj(self.datadir,  train=True, download=True, transform=train_transform)
            test_ds = self.full_data_obj(self.datadir,  train=False, download=True, transform=test_transform)
            logging.info(os.getcwd())
        # X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
        # X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets
        
        return train_ds, test_ds  #Complete Dataset

# got it 加载不同子数据集
    def load_sub_data(self, client_index, train_ds, test_ds):
        '''
        clinet_index：client的index，在client_dataidx_map中根据不同index取不同partition的数据集
        train_ds，test_ds：完整的TrainSet和TestSet
        '''
        # Maybe only ``federated`` needs this.
        dataidxs = self.client_dataidx_map[client_index]
        train_data_local_num = len(dataidxs)
        labeled_idx = self.labeled_data_idx_map[client_index]
        unlabeled_idx = self.unlabled_data_idx_map[client_index]

        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "sub_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        train_ds_local = self.sub_data_obj(self.datadir, dataidxs=dataidxs, train=True, transform=train_transform,
                full_dataset=train_ds)
        label_train_ds_local = self.sub_data_obj(self.datadir, dataidxs=labeled_idx, train=True, transform=train_transform,
                full_dataset=train_ds)
        unlabel_train_ds_local = self.sub_data_obj(self.datadir, dataidxs=unlabeled_idx, train=True, transform=train_transform,
                full_dataset=train_ds)
        fssl_transform = self.get_local_stransform(train=True)
        local_labeled_ds_PC = FSSL_Dataset(data=train_ds_local.data, 
                                           targets=train_ds_local.targets, 
                                           ulb=False, dataset=self.args.dataset, transform=fssl_transform)
        local_unlabeled_ds_PC = FSSL_Dataset(data=train_ds_local.data, 
                                             targets=train_ds_local.targets, 
                                             ulb=True, dataset=self.args.dataset, transform=fssl_transform)
        local_labeled_ds_PD = FSSL_Dataset(data=label_train_ds_local.data, 
                                           targets=label_train_ds_local.targets, 
                                           ulb=False,dataset=self.args.dataset,  transform=fssl_transform)
        local_unlabeled_ds_PD = FSSL_Dataset(data=unlabel_train_ds_local.data, 
                                             targets=unlabel_train_ds_local.targets, 
                                             ulb=True, dataset=self.args.dataset, transform=fssl_transform)
        
        # get the original data without transforms, so it's in [0, 255] np array rather than Tensor
        train_ori_data = np.array(train_ds_local.data)
        train_ori_targets = np.array(train_ds_local.targets)
        local_pesudolabel_data = np.concatenate((label_train_ds_local.data, unlabel_train_ds_local.data), axis=0)
        local_pesudolabel_targets = np.concatenate((label_train_ds_local.targets, -1 * np.ones_like(unlabel_train_ds_local.targets)), axis=0)
        local_ds_pesudolabel_PD = FSSL_Dataset(data=local_pesudolabel_data, 
                                              targets=local_pesudolabel_targets, 
                                             ulb=False, dataset=self.args.dataset, transform=fssl_transform)


        return train_ds_local, train_data_local_num, train_ori_data, train_ori_targets,\
        local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD, local_unlabeled_ds_PD, local_ds_pesudolabel_PD

    def get_dataloader(self, train_ds, test_ds,shuffle=True, drop_last=False, train_sampler=None, num_workers=1):
        logging.info(f"shuffle: {shuffle}, drop_last:{drop_last}, train_sampler:{train_sampler} ")
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=shuffle,   # dl means dataloader
                                drop_last=drop_last, sampler=train_sampler, num_workers=num_workers) # sampler定义自己的sampler策略，如果指定这个参数，则shuffle必须为False
        if test_ds is not None:
            test_dl = data.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=True,
                                drop_last=False, num_workers=num_workers)  # drop_last为True剩余的数据不够一个batch会扔掉
            return train_dl, test_dl
        else:
            return train_dl

# got it 将label设为np
    def get_y_train_np(self, train_ds):
        if self.dataset in ["fmnist"]:
            y_train = train_ds.targets.data
        elif self.dataset in ["SVHN"]:
            y_train = train_ds.labels
        else:
            y_train = train_ds.targets
        y_train_np = np.array(y_train)
        return y_train_np

    def partial_labeled_client(self):
        unlabeled_client_num = math.ceil(self.args.unlabeled_client_percentage * self.args.client_num_in_total)
        self.unlabeled_client_idx = np.random.choice(range(self.args.client_num_in_total), unlabeled_client_num,
                                                         replace=False)
        if self.args.client_num_in_total == 10:
            if self.args.unlabeled_client_percentage==0.6:  
                if self.args.dataset == 'cifar10':
                    self.unlabeled_client_idx = [0, 7, 5, 4, 9, 2]
                elif self.args.dataset == 'cifar100':
                    self.unlabeled_client_idx = [2, 6, 9, 3, 4, 5]
                elif self.args.dataset == 'fmnist':
                    self.unlabeled_client_idx = [4, 8, 6, 0, 7, 2]
                elif self.args.dataset == 'SVHN':
                    self.unlabeled_client_idx = [8, 6, 7, 5, 3, 0]
            elif self.args.unlabeled_client_percentage==0.7: 
                if self.args.dataset == 'cifar10':
                    self.unlabeled_client_idx = [0, 7, 5, 4, 9, 2, 1]
                elif self.args.dataset == 'cifar100':
                    self.unlabeled_client_idx = [2, 6, 9, 3, 4, 5, 1]
                elif self.args.dataset == 'fmnist':
                    self.unlabeled_client_idx = [4, 8, 6, 0, 7, 2, 3]
                elif self.args.dataset == 'SVHN':
                    self.unlabeled_client_idx = [8, 6, 7, 5, 3, 0, 2]
            elif self.args.unlabeled_client_percentage==0.8: 
                if self.args.dataset == 'cifar10':
                    self.unlabeled_client_idx = [0, 7, 5, 4, 9, 2, 1,3]
                elif self.args.dataset == 'cifar100':
                    self.unlabeled_client_idx = [2, 6, 9, 3, 4, 5, 1, 0]
                elif self.args.dataset == 'fmnist':
                    self.unlabeled_client_idx = [4, 8, 6, 0, 7, 2, 3, 1]
                elif self.args.dataset == 'SVHN':
                    self.unlabeled_client_idx = [8, 6, 7, 5, 3, 0, 2, 1]
            elif self.args.unlabeled_client_percentage==0.9:  
                if self.args.dataset == 'cifar10':
                    self.unlabeled_client_idx = [0, 7, 5, 4, 9, 2, 1,3,6]
                elif self.args.dataset == 'cifar100':
                    self.unlabeled_client_idx = [2, 6, 9, 3, 4, 5, 1, 0,8]
                elif self.args.dataset == 'fmnist':
                    self.unlabeled_client_idx = [4, 8, 6, 0, 7, 2, 3, 1,9]
                elif self.args.dataset == 'SVHN':
                    self.unlabeled_client_idx = [8, 6, 7, 5, 3, 0, 2, 1,4]
            elif self.args.unlabeled_client_percentage==0.5:  
                if self.args.dataset == 'cifar10':
                    self.unlabeled_client_idx = [0, 7, 5, 4, 9]
                elif self.args.dataset == 'cifar100':
                    self.unlabeled_client_idx = [2, 6, 9, 3, 4]
                elif self.args.dataset == 'fmnist':
                    self.unlabeled_client_idx = [4, 8, 6, 0, 7]
                elif self.args.dataset == 'SVHN':
                    self.unlabeled_client_idx = [8, 6, 7, 5, 3]
            elif self.args.unlabeled_client_percentage==0.4:  
                if self.args.dataset == 'cifar10':
                    self.unlabeled_client_idx = [0, 7,  4, 9]
                elif self.args.dataset == 'cifar100':
                    self.unlabeled_client_idx = [2, 6, 3, 4]
                elif self.args.dataset == 'fmnist':
                    self.unlabeled_client_idx = [4, 8, 0, 7]
                elif self.args.dataset == 'SVHN':
                    self.unlabeled_client_idx = [8, 6, 5, 3]   
        elif self.args.client_num_in_total == 50 and self.args.unlabeled_client_percentage==0.6:
            self.unlabeled_client_idx = [26, 37,  9, 36, 31, 18, 46, 39,  4, 12, 25, 28,  5,  6, 24,  3, 21,\
                                            29, 45, 40, 48, 49, 16, 38,  2, 32,  8, 47, 17, 22]
        elif self.args.client_num_in_total == 100 and self.args.unlabeled_client_percentage==0.6:
            self.unlabeled_client_idx = [31, 46, 22, 24, 17, 35, 11, 88, 56, 97, 99, 66, 27, 53, 98, 51, 77,\
                                        40, 47, 13, 73, 63, 90,  4, 23, 67, 25, 43, 83, 74, 81, 16,  1, 61,\
                                        34, 33, 38, 50, 72, 85, 87, 30, 21,  5,  2,  0, 15, 57, 41, 42, 36,\
                                        92, 62, 54, 44, 32, 82, 58, 39, 6]
                
        self.labeled_client_idx = np.setdiff1d(range(self.args.client_num_in_total), self.unlabeled_client_idx)
        logging.info("labeled_client_idx = " + str(self.labeled_client_idx))
        logging.info("unlabeled_client_idx = " + str(self.unlabeled_client_idx))


    def federated_standalone_split(self):
        # For cifar10, cifar100, SVHN, FMNIST
        train_ds, test_ds = self.load_full_data()
        y_train_np = self.get_y_train_np(train_ds)  # 把label转换成np类型

        self.train_data_global_num = y_train_np.shape[0]
        self.test_data_global_num = len(test_ds)  # 以数据量来计算，而不是DataLoader的迭代次数

        self.client_dataidx_map, self.train_cls_local_counts_dict = self.partition_data(y_train_np, self.train_data_global_num)

        
        self.labeled_data_idx_map, self.unlabled_data_idx_map = self.partial_labeled_data(self.args.labeled_data_percentage,
                                                                            self.client_dataidx_map)
        
        save_info = {"client_dataidx_map": self.client_dataidx_map,
                     "labeled_data_idx_map": self.labeled_data_idx_map,
                     "unlabled_data_idx_map":self.unlabled_data_idx_map}
        # with open('{}_client_{}_dataset_{}_alpha_{}_seed.pkl'.format(self.args.client_num_in_total, 
        #                                                              self.args.dataset, self.args.partition_alpha,
        #                                                              self.args.seed), 'wb') as file:
        #  # 使用pickle模块将字典保存到文件中
        #     pickle.dump(save_info, file)

        logging.info("train_cls_local_counts_dict = " + str(self.train_cls_local_counts_dict))
        
        self.partial_labeled_client()



        self.train_data_global_dl, self.test_data_global_dl = self.get_dataloader(   # train_data_global_dataloader and test_data_global_dataloader
                train_ds, test_ds,  
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)



        self.train_data_local_num_dict = dict()  # 记录不同client的数据数量
        self.test_data_local_num_dict = dict()
        self.train_data_local_ori_dict = dict()
        self.train_targets_local_ori_dict = dict()
        self.test_data_local_dl_dict = dict()
        self.local_labeled_ds_PC_dict = dict()
        self.local_unlabeled_ds_PC_dict = dict()
        self.local_labeled_ds_PD_dict = dict()
        self.local_unlabeled_ds_PD_dict = dict()
        self.local_ds_pesudolabel_PD_dict = dict()

        for client_index in range(self.client_number):
            # 在这个里面主要拿出原始train数据和label，因为后面不同的算法需要对train data进行不同的transform
            # 如果直接传递dataset，会导致在分配数据的时候就提前建好许多的transform
            train_ds_local, train_data_local_num, train_ori_data, train_ori_targets,\
            local_labeled_ds_PC, local_unlabeled_ds_PC, \
            local_labeled_ds_PD, local_unlabeled_ds_PD,local_ds_pesudolabel_PD = \
            self.load_sub_data(client_index, train_ds, test_ds)

            self.local_labeled_ds_PC_dict[client_index] = local_labeled_ds_PC
            self.local_unlabeled_ds_PC_dict[client_index] = local_unlabeled_ds_PC
            self.local_labeled_ds_PD_dict[client_index] = local_labeled_ds_PD
            self.local_unlabeled_ds_PD_dict[client_index] = local_unlabeled_ds_PD
            self.train_data_local_num_dict[client_index] = train_data_local_num
            self.local_ds_pesudolabel_PD_dict[client_index] = local_ds_pesudolabel_PD
            logging.info("client_ID = %d, local_train_sample_number = %d" % \
                         (client_index, train_data_local_num))

            #train_sampler = self.get_train_sampler(train_ds_local, rank=client_index, distributed=False)
            #shuffle = train_sampler is None  # 非空则shuffle得设为False

            # training batch size = 64; algorithms batch size = 32
            train_data_local_dl = self.get_dataloader(train_ds_local, test_ds=None,
                                                                          shuffle=True, drop_last=False, num_workers=self.num_workers)
            logging.info("client_index = %d, batch_num_train_local = %d" % (
                client_index, len(train_data_local_dl))) # 每个local client有多少batch的数据

            self.train_data_local_ori_dict[client_index] = train_ori_data
            self.train_targets_local_ori_dict[client_index] = train_ori_targets


    # centralized loading
    def load_centralized_data(self):
        self.train_ds, self.test_ds = self.load_full_data()
        self.train_data_num = len(self.train_ds)
        self.test_data_num = len(self.test_ds)
        self.train_dl, self.test_dl = self.get_dataloader(
                self.train_ds, self.test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)


# got  it 将trainset按不同策略分开
    def partition_data(self, y_train_np, train_data_num):
        logging.info("partition_method = " + (self.partition_method))
        if self.partition_method in ["homo", "iid"]:
            total_num = train_data_num
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.client_number)
            client_dataidx_map = {i: batch_idxs[i] for i in range(self.client_number)}

        # Dirichlet分布函数实现non-iid
        elif self.partition_method == "hetero":
            min_size = 0
            K = self.class_num    # 类别数
            N = y_train_np.shape[0]  # 训练数据总数
            logging.info("N = " + str(N))
            client_dataidx_map = {}  # idx_map 记录client idx拥有的数据

            while min_size < self.class_num:
                idx_batch = [[] for _ in range(self.client_number)]
                # for each class in the dataset
                for k in range(K):  # 每个类别
                    idx_k = np.where(y_train_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.partition_alpha, self.client_number))
                    if self.dirichlet_balance:
                        argsort_proportions = np.argsort(proportions, axis=0)
                        if k != 0:
                            used_p = np.array([len(idx_j) for idx_j in idx_batch])
                            argsort_used_p = np.argsort(used_p, axis=0)
                            inv_argsort_proportions = argsort_proportions[::-1]
                            proportions[argsort_used_p] = proportions[inv_argsort_proportions]
                    else:
                        proportions = np.array([p * (len(idx_j) < N / self.client_number) for p, idx_j in zip(proportions, idx_batch)])

                    ## set a min value to smooth, avoid too much zero samples of some classes.
                    if self.dirichlet_min_p is not None:
                        proportions += float(self.dirichlet_min_p)
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                client_dataidx_map[j] = idx_batch[j]

        elif self.partition_method > "noniid-#label0" and self.partition_method <= "noniid-#label9":
            num = eval(self.partition_method[13:])
            if self.dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
                num = 1
                K = 2
            else:
                K = self.class_num
            if num == 10:
                client_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(self.client_number)}
                for i in range(10):
                    idx_k = np.where(y_train_np==i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.client_number)
                    for j in range(self.client_number):
                        client_dataidx_map[j]=np.append(client_dataidx_map[j],split[j])
            else:
                times=[0 for i in range(10)]
                contain=[]
                for i in range(self.client_number):
                    current=[i%K]
                    times[i%K]+=1
                    j=1
                    while (j<num):
                        ind=random.randint(0,K-1)
                        if (ind not in current):
                            j=j+1
                            current.append(ind)
                            times[ind]+=1
                    contain.append(current)
                client_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(self.client_number)}
                for i in range(K):
                    idx_k = np.where(y_train_np==i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k,times[i])
                    ids=0
                    for j in range(self.client_number):
                        if i in contain[j]:
                            client_dataidx_map[j]=np.append(client_dataidx_map[j],split[ids])
                            ids+=1

        elif self.partition_method == "long-tail":
            if self.client_number == 10 or self.client_number == 100:
                pass
            else:
                raise NotImplementedError

            # There are  self.client_number // self.class_num clients share the \alpha proportion of data of one class
            main_prop = self.partition_alpha / (self.client_number // self.class_num)

            # There are (self.client_number - self.client_number // self.class_num) clients share the tail of one class
            tail_prop = (1 - main_prop) / (self.client_number - self.client_number // self.class_num)

            client_dataidx_map = {}
            # for each class in the dataset
            K = self.class_num
            idx_batch = [[] for _ in range(self.client_number)]
            for k in range(K):
                idx_k = np.where(y_train_np == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.array([tail_prop for _ in range(self.client_number)])
                main_clients = np.array([k + i * K for i in range(self.client_number // K)])
                proportions[main_clients] = main_prop
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                client_dataidx_map[j] = idx_batch[j]

            
        train_cls_local_counts_dict = record_net_data_stats(y_train_np, client_dataidx_map)

        return client_dataidx_map, train_cls_local_counts_dict



    def get_local_stransform(self, train):

        color_jitter = transforms.ColorJitter(
            0.8 , 0.8, 0.8, 0.2
        )
        if train:
            if self.dataset in ['fmnist']:
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            elif self.dataset in ['SVHN']:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=32),
                    transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor()])
            else:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(Data_Loader.dataset_mean_dict[self.dataset], Data_Loader.dataset_std_dict[self.dataset])])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(Data_Loader.dataset_mean_dict[self.dataset], Data_Loader.dataset_std_dict[self.dataset])])
        return transform











