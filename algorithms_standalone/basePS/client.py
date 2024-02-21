import copy
import logging
import math
import os
import sys
import numpy as np

import torch
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from algorithms.basePS.ps_client_trainer import PSTrainer


from data_preprocessing.cifar10.datasets import (
    Dataset_Personalize,
    Dataset_Personalize4BYOL,
    Dataset_Personalize4BYOL_FMNIST,
    Dataset_Personalize4PartialLebel,
    Dataset_Personalize4PartialLebel_FMNIST)

from data_aug import *
from pseduo_dataset import *

from transform_lib import *

from ssl_model.ssl_utils import *
class Client(PSTrainer):

    def __init__(self, client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, local_ds_pesudolabel_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets,  train_data_num,
                         device, args, model_trainer)

        # 生成RX、RXnoise、GX数据的模型

        # False means this client has no label data; True means the data of this client has label.
        self.label_flag = True

        self.train_ori_data = train_ori_data
        self.train_ori_targets = train_ori_targets
        self.train_cls_counts_dict = train_cls_counts_dict
        self.local_labeled_ds_PC = local_labeled_ds_PC
        self.local_unlabeled_ds_PC = local_unlabeled_ds_PC
        self.local_labeled_ds_PD = local_labeled_ds_PD
        self.local_unlabeled_ds_PD = local_unlabeled_ds_PD
        self.local_ds_pesudolabel_PD = local_ds_pesudolabel_PD
        self.dataset_num = dataset_num

        self.local_num_iterations = math.ceil(len(self.train_ori_data) / self.args.batch_size)
        self._construct_dataloader()
        self._local_data_weakTrans()
        # self._construct_train_ori_dataloader()
        # self._local_data_BYOLDataloader()
        
        # self._local_data_StrongTrans()
        # self._partial_labeled_dataset()
        # self._partial_labeled_dataset_split()

    def _construct_dataloader(self):
        self.local_labeled_dl_PC = torch.utils.data.DataLoader(dataset=self.local_labeled_ds_PC,
                                                                  batch_size=self.args.batch_size, shuffle=True,
                                                                  drop_last=False)
        self.local_unlabeled_dl_PC = torch.utils.data.DataLoader(dataset=self.local_unlabeled_ds_PC,
                                                                  batch_size=self.args.batch_size, shuffle=True,
                                                                  drop_last=False)
        self.local_labeled_dl_PD = torch.utils.data.DataLoader(dataset=self.local_labeled_ds_PD,
                                                                  batch_size=self.args.batch_size, shuffle=True,
                                                                  drop_last=False)
        self.local_unlabeled_dl_PD = torch.utils.data.DataLoader(dataset=self.local_unlabeled_ds_PD,
                                                                  batch_size=self.args.batch_size, shuffle=True,
                                                                  drop_last=False)
        self.local_dl_pesudolabel_PD = torch.utils.data.DataLoader(dataset=self.local_ds_pesudolabel_PD,
                                                                  batch_size=self.args.batch_size, shuffle=True,
                                                                  drop_last=False)
        
        

    def _construct_train_ori_dataloader(self):
        # ---------------------generate local train dataloader for Fed Step--------------------------#
        train_ori_transform = transforms.Compose([])
        if self.args.dataset == 'fmnist':
            train_ori_transform.transforms.append(transforms.Resize(32))
        train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['fmnist']:
            train_ori_transform.transforms.append(RandAugmentMC(n=2, m=10))
        train_ori_transform.transforms.append(transforms.ToTensor())
        train_ori_transform.transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
        print(self.train_ori_data.shape)
        train_ori_dataset = Dataset_Personalize(self.train_ori_data, self.train_ori_targets,
                                                transform=train_ori_transform)
        self.local_train_dataloader = torch.utils.data.DataLoader(dataset=train_ori_dataset,
                                                                  batch_size=32, shuffle=True,
                                                                  drop_last=False)
    def _local_data_weakTrans(self):
        weak_trans, _ = transform_pseudo_label(self.args.dataset)
        weak_dataset = Dataset_Personalize(self.train_ori_data, self.train_ori_targets,
                                           transform=weak_trans)

        self.local_dl_weak = torch.utils.data.DataLoader(dataset=weak_dataset,
                                                                batch_size=self.args.batch_size, shuffle=True,
                                                                drop_last=False)
    def prepare_embedding_by_SSFL_model(self, SSFL_model):
        X, y = inference_by_SSFL_model(SSFL_model, self.local_dataloader, self.device)
        self.embedding_trainLoader = create_data_loaders_from_arrays(X, y, 64)


    def _local_data_StrongTrans(self):
        _, strong_trans = transform_pseudo_label(self.args.dataset)
        strong_dataset = Dataset_Personalize(self.train_ori_data, self.train_ori_targets,
                                           transform=strong_trans)

        self.local_dataloader_strong = torch.utils.data.DataLoader(dataset=strong_dataset,
                                                                batch_size=self.args.batch_size, shuffle=True,
                                                                drop_last=False)

    def _local_data_BYOLDataloader(self):
        if self.args.dataset == 'fmnist':
            train_ori_transform = transforms.Compose([])
            train_ori_transform.transforms.append(transforms.Resize(32))
            train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_ori_transform.transforms.append(transforms.ToTensor())
            transform = train_ori_transform
            ssfl_local_dataset = Dataset_Personalize4BYOL_FMNIST(self.train_ori_data, self.train_ori_targets,
                                                          transform=transform)
        else:
            transform = SimCLRTransform(32, False)
            ssfl_local_dataset = Dataset_Personalize4BYOL(self.train_ori_data, self.train_ori_targets,
                                             transform=transform)

        self.ssfl_local_dataloader = torch.utils.data.DataLoader(dataset=ssfl_local_dataset,
                                                                   batch_size=self.args.batch_size, shuffle=True,
                                                                   drop_last=True)

    def generate_pseduo_label_dataset(self):
        weak_trans, strong_trans = transform_pseudo_label(self.args.dataset)
        generate_pseduo_dataset = Generate_Pseduo_Label_Dataset(self.train_ori_data, self.train_ori_targets,
                                           weak_transform=weak_trans, strong_transform=strong_trans)
        self.generate_pseduo_dataloader = torch.utils.data.DataLoader(dataset=generate_pseduo_dataset,
                                                                batch_size=self.args.batch_size, shuffle=True,
                                                                drop_last=False)



    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.client_timer.local_outer_epoch_idx
        iterations = self.client_timer.local_outer_iter_idx
        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.trainer.lr_schedule(epochs)


    def decompression(self,model_params):
        # TODO decopression algorithm on global distributed model

        return model_params

    def _partial_labeled_dataset(self):
        unlabel_idx = np.random.choice(range(len(self.train_ori_targets)), math.floor(len(self.train_ori_targets) * self.args.unlabeled_client_percentage), replace=False)
        self.train_ori_pseudo_targets = copy.deepcopy(self.train_ori_targets)
        for idx in unlabel_idx:
            self.train_ori_pseudo_targets[idx] = -1

        if self.args.dataset == 'fmnist':
            train_ori_transform = transforms.Compose([])
            train_ori_transform.transforms.append(transforms.Resize(32))
            train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_ori_transform.transforms.append(transforms.ToTensor())
            transform = train_ori_transform
            ssfl_local_dataset = Dataset_Personalize4PartialLebel_FMNIST(self.train_ori_data, self.train_ori_pseudo_targets,
                                                          self.train_ori_targets,transform=transform)
        else:
            transform = SimCLRTransform(32, False)
            ssfl_local_dataset = Dataset_Personalize4PartialLebel(self.train_ori_data, self.train_ori_pseudo_targets,
                                            self.train_ori_targets,transform=transform)

        self.fssl_dl_partial_labeled = torch.utils.data.DataLoader(dataset=ssfl_local_dataset,
                                                                   batch_size=self.args.batch_size, shuffle=True,
                                                                   drop_last=True)
    



    def _partial_labeled_dataset_split(self):
        unlabel_idx = np.random.choice(range(len(self.train_ori_targets)),
                                       math.floor(len(self.train_ori_targets) * self.args.partial_unlabel_percentage),
                                       replace=False)
        unlabel_data = copy.deepcopy(self.train_ori_data)
        unlabel_targets = copy.deepcopy(self.train_ori_targets)
        data_shape = self.train_ori_data.shape
        label_idx = np.setdiff1d(np.array(range(len(self.train_ori_targets))), unlabel_idx)
        if self.args.dataset == 'fmnist':
            label_data = np.empty((len(label_idx), data_shape[1], data_shape[2]))
        else:
            label_data = np.empty((len(label_idx), data_shape[1], data_shape[2], data_shape[3]))
        label_targets = np.empty((len(label_idx),))
        for num, idx in enumerate(label_idx):
            label_data[num] = self.train_ori_data[idx]
            label_targets[num] = self.train_ori_targets[idx]
        unlabel_data = np.delete(unlabel_data, label_idx, axis=0)
        unlabel_targets = np.delete(unlabel_targets, label_idx, axis=0)

        self.labeled_dl = self.ssfl_dl_generate(label_data, label_targets)
        self.unlabeled_dl = self.ssfl_dl_generate(unlabel_data, unlabel_targets)

    def ssfl_dl_generate(self, data, targets):
        if self.args.dataset == 'fmnist':
            train_ori_transform = transforms.Compose([])
            train_ori_transform.transforms.append(transforms.Resize(32))
            train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_ori_transform.transforms.append(transforms.ToTensor())
            transform = train_ori_transform
            ssfl_local_dataset = Dataset_Personalize4BYOL_FMNIST(data, targets,
                                                                 transform=transform)
        elif self.args.dataset == 'SVHN':
            train_ori_transform = transforms.Compose([])
            train_ori_transform.transforms.append(transforms.ToTensor())
            train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_ori_transform.transforms.append(
                transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)))
            transform = train_ori_transform
            ssfl_local_dataset = Dataset_Personalize4BYOL(data, targets,
                                                          transform=transform)
        else:
            transform = SimCLRTransform(32, True)
            ssfl_local_dataset = Dataset_Personalize4BYOL(data, targets,
                                                          transform=transform)

        ssfl_local_dataloader = torch.utils.data.DataLoader(dataset=ssfl_local_dataset,
                                                            batch_size=self.args.batch_size, shuffle=True,
                                                            drop_last=False)
        return ssfl_local_dataloader

    def _local_data_StrongTrans(self):

        _, strong_trans = transform_pseudo_label(self.args.dataset)

        strong_dataset = Dataset_Personalize(self.train_ori_data, self.train_ori_targets,
                                           transform=strong_trans)

        self.local_strong_dl = torch.utils.data.DataLoader(dataset=strong_dataset,
                                                                batch_size=self.args.batch_size, shuffle=True,
                                                                drop_last=False)

    # this method set the params to local trainer, local trainer usually more than one.
    def set_local_params(self, params):
        self.trainer.set_model_params(params)

    def run_train(self, round, downloaded_model_params, label_flag):
        decompression_model_params = self.decompression(downloaded_model_params)

        self.trainer.set_model_params(decompression_model_params)
        self.train_OURS_PC(round, label_flag)

        return self.upload()



    def train_OURS_PC(self, round, label_flag):
        if label_flag:
            logging.info("##########This Client has labeled data###########")
            for epoch in range(self.args.global_epochs_per_round):
            # self.trainer.train_classifier(round, epoch, self.local_train_dataloader, self.device)
                self.trainer.train_semiFed_model_labeled_client_PC(round, epoch, self.local_labeled_dl_PC, self.device)
        else:
            logging.info("##########This Client has no labeled data###########")
            for epoch in range(self.args.global_epochs_per_round):
            # self.trainer.train_classifier(round, epoch, self.local_train_dataloader, self.device)
                self.trainer.train_semiFed_model_unlabeled_client_PC(round, epoch, self.local_unlabeled_dl_PC, self.device)                                                           

    def run_train_PD(self, round, downloaded_model_params):
        decompression_model_params = self.decompression(downloaded_model_params)

        self.trainer.set_model_params(decompression_model_params)
        self.train_OURS_PD(round, global_params=copy.deepcopy(decompression_model_params))

        return self.upload()

        
    def train_OURS_PD(self, round, global_params):
        logging.info("########## We use OURS method in PD setting ###########")
        for epoch in range(self.args.global_epochs_per_round):
            # self.trainer.train_classifier(round, epoch, self.local_train_dataloader, self.device)
            self.trainer.train_semiFed_model_PD(round, epoch, global_params, 
                                     self.local_labeled_dl_PD, self.local_unlabeled_dl_PD,
                                     self.device)
    
    
    def compression(self):
        # TODO compression algorithm on local model
        if self.args.model == 'SemiFed_BYOL':
            compressed_model_params = copy.deepcopy(self.trainer.get_SemiFed_BYOL_params())
        else:
            compressed_model_params = copy.deepcopy(self.trainer.get_model_params())

        return compressed_model_params


    def upload(self):
        upload_info = {}
        upload_info['MODEL_PARAMS'] = self.compression()
        upload_info['SAMPLE_NUM'] = self.local_sample_number

        return upload_info







