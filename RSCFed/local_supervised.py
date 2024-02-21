import numpy as np
import torch
import torch.optim
import copy
import logging

from algorithms_standalone.basePS.client import Client
from model.SSFL_ResNet18 import ResNet18
from RSCFed.rscfed_dataset import *
from utils.set import *

class SupervisedLocalUpdate(Client):
    def __init__(self,client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, local_ds_pesudolabel_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, local_ds_pesudolabel_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num)
        self.args = args
        self.epoch = 0
        self.iter_num = 0
        # self.confuse_matrix = torch.zeros((5, 5)).cuda()
        self.base_lr = self.args.lr
        self.max_grad_norm = 5

        net = ResNet18(args=args, num_classes=self.args.model_output_dim, image_size=32,
                             model_input_channels=self.args.model_input_channels)

        self.model = net.to(self.device)

    def train(self,  net_w):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.to(self.device).train()

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.base_lr, momentum=0.9,
                                             weight_decay=5e-4)


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        loss_fn = torch.nn.CrossEntropyLoss()
        epoch_loss = AverageMeter()
        logging.info('Begin supervised training')
        for epoch in range(self.args.global_epochs_per_round):
            batch_loss = []
            for i, ( image_batch, _, label_batch) in enumerate(self.local_labeled_dl_PC):

                image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device).view(-1, )
                inputs = image_batch
                _, outputs = self.model(inputs)


                loss_classification = loss_fn(outputs, label_batch)
                loss = loss_classification
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()
                batch_loss.append(loss.item())
                self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            epoch_loss.update(np.array(batch_loss).mean(),1)

        self.model.cpu()
        return self.model.state_dict(), epoch_loss.avg, copy.deepcopy(
            self.optimizer.state_dict())
