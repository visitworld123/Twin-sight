import numpy as np
import torch
import torch.optim
import copy
import logging

from algorithms_standalone.basePS.client import Client
from model.SSFL_ResNet18 import ResNet18
from RSCFed.rscfed_dataset import *
from model.SVHN_model import SVHN_model
from FedIRM.confuse_matrix import *
from utils.log_info import log_info

class SupervisedLocalUpdate(Client):
    def __init__(self,client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num)
        self.args = args
        self.epoch = 0
        self.iter_num = 0
        self.base_lr = 0.01
        self.max_grad_norm = 5
        if self.args.model == 'SVHN_model':
            net = SVHN_model(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        else:
            net = ResNet18(args=args, num_classes=self.args.model_output_dim, image_size=32,
                             model_input_channels=self.args.model_input_channels)

        self.model = net.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=0.01, momentum=0.9,
                                         weight_decay=5e-4)
        self.confuse_matrix = torch.zeros((self.args.num_classes, self.args.num_classes)).to(self.device)
    #     self._local_su_dl()
    #
    # def _local_su_dl(self):
    #     dl, ds = get_dataloader(self.train_ori_data, self.train_ori_targets, self.args.dataset, self.args.batch_size, is_labeled=True)
    #     self.local_su_dl = dl


    def train(self, net_w):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.to(self.device).train()



        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.01

        loss_fn = torch.nn.CrossEntropyLoss()
        logging.info('Begin supervised training')

        # train and update
        epoch_loss = []
        logging.info('begin training')
        iter_max = len(self.local_labeled_dl_PC)
        batch_loss = []
        logging.info('\n=> FedIRM SU Training Epoch #%d, LR=%.4f' % (0, self.optimizer.param_groups[0]['lr']))
        for i, (image_batch, aug_image_batch, label_batch) in enumerate(self.local_labeled_dl_PC):
            image_batch, aug_image_batch, label_batch = image_batch.to(self.device), aug_image_batch.to(self.device), label_batch.to(self.device)
            ema_inputs = aug_image_batch
            batch_size = image_batch.size(0)
            inputs = image_batch
            _, outputs = self.model(inputs)
            _, aug_outputs = self.model(ema_inputs)
            # TODO
            labels = torch.zeros(batch_size, self.args.num_classes)
            for i, label in enumerate(label_batch):
                labels[i][label] = 1
            with torch.no_grad():
                self.confuse_matrix = self.confuse_matrix + get_confuse_matrix(outputs, labels, self.args.num_classes, self.device)
            loss_classification = loss_fn(outputs, label_batch.long()) + loss_fn(aug_outputs, label_batch.long())
            loss = loss_classification
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss.append(loss.item())
            self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        with torch.no_grad():
            self.confuse_matrix = self.confuse_matrix / (1.0 * self.args.global_epochs_per_round * iter_max)
        logging.info("client {} updated".format(self.client_index))
        log_info('scalar', 'FedIRM_{role}_{index}_train_loss'.format(role='client', index=self.client_index),
                 sum(epoch_loss) / len(epoch_loss), 0, self.args.record_tool)
        return  self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())

