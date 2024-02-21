from algorithms_standalone.basePS.basePSmanager import BasePSManager
import logging
import copy
import random
from RSCFed.rscfed_dataset import  *

from FedIRM.local_supervised import SupervisedLocalUpdate
from FedIRM.local_unsupervised import UnsupervisedLocalUpdate
from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer
import math
import numpy as np
from model.SSFL_ResNet18 import ResNet18
from algorithms_standalone.fedavg.aggregator import FedAVGAggregator
from utils.tool import *
from FedIRM.FedAvg import *
from model.SVHN_model import SVHN_model
from utils.set import *

class FedIRMManager(BasePSManager):
    def __init__(self, device, args):
        super().__init__(device, args)

        self.lab_trainer_locals = {}
        self.unlab_trainer_locals = {}

        logging.info("Unlabeled Client : %s" % (str(self.unlabeled_client_idx)))

        self.net_glob = ResNet18(args=self.args, num_classes=self.args.model_output_dim, image_size=32,
                                     model_input_channels=self.args.model_input_channels)

        self.net_glob.train()
        self.w_glob = self.net_glob.state_dict()

        self.w_locals = {} # weight
        self.w_ema_unsup = {}
        self.sup_net_locals = {}  # model
        self.unsup_net_locals = {}
        # self._load_existing_SSFL_model()
        self._setup_clients()
        # ================================================
        self._setup_server()

    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                             device=self.device, **self.other_params)
        init_state_kargs = {}  # self.get_init_state_kargs()   # 1.设置selected_clients 为所有client  2.FedAvg中init_state_kargs={}

        model_trainer = create_trainer(  # 设置
            self.args, self.device, model, train_data_global_num=self.train_data_global_num,
            test_data_global_num=self.test_data_global_num, train_data_global_dl=self.train_data_global_dl,
            test_data_global_dl=self.test_data_global_dl, train_data_local_num_dict=self.train_data_local_num_dict,
            class_num=self.class_num, server_index=0, role='server', **init_state_kargs
        )

        # model_trainer = create_trainer(self.args, self.device, model)
        self.aggregator = FedAVGAggregator(self.train_data_global_dl, self.test_data_global_dl, self.train_data_global_num,
                self.test_data_global_num, self.train_data_local_num_dict, self.args.client_num_in_total, self.device,
                self.args, model_trainer)

        # self.aggregator.traindata_cls_counts = self.traindata_cls_counts
        logging.info("############_setup_server (END)#############")

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        init_state_kargs = self.get_init_state_kargs()  # 最开始将sample_client设置为所有clinet [0,1,2,...,client_num]
        # for client_index in range(self.args.client_num_in_total):
        for client_index in range(self.number_instantiated_client):

            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                                    device=self.device, **self.other_params)

            model_trainer = create_trainer(self.args, self.device, model, class_num=self.class_num,
                                            other_params=self.other_params, client_index=client_index, role='client')
            if client_index in self.labeled_client_idx:

                client = SupervisedLocalUpdate(client_index, train_ori_data=self.train_data_local_ori_dict[client_index],
                             train_ori_targets=self.train_targets_local_ori_dict[client_index],
                             train_data_num=self.train_data_local_num_dict[client_index],
                             local_labeled_ds_PC = self.local_labeled_ds_PC_dict[client_index], 
                             local_unlabeled_ds_PC = self.local_unlabeled_ds_PC_dict[client_index], 
                             local_labeled_ds_PD = self.local_labeled_ds_PD_dict[client_index],
                             local_unlabeled_ds_PD = self.local_unlabeled_ds_PD_dict[client_index],
                             local_ds_pesudolabel_PD=self.local_ds_pesudolabel_PD_dict[client_index],
                             train_cls_counts_dict = self.train_cls_local_counts_dict[client_index],
                             device=self.device, args=self.args, model_trainer=model_trainer,
                             dataset_num=self.train_data_global_num)
                self.lab_trainer_locals[client_index] = client
                self.w_locals[client_index] = copy.deepcopy(self.w_glob)
                self.sup_net_locals[client_index] = copy.deepcopy(self.net_glob)
            elif client_index in self.unlabeled_client_idx:
                client = UnsupervisedLocalUpdate(client_index, train_ori_data=self.train_data_local_ori_dict[client_index],
                             train_ori_targets=self.train_targets_local_ori_dict[client_index],
                             train_data_num=self.train_data_local_num_dict[client_index],
                             local_labeled_ds_PC = self.local_labeled_ds_PC_dict[client_index], 
                             local_unlabeled_ds_PC = self.local_unlabeled_ds_PC_dict[client_index], 
                             local_labeled_ds_PD = self.local_labeled_ds_PD_dict[client_index],
                             local_unlabeled_ds_PD = self.local_unlabeled_ds_PD_dict[client_index],
                             local_ds_pesudolabel_PD=self.local_ds_pesudolabel_PD_dict[client_index],
                             train_cls_counts_dict = self.train_cls_local_counts_dict[client_index],
                             device=self.device, args=self.args, model_trainer=model_trainer,
                             dataset_num=self.train_data_global_num)
                self.unlab_trainer_locals[client_index] = client
                self.w_locals[client_index] = copy.deepcopy(self.w_glob)
                self.w_ema_unsup[client_index] = copy.deepcopy(self.w_glob)
                self.unsup_net_locals[client_index] = copy.deepcopy(self.net_glob)

                # client.prepare_embedding_by_SSFL_model(self.embedding_model)
            self.client_list.append(client)

        # choose client under certain percentage to preserver label to standalone a new SSFL scenario



        logging.info("############setup_clients (END)#############")
    def train(self):
        for com_round in range(self.args.comm_round):
            logging.info("************* Comm round %d begins *************" % com_round)
            loss_locals = []
            w_locals = []
            loss_avg = AverageMeter()
            
            client_indexes = self.aggregator.client_sampling(  # 每一轮Sample一些client
                com_round, self.args.client_num_in_total,
                self.args.client_num_per_round)

            for client_idx in client_indexes:
                if client_idx in self.labeled_client_idx:
                    local = self.lab_trainer_locals[client_idx]

                    w, loss, op = local.train(self.sup_net_locals[client_idx].state_dict())
                    w_locals.append(copy.deepcopy(w))
                    loss_avg.update(loss, 1)
                else:
                    if com_round * self.args.global_epochs_per_round >= 20:
                        local = self.unlab_trainer_locals[client_idx]
                        w, loss, op = local.train(self.unsup_net_locals[client_idx].state_dict(), com_round * self.args.global_epochs_per_round, avg_matrix)
                        w_locals.append(copy.deepcopy(w))
                        loss_avg.update(loss, 1)

                with torch.no_grad():
                    avg_matrix = self.lab_trainer_locals[self.labeled_client_idx[0]].confuse_matrix
                    for idx in self.labeled_client_idx[1:]:
                        avg_matrix = avg_matrix + self.lab_trainer_locals[idx].confuse_matrix
                    avg_matrix = avg_matrix / len(self.labeled_client_idx)

            with torch.no_grad():
                if len(w_locals) > 0:
                    w_glob = FedAvg(w_locals)

            self.net_glob.load_state_dict(w_glob)

            for i in self.labeled_client_idx:
                self.sup_net_locals[i].load_state_dict(w_glob)
            if com_round * self.args.global_epochs_per_round >= 20:
                for i in self.unlabeled_client_idx:
                    self.unsup_net_locals[i].load_state_dict(w_glob)
            print(loss_avg.avg, com_round)
            logging.info('Loss Avg {} Round {} '.format(loss_avg.avg, com_round))
            self.aggregator.set_global_model_params(self.net_glob.state_dict())
            avg_acc = self.aggregator.test_on_server_for_round(com_round, self.test_data_global_dl)
            self.test_acc_list.append(avg_acc)
            logging.info(f"This round: {com_round} acc is: {avg_acc}")