import torchvision.datasets

from algorithms_standalone.basePS.basePSmanager import BasePSManager
import logging
import copy
import random
from RSCFed.rscfed_dataset import  *

from RSCFed.local_supervised import SupervisedLocalUpdate
from RSCFed.local_unsupvervised import UnsupervisedLocalUpdate
from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer
import math
import numpy as np
from model.SSFL_ResNet18 import ResNet18
from algorithms_standalone.fedavg.aggregator import FedAVGAggregator
from utils.tool import *
from RSCFed.FedAvg import *
from utils.set import *

class RSCFedManager(BasePSManager):
    def __init__(self, device, args):
        super().__init__(device, args)

        self.lab_trainer_locals = {}
        self.unlab_trainer_locals = {}

        logging.info("Unlabeled Client : %s" % (str(self.unlabeled_client_idx)))
        self.net_glob = ResNet18(args=self.args, num_classes=self.args.model_output_dim, image_size=32,
                                    model_input_channels=self.args.model_input_channels)

        self.net_glob.train()
        self.w_glob = self.net_glob.state_dict()

        self.w_locals = {}
        self.w_ema_unsup = {}
        self.sup_net_locals = {}
        self.unsup_net_locals = {}
        # self._load_existing_SSFL_model()
        self._setup_clients()
        # ================================================
        self._setup_server()

    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.num_classes,
                            device=self.device, **self.other_params)
        init_state_kargs = {} #self.get_init_state_kargs()   # 1.设置selected_clients 为所有client  2.FedAvg中init_state_kargs={}


        model_trainer = create_trainer(     # 设置
            self.args, self.device, model,train_data_global_num=self.train_data_global_num,
            test_data_global_num=self.test_data_global_num, train_data_global_dl=self.train_data_global_dl,
            test_data_global_dl=self.test_data_global_dl, train_data_local_num_dict=self.train_data_local_num_dict,
            class_num=self.class_num,server_index=0, role='server',**init_state_kargs
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
                client = UnsupervisedLocalUpdate(client_index,train_ori_data=self.train_data_local_ori_dict[client_index],
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

        dist_scale_f = 1

        total_lenth = sum([client.local_sample_number for client in self.client_list])
        each_lenth = [client.local_sample_number for client in self.client_list]
        client_freq = [client.local_sample_number / total_lenth for client in self.client_list]

        for round in range(self.comm_round):
            logging.info("************* Comm round %d begins *************" % round)

            loss_avg = AverageMeter()
            clt_this_comm_round = []
            w_per_meta = []

            for meta_round in range(2):
                logging.info("################Communication round : {}".format(round))
                client_indexes = self.aggregator.client_sampling(  # 每一轮Sample一些client
                    round, self.args.client_num_in_total,
                    self.args.client_num_per_round)

                clt_list_this_meta_round = client_indexes
                clt_this_comm_round.extend(clt_list_this_meta_round)
                chosen_sup = [j for j in self.labeled_client_idx if j in clt_list_this_meta_round]
                logging.info(f'Comm round {round} meta round {meta_round} chosen client: {clt_list_this_meta_round}')
                w_locals_this_meta_round = []

                for client_idx in clt_list_this_meta_round:
                    if client_idx in self.labeled_client_idx:
                        client = self.client_list[client_idx]
                        w, loss, op = client.train(
                            self.sup_net_locals[client_idx].state_dict())  # network, loss, optimizer
                        log_info('scalar', 'Supervised loss on sup client %d' % client_idx, loss, 0,
                                 self.args.record_tool)
                        w_locals_this_meta_round.append(copy.deepcopy(w))
                        loss_avg.update(loss, 1)
                        logging.info('Labeled client {} sample num: '
                                     '{} training loss : {} '.format(client_idx, client.local_sample_number, loss))
                    else:
                        client = self.client_list[client_idx]
                        w, w_ema, loss, op, ratio, \
                        all_pseu, test_right, test_right_ema, \
                        same_pred_num = client.train(
                            self.unsup_net_locals[client_idx].state_dict(),
                            client_idx)
                        log_info('scalar', 'Unsupervised loss on unsup client %d' % client_idx, loss,
                                 0, self.args.record_tool)
                        w_locals_this_meta_round.append(copy.deepcopy(w))
                        self.w_ema_unsup[client_idx] = copy.deepcopy(w_ema)
                        loss_avg.update(loss, 1)
                        logging.info(
                            'Unlabeled client {} sample num: {} Training loss: {}, unsupervised loss ratio: {}, \
                             {} correct by model, {} correct by ema before train'.format(
                                client_idx, client.local_sample_number, loss,
                                ratio,test_right, test_right_ema))
                each_lenth_this_meta_round = [each_lenth[clt] for clt in clt_list_this_meta_round]
                each_lenth_this_meta_raw = copy.deepcopy(each_lenth_this_meta_round)

                total_lenth_this_meta = sum(each_lenth_this_meta_round)
                clt_freq_this_meta_round = [i / total_lenth_this_meta for i in each_lenth_this_meta_round]

                logging.info('Based on data amount: ' + f'{clt_freq_this_meta_round}')
                clt_freq_this_meta_raw = copy.deepcopy(clt_freq_this_meta_round)

                w_avg_temp = FedAvg(w_locals_this_meta_round, clt_freq_this_meta_round)
                dist_list = []
                for cli_idx in range(self.args.client_num_per_round):
                    dist = model_dist(w_locals_this_meta_round[cli_idx], w_avg_temp)
                    dist_list.append(dist)
                print(
                    'Normed dist * 1e4 : ' + f'{[dist_list[i] * 1e5 / each_lenth_this_meta_raw[i] for i in range(self.args.client_num_per_round)]}')

                if len(chosen_sup) != 0:
                    clt_freq_this_meta_uncer = [
                        np.exp(-dist_list[i] * dist_scale_f/ each_lenth_this_meta_raw[i]) * clt_freq_this_meta_round[
                            i] for i
                        in
                        range(self.args.client_num_per_round)]
                else:
                    clt_freq_this_meta_uncer = [
                        np.exp(-dist_list[i] * dist_scale_f / each_lenth_this_meta_raw[i]) * clt_freq_this_meta_round[i]
                        for i in range(self.args.client_num_per_round)]
                total = sum(clt_freq_this_meta_uncer)
                clt_freq_this_meta_dist = [clt_freq_this_meta_uncer[i] / total for i in
                                           range(self.args.client_num_per_round)]
                clt_freq_this_meta_round = clt_freq_this_meta_dist
                print('After dist-based uncertainty : ' + f'{clt_freq_this_meta_round}')

                assert sum(clt_freq_this_meta_round) - 1.0 <= 1e-3, "Error: sum(freq) != 0"
                w_this_meta = FedAvg(w_locals_this_meta_round, clt_freq_this_meta_round)
                avg_acc = self.aggregator.test_on_server_for_round(round, self.test_data_global_dl)
                self.test_acc_list.append(avg_acc)
                logging.info(f"This meta_round: {meta_round} acc is: {avg_acc}")
                self.aggregator.set_global_model_params(copy.deepcopy(w_this_meta))
                w_per_meta.append(w_this_meta)
            each_lenth_this_round = [each_lenth[clt] for clt in clt_this_comm_round]

            with torch.no_grad():
                freq = [1 / 3 for i in range(3)]
                w_glob = FedAvg(w_per_meta, freq)

            self.net_glob.load_state_dict(w_glob)

            for i in self.labeled_client_idx:
                self.sup_net_locals[i].load_state_dict(w_glob)
            for i in self.unlabeled_client_idx:
                self.unsup_net_locals[i].load_state_dict(w_glob)

            logging.info(
                '************ Loss Avg {},  Round {} ends ************  '.format(loss_avg.avg, round))
            self.aggregator.set_global_model_params(self.net_glob.state_dict())
            avg_acc = self.aggregator.test_on_server_for_round(round, self.test_data_global_dl)
            self.test_acc_list.append(avg_acc)
            logging.info(f"This round: {round} acc is: {avg_acc}")