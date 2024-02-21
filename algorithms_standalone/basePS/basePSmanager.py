import logging
import copy

import torch

from utils.metrics import Metrics
from utils.data_utils import (
    get_selected_clients_label_distribution
)
from data_preprocessing.build import load_data

from utils.tool import *
from model.SSFL_ResNet18 import *
from ssl_model.BYOL import *
from ssl_model.ssl_utils import *

class BasePSManager(object):
    def __init__(self, device, args):
        self.device = device
        self.args = args
        # ================================================
        self._setup_datasets()
        self.selected_clients = None
        self.client_list = []
        self.aggregator = None
        self.metrics = Metrics([1], task=self.args.task)
        # ================================================
        self.number_instantiated_client = self.args.client_num_in_total
        self.client_sample_weight = [ 10  for i in range(self.args.client_num_in_total)]  # initialize the weight is 10 for every client



        # aggregator will be initianized in _setup_server()
        self.comm_round = self.args.comm_round
        # ================================================
        #    logging all acc
        self.test_acc_list = []
        # self._prepare_test_dataLoader(self.embedding_model)

    def _load_existing_SSFL_model(self):
        self.embedding_model = ResNet18_BYOL(self.args.num_classes)
        param = torch.load('alpha0.1_10client_SSFL_CIFAR10.pth', map_location=self.device)
        self.embedding_model.fc = MLP(512, 2048, 4096)
        self.embedding_model.load_state_dict(param)
        self.embedding_model.fc = nn.Identity()


# got it
    def _setup_datasets(self):
        # dataset = load_data(self.args, self.args.dataset)

        train_data_global_num, test_data_global_num, train_data_global_dl, test_data_global_dl,train_data_local_num_dict, \
        train_data_local_ori_dict, train_targets_local_ori_dict, class_num, \
        other_params = load_data(load_as="training", args=self.args, process_id=0, mode="standalone", task="federated", data_efficient_load=True,
                                 dirichlet_balance=False, dirichlet_min_p=None,dataset=self.args.dataset, datadir=self.args.data_dir,
                                 partition_method=self.args.partition_method, partition_alpha=self.args.partition_alpha,
                                 client_number=self.args.client_num_in_total, batch_size=self.args.batch_size, num_workers=self.args.data_load_num_workers,
                                 data_sampler=self.args.data_sampler,resize=self.args.dataset_load_image_size, augmentation=self.args.dataset_aug)


        self.other_params = other_params
        self.train_data_global_dl = train_data_global_dl
        self.test_data_global_dl = test_data_global_dl
        self.selected_clients_total = []
        self.train_data_global_num = train_data_global_num
        self.test_data_global_num = test_data_global_num

        self.train_data_local_num_dict = train_data_local_num_dict  # {client_idx: client_idx_train_num}
        self.train_data_local_ori_dict = train_data_local_ori_dict  # {client_idx: client_idx_train_ori_data}
        self.train_targets_local_ori_dict = train_targets_local_ori_dict # {client_idx: client_idx_ori_targets}
        self.client_dataidx_map = other_params['client_dataidx_map']
        self.train_cls_local_counts_dict = other_params['train_cls_local_counts_dict']
        self.labeled_client_idx = self.other_params["labeled_client_idx"]
        self.unlabeled_client_idx = self.other_params["unlabeled_client_idx"] 
        self.local_labeled_ds_PC_dict = self.other_params["local_labeled_ds_PC_dict"]
        self.local_unlabeled_ds_PC_dict = self.other_params["local_unlabeled_ds_PC_dict"]
        self.local_labeled_ds_PD_dict = self.other_params["local_labeled_ds_PD_dict"]
        self.local_unlabeled_ds_PD_dict = self.other_params["local_unlabeled_ds_PD_dict"]
        self.local_ds_pesudolabel_PD_dict = self.other_params["local_ds_pesudolabel_PD_dict"]

        self.class_num = class_num

        if "train_cls_local_counts_dict" in self.other_params:
            # 字典嵌套字典
            self.train_cls_local_counts_dict = self.other_params["train_cls_local_counts_dict"]  # {client_idx:{label0: labe0_num_client_idx,...,label9: labe9_num_client_idx}}
            # Adding missing classes to list
            classes = list(range(self.class_num)) # [0,1,2,3,4,...,class_num - 1]
            for key in self.train_cls_local_counts_dict:
                # key means the client index
                if len(classes) != len(self.train_cls_local_counts_dict[key]): # client_key没有所有类别
                    # print(len(classes))
                    # print(len(train_data_cls_counts[key]))
                    add_classes = set(classes) - set(self.train_cls_local_counts_dict[key])
                    # print(add_classes)
                    for e in add_classes:
                        self.train_cls_local_counts_dict[key][e] = 0   # 把剩下的类补全，都为0
        else:
            self.train_cls_local_counts_dict = None


    def _setup_server(self):
        pass

    def _setup_clients(self):
        pass


    def _prepare_test_dataLoader(self, SSFL_model):
        X_test, y_test = inference_by_SSFL_model(SSFL_model, self.aggregator.test_dataloader, self.device)
        self.global_test_dataloader_SSFL = create_data_loaders_from_arrays(X_test, y_test, 64)

    def test(self):
        logging.info("################test_on_server_for_all_clients : {}".format(
            self.server_timer.global_outer_epoch_idx))
        avg_acc = self.aggregator.test_on_server_for_all_clients(
            self.server_timer.global_outer_epoch_idx, self.total_test_tracker, self.metrics)

        return avg_acc


    def get_init_state_kargs(self):
        self.selected_clients = [i for i in range(self.args.client_num_in_total)]
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.selected_clients_label_distribution = get_selected_clients_label_distribution(   # 整体selected_client的label分布 list[,,,,]每一个元素代表选中的客户端所有的这一类别的数量
                self.local_cls_num_list_dict, self.class_num, self.selected_clients, min_limit=1)
            init_state_kargs = {"weight": None, "selected_cls_num_list": self.selected_clients_label_distribution,
                                "local_cls_num_list_dict": self.local_cls_num_list_dict}
        else:
            init_state_kargs = {}
        return init_state_kargs


    def get_update_state_kargs(self):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss", "local_FocalLoss", "local_LDAMLoss"]:
            self.selected_clients_label_distribution = get_selected_clients_label_distribution(
                self.local_cls_num_list_dict, self.class_num, self.selected_clients, min_limit=1)
            update_state_kargs = {"weight": None, "selected_cls_num_list": self.selected_clients_label_distribution,
                                "local_cls_num_list_dict": self.local_cls_num_list_dict}
        else:
            update_state_kargs = {}
        return update_state_kargs



    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.server_timer.global_outer_epoch_idx
        iterations = self.server_timer.global_outer_iter_idx

        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.aggregator.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.aggregator.trainer.lr_schedule(epochs)

    # ==============train clients and add results to aggregator ===================================
    def train(self):
        for round in range(self.comm_round):

            logging.info("################Communication round : {}".format(round))
            # w_locals = []
            downloaded_model_params = copy.deepcopy(self.aggregator.compressed_global_model_param())
                # ========================SCAFFOLD=====================#
                # if self.args.scaffold:
                #     c_global_para = self.aggregator.c_model_global.state_dict()
                #     global_other_params["c_model_global"] = c_global_para
                # ========================SCAFFOLD=====================#

# ----------------- sample clinet saving in manager------------------#

            client_indexes = self.aggregator.client_sampling(    # 每一轮Sample一些client
                    round, self.args.client_num_in_total,
                    self.args.client_num_per_round)
            # client_indexes = np.random.choice(self.labeled_client_idx, math.ceil(len(self.labeled_client_idx) * 0.5), replace=False)
            logging.info("sampling client_indexes = %s" % str(client_indexes))

            self.selected_clients = client_indexes
            self.selected_clients_total.append((self.selected_clients,round))
            # global_time_info = self.server_timer.get_time_info_to_send()
            # update_state_kargs = self.get_update_state_kargs()     # do nothing

            # -----------------train model using algorithm_train and aggregate------------------#
            if self.args.SSFL_setting == 'partial_client':
                self.train_locally_per_round_PC(round, downloaded_model_params)
            elif self.args.SSFL_setting == 'partial_data':
                self.train_locally_per_round_PD(round, downloaded_model_params)
            # -----------------aggregation procedure------------------#
            self.aggregator.aggregation()
            #
            # if round % 100 == 0:
            #     self.aggregator.save_server_model('aplha{}_SimCLR_CIFAR10_{}.pth'.format(self.args.partition_alpha,round))
            # if round == 999:
            #     self.aggregator.save_server_model('aplha{}_SimCLR_CIFAR10_{}.pth'.format(self.args.partition_alpha, round))
            avg_acc = self.aggregator.test_on_server_for_round(round, self.test_data_global_dl)
            self.test_acc_list.append(avg_acc)
            logging.info(f"This round: {round} acc is: {avg_acc}")


    def train_locally_per_round_PC(self, round, downloaded_model_params):
        uploaded_clients_weights = []
        uploaded_models_params = []
        add_client_index = []
        clients_label_flags = []
        copy_downloaded_model_params = copy.deepcopy(downloaded_model_params)
        for num, client_index in enumerate(self.selected_clients):
            # Update client config before training
            # get the one of the current selected client
            client = self.client_list[client_index]
            label_flag = True if client_index in self.labeled_client_idx else False
            upload_info = client.run_train(round, copy_downloaded_model_params, label_flag)
            clients_label_flags.append(label_flag)
            uploaded_models_params.append(upload_info['MODEL_PARAMS'])
            uploaded_clients_weights.append(upload_info['SAMPLE_NUM'])
            # if client.label_flag == True:
            #     uploaded_class_centroids.append(upload_info['CLASS_CENTROID'])
            #     uploaded_class_centroid_weights.append(upload_info['SAMPLE_NUM'])
        uploaded_info_for_server = {'CLIENTS_WEIGHTS':uploaded_clients_weights,
                                    'MODELS_PARAMS': uploaded_models_params,
                                    'CLIENTS_LABEL_FLAGS': clients_label_flags}
        logging.info(uploaded_info_for_server['CLIENTS_LABEL_FLAGS'])

        self.aggregator.download_info_from_selected_clients(uploaded_info_for_server)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(self.selected_clients))

    def train_locally_per_round_PD(self, round, downloaded_model_params):
        uploaded_clients_weights = []
        uploaded_models_params = []
        copy_downloaded_model_params = copy.deepcopy(downloaded_model_params)
        for num, client_index in enumerate(self.selected_clients):
            # Update client config before training
            # get the one of the current selected client
            client = self.client_list[client_index]
            # if client.label_flag == False and self.args.model != 'SemiFed':
            #     logging.info('client {} have no label'.format(client.client_index))
            #     continue
            upload_info = client.run_train_PD(round, copy_downloaded_model_params)
            uploaded_models_params.append(upload_info['MODEL_PARAMS'])
            uploaded_clients_weights.append(upload_info['SAMPLE_NUM'])
            # if client.label_flag == True:
            #     uploaded_class_centroids.append(upload_info['CLASS_CENTROID'])
            #     uploaded_class_centroid_weights.append(upload_info['SAMPLE_NUM'])
        uploaded_info_for_server = {'CLIENTS_WEIGHTS':uploaded_clients_weights,
                                    'MODELS_PARAMS': uploaded_models_params}

        self.aggregator.download_info_from_selected_clients(uploaded_info_for_server)
        logging.info("sampling client_indexes = %s finished the update and upload the info" % str(self.selected_clients))

