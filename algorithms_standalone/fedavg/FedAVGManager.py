
import logging

from .client import FedAVGClient
from .aggregator import FedAVGAggregator


from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer


class FedAVGManager(BasePSManager):
    def __init__(self, device, args):
        super().__init__(device, args)

        self.global_epochs_per_round = self.args.global_epochs_per_round
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

            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.num_classes,
                            device=self.device, **self.other_params)

            model_trainer = create_trainer(self.args, self.device, model,class_num=self.class_num,
                                           other_params=self.other_params,client_index=client_index, role='client',
                                           **init_state_kargs)

            client = FedAVGClient(client_index, train_ori_data=self.train_data_local_ori_dict[client_index],
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

            # client.prepare_embedding_by_SSFL_model(self.embedding_model)

            self.client_list.append(client)
        logging.info("############setup_clients (END)#############")

