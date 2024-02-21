

from algorithms_standalone.basePS.client import Client

from model.build import create_model



class FedAVGClient(Client):
    def __init__(self,client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, local_ds_pesudolabel_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, local_ds_pesudolabel_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num)
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round

        # override the PSClientManager's timer


        #========================SCAFFOLD=====================#
        if self.args.scaffold:
            self.c_model_local = create_model(self.args,
                model_name=self.args.model, output_dim=self.args.model_output_dim)
            for name, params in self.c_model_local.named_parameters():
                params.data = params.data*0


    # override
    def lr_schedule(self, num_iterations, warmup_epochs):
        epoch = None
        iteration = None
        round_idx = self.client_timer.local_comm_round_idx
        if self.args.sched == "no":
            pass
        else:
            if round_idx < warmup_epochs:
                # Because gradual warmup need iterations updates
                self.trainer.warmup_lr_schedule(round_idx*num_iterations)
            else:
                self.trainer.lr_schedule(round_idx)










