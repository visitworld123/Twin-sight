import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))



from algorithms.basePS.ps_aggregator import PSAggregator


class Aggregator(PSAggregator):
    def __init__(self, train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer):
        super().__init__(train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer)


    def get_max_comm_round(self):
        pass






















