

from algorithms_standalone.basePS.aggregator import Aggregator


from model.build import create_model


class FedAVGAggregator(Aggregator):
    def __init__(self,train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer):
        super().__init__(train_dataloader, test_dataloader, train_data_num, test_data_num,
                        train_data_local_num_dict, worker_num, device,args, model_trainer)

        if self.args.scaffold:
            self.c_model_global = create_model(self.args,
                model_name=self.args.model, output_dim=self.args.model_output_dim)
            for name, params in self.c_model_global.named_parameters():
                params.data = params.data*0

    def get_max_comm_round(self):
        # return self.args.max_comm_round + 1
        return self.args.comm_round

