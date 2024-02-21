import logging

from utils.data_utils import (
    get_name_params_difference
)


# 每个client一个
class PSTrainer(object):

    def __init__(self, client_index, train_ori_data, train_ori_targets, train_data_num,
                device, args, model_trainer):

        self.args = args
        self.client_index = client_index
        self.train_ori_data = train_ori_data
        self.train_ori_targets = train_ori_targets
        self.local_sample_number = train_data_num
        

        logging.info(f"Initializing client: {self.client_index}"+
                    f" len(train_data) (local data num): {len(self.train_ori_data)} ")


        self.device = device
        self.trainer = model_trainer
        # =============================================

# # got it
#     def get_num_iterations(self):
#         local_num_iterations = get_local_num_iterations(self.local_sample_number, self.args.batch_size)  # local_num_iterations = local_sample_number // batch_size
#
#         # 平均需要的iterations数 = 总数据量 // work数量 // batch_size = 每个work平均数据量 // batch_size
#         global_num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
#         return local_num_iterations, global_num_iterations

    def get_trainer_model(self):
        return self.trainer.get_model()

    def update_state(self, **kwargs):
        self.trainer.update_state(**kwargs)



    def lr_schedule(self, progress):
        self.trainer.lr_schedule(progress)

    def warmup_lr_schedule(self, iterations):
        self.trainer.warmup_lr_schedule(iterations)


    def set_trainer_model_params(self, weights):
        '''

        '''
        self.trainer.set_model_params(weights)
        
    def clear_grad_params(self):
        self.trainer.clear_grad_params()

    def update_model_with_grad(self):
        self.trainer.update_model_with_grad()

# got it 创建一个可迭代的DataLoader对象
    def get_train_batch_data(self):
        try:
            train_batch_data = self.train_local_iter.next()
            logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(self.train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data

# got it
    def get_model_params(self):
        '''
        return:
        @compressed_weights: 压缩后的model参数
        @model_indexes：只在compress中用处，在没有压缩的模型中为None
        '''
        weights = self.trainer.get_model_params()
        model_indexes = None

        return weights, model_indexes


    def get_model_diff_params(self, previous_model):
        weights = self.trainer.get_model_params()
        weights_diff = get_name_params_difference(previous_model, weights)

        compressed_weights_diff = weights_diff
        model_indexes = None

        return compressed_weights_diff, model_indexes


    def get_model_grads(self):
        named_grads = self.trainer.get_model_grads()
        # logging.debug(named_grads)
        compressed_grads = named_grads
        grad_indexes = None

        return compressed_grads, grad_indexes



    def local_test(self, epoch, tracker=None, metrics=None):
        self.trainer.test(self.test_local, self.device, self.args,
            epoch, tracker, metrics)

