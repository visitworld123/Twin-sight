import copy

import torch

from utils.data_utils import(
    get_data,
    filter_parameters,
    mean_std_online_estimate,
    retrieve_mean_std,
    get_tensors_norm,
    average_named_params,
    idv_average_named_params,
    get_name_params_div,
    get_name_params_sum,
    get_name_params_difference,
    get_name_params_difference_norm,
    get_name_params_difference_abs,
    get_named_tensors_rotation,
    calculate_metric_for_named_tensors,
    get_diff_tensor_norm,
    get_tensor_rotation,
    calculate_metric_for_whole_model,
    calc_client_divergence,
    check_device
)

from utils.tensor_buffer import (
    TensorBuffer
)



class Averager(object):
    """
        Responsible to implement average.
        There maybe some history information need to be memorized.
    """
    def __init__(self, args, model):
        self.args = args

    def get_average_weight(self, sample_num_list, avg_weight_type=None, global_outer_epoch_idx=0,
            inplace=True):
        '''
        sample_num_list : [(sample_number, label_flag)]
        '''
        # balance_sample_number_list = []
        average_weights_dict_list = []
        sum = 0
        inv_sum = 0 

        sample_num_list = copy.deepcopy(sample_num_list)
        # for i in range(0, len(sample_num_list)):
        #     sample_num_list[i] * np.random.random(1)
        for i in range(0, len(sample_num_list)):
            local_sample_number,_ = sample_num_list[i]
            inv_sum = None
            sum += local_sample_number

        for i in range(0, len(sample_num_list)):
            local_sample_number,flag = sample_num_list[i]
            if avg_weight_type == 'datanum':
                weight_by_sample_num = local_sample_number / sum
            average_weights_dict_list.append(weight_by_sample_num)

        homo_weights_list = average_weights_dict_list
        return average_weights_dict_list, homo_weights_list











