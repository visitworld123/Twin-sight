import copy

import torch


def get_average_weight(weights_list):
    # balance_sample_number_list = []
    average_weights_dict_list = []
    sum = 0

    weights_list = copy.deepcopy(weights_list)
    # for i in range(0, len(sample_num_list)):
    #     sample_num_list[i] * np.random.random(1)

    for i in range(0, len(weights_list)):
        local_sample_number = weights_list[i]
        inv_sum = None
        sum += local_sample_number

    for i in range(0, len(weights_list)):
        local_sample_number = weights_list[i]

        weight_by_sample_num = local_sample_number / sum

        average_weights_dict_list.append(weight_by_sample_num)

    return average_weights_dict_list

def average_named_params(named_params_list, average_weights_dict_list):
    """
        This is a weighted average operation.
        average_weights_dict_list: includes weights with respect to clients. Same for each param.
        inplace:  Whether change the first client's model inplace.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))
    averaged_params = copy.deepcopy(named_params_list[0])
#   averaged_params is the dict of all parameters      #
    for k in averaged_params.keys():
        for i in range(0, len(named_params_list)):  # model个数
            if type(named_params_list[0]) is tuple or type(named_params_list[0]) is list:
                local_sample_number, local_named_params = named_params_list[i]
            else:
                local_named_params = named_params_list[i]
            # logging.debug("aggregating ---- local_sample_number/sum: {}/{}, ".format(
            #     local_sample_number, sum))
            w = average_weights_dict_list[i]  # 不同client的权重
            # w = torch.full_like(local_named_params[k], w).detach()
            if i == 0:
                averaged_params[k] = (local_named_params[k] * w).type(averaged_params[k].dtype)
            else:
                averaged_params[k] += (local_named_params[k].to(averaged_params[k].device) * w).type(
                    averaged_params[k].dtype)
    return averaged_params

def federated_averaging_by_params(models_params, weights):
    """Compute weighted average of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    average_weights_dict_list = get_average_weight(weights)
    model_params = average_named_params(models_params, average_weights_dict_list)

    return model_params


def federated_averaging_only_params(models, weights):
    """Compute weighted average of model parameters. Use model parameters only.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    if models == [] or weights == []:
        return None

    model, total_weights = weighted_sum_only_params(models, weights)
    model_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, params in model_params.items():
            model_params[name].set_(model_params[name] / total_weights)

    return model


def weighted_sum(models, weights):
    """Compute weighted sum of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
        float: Sum of weights.
    """
    if models == [] or weights == []:
        return None
    model = copy.deepcopy(models[0])
    model_sum_params = copy.deepcopy(models[0].state_dict())

    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].state_dict())
                params += model_params[name] * weights[i]
            model_sum_params[name] = params
    model.load_state_dict(model_sum_params)
    return model, sum(weights)


def weighted_sum_only_params(models, weights):
    """Compute weighted sum of model parameters. Use model parameters only.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
        float: Sum of weights.
    """
    if models == [] or weights == []:
        return None

    model_sum = copy.deepcopy(models[0])
    model_sum_params = dict(model_sum.named_parameters())

    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].named_parameters())
                params += model_params[name] * weights[i]
            model_sum_params[name].set_(params)
    return model_sum


def equal_weight_averaging(models):
    if models == []:
        return None

    model_avg = copy.deepcopy(models[0])
    model_avg_params = dict(model_avg.named_parameters())

    with torch.no_grad():
        for name, params in model_avg_params.items():
            for i in range(1, len(models)):
                model_params = dict(models[i].named_parameters())
                params += model_params[name]
            model_avg_params[name].set_(params / len(models))
    return model_avg
