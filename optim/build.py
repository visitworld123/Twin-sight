import logging

import torch
import optim.AdamW

from utils.data_utils import scan_model_with_depth
from utils.model_utils import build_param_groups
# Discard
# from .fedprox import FedProx


"""
    args.opt in 
    ["sgd", "adam"]
    --lr
    --momentum
    --clip-grad # wait to be developed
    --weight-decay, --wd
"""



def create_optimizer(args, model=None, params=None, **kwargs):
    if "role" in kwargs:
        role = kwargs["role"]
    else:
        role = args.role

    # params has higher priority than model
    if params is not None:
        params_to_optimizer = params
    else:
        if model is not None:
            params_to_optimizer = model.parameters()
        else:
            raise NotImplementedError
        pass

    if (role == 'server') and (args.algorithm in [
        'FedAvg']):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params_to_optimizer),
            lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
    else:
        optimizer = torch.optim.SGD(params_to_optimizer,
            lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)

    return optimizer







