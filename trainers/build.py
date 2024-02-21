import os
import sys

from .normal_trainer import NormalTrainer

from optim.build import create_optimizer
from loss_fn.build import create_loss



def create_trainer(args, device, model=None, **kwargs):

    params = None
    optimizer = create_optimizer(args, model, params=params, **kwargs)


    criterion = create_loss(args, device, **kwargs)

    model_trainer = NormalTrainer(model, device, criterion, optimizer,  args, **kwargs)

    return model_trainer












