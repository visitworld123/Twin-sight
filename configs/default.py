from curses.ascii import FF
from logging import FATAL
import os
import random

# from .config import CfgNode as CN
from .config import CfgNode as CN

_C = CN()

_C.wandb_name = 'Test'
_C.wandb_project = 'ICLR2024_Twin-Sight'
_C.sim = 1.0
_C.insDis = 0.5
_C.SSFL_setting = 'partial_client' # partial_client or partial_data
_C.sup_agg_weight = 0.4
_C.SSL_method = 'freematch'
_C.PD_ulb_loss_ratio = 5.0

_C.PC_ulb_loss_ratio = 0.5
_C.PC_use_pesudo_label = False


_C.fedprox = True
_C.fedprox_mu = 0.5
_C.algorithm = 'FedAvg'

_C.lr = 0.01

_C.unlabeled_client_percentage = 0.6
_C.labeled_data_percentage = 0.05


_C.fixmatch_T = 0.5
_C.fixmatch_threshold = 0.95

_C.BYOL = False
_C.dataset = 'cifar10'
_C.client_num_in_total = 10
_C.client_num_per_round = 5
_C.gpu_index = 1 # for centralized training or standalone usage
_C.num_classes = 10
_C.model_output_dim = 10
_C.data_dir = '/data/zqy/data'
_C.partition_method = 'hetero'
_C.partition_alpha = 0.1
_C.model = 'SemiFed_SimCLR'
_C.model_input_channels = 3


_C.scaffold = False

_C.global_epochs_per_round = 1
_C.comm_round = 500

_C.seed = 0

_C.record_tool = 'wandb'  # using wandb or tensorboard

_C.batch_size = 64



# ---------------------------------------------------------------------------- #
# wandb settings
# ---------------------------------------------------------------------------- #
_C.entity = None
_C.project = 'test'
_C.wandb_upload_client_list = [0, 1] # 0 is the server
_C.wandb_save_record_dataframe = False
_C.wandb_offline = False
_C.wandb_record = False



# ---------------------------------------------------------------------------- #
# task settings
# ---------------------------------------------------------------------------- #
_C.task = 'classification' #    ["classification", "stackoverflow_lr", "ptb"]

# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #

_C.dataset_aug = "default"
_C.dataset_resize = False
_C.dataset_load_image_size = 32

_C.data_efficient_load = True    #  Efficiently load dataset, only load one full dataset, but split to many small ones.
_C.data_save_memory_mode = False    #  Clear data iterator, for saving memory, but may cause extra time.

_C.dirichlet_min_p = None #  0.001    set dirichlet min value for letting each client has samples of each label
_C.dirichlet_balance = False # This will try to balance dataset partition among all clients to make them have similar data amount

_C.data_load_num_workers = 1



# ---------------------------------------------------------------------------- #
# data sampler
# ---------------------------------------------------------------------------- #
_C.data_sampler = "random"  # 'random'



# ---------------------------------------------------------------------------- #
# data_preprocessing
# ---------------------------------------------------------------------------- #
_C.data_transform = "NormalTransform"  # or FLTransform
_C.TwoCropTransform = False

# ---------------------------------------------------------------------------- #
# record config
# ---------------------------------------------------------------------------- #
_C.record_dataframe = False
_C.record_level = 'epoch'   # iteration


# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #

_C.model_out_feature = False
_C.model_out_feature_layer = "last"
_C.model_feature_dim = 512



# ---------------------------------------------------------------------------- #
# generator
# ---------------------------------------------------------------------------- #
_C.image_resolution = 32


# ---------------------------------------------------------------------------- #
# Contrastive
# ---------------------------------------------------------------------------- #
_C.Contrastive = "no"                   # SimCLR, SupCon



# ---------------------------------------------------------------------------- #
# Average weight
# ---------------------------------------------------------------------------- #
"""[even, datanum, inv_datanum, inv_datanum2datanum, even2datanum,
        ]
"""
# datanum2others is not considerred for now.
_C.fedavg_avg_weight_type = 'datanum'   


# ---------------------------------------------------------------------------- #
# loss function
# ---------------------------------------------------------------------------- #
_C.loss_fn = 'CrossEntropy'



_C.wd = 0.0001
_C.momentum = 0.9
_C.nesterov = False
_C.clip_grad = False



# ---------------------------------------------------------------------------- #
# Learning rate schedule parameters
# ---------------------------------------------------------------------------- #
_C.sched = 'no'   # no (no scheudler), StepLR MultiStepLR  CosineAnnealingLR
_C.lr_decay_rate = 0.992
_C.step_size = 1
_C.lr_milestones = [30, 60]
_C.lr_T_max = 20
_C.lr_eta_min = 0.005
_C.lr_warmup_type = 'constant' # constant, gradual.
_C.warmup_epochs = 0
_C.lr_warmup_value = 0.1


# ---------------------------------------------------------------------------- #
# logging
# ---------------------------------------------------------------------------- #
_C.level = 'INFO' # 'INFO' or 'DEBUG'

