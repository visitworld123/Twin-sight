import argparse
import logging
import os
import socket
import sys

import torch
# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from utils.logger import (
    logging_config
)

from configs import get_cfg

from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
from RSCFed.RSCFedManager import RSCFedManager
from FedIRM.FedIRMManager import FedIRMManager
from utils.set import *

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str,
                        help="specify add which type of config")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    # parse python script input parameters

    #----------loading personalized params-----------------#
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args.config_file)
    #### set up cfg ####
    # default cfg
    cfg = get_cfg()

    # add registered cfg
    # some arguments that are needed by build_config come from args.
    cfg.setup(args)

    # Build config once again
    #cfg.setup(args)
    cfg.mode = 'standalone'

    # cfg.rank = process_id
    # if cfg.algorithm in ['FedAvg', 'AFedAvg', 'PSGD', 'APSGD', 'Local_PSGD']:
    #     cfg.client_index = process_id - 1
    # elif cfg.algorithm in ['DPSGD', 'DCD_PSGD', 'CHOCO_SGD', 'SAPS_FL']:
    #     cfg.client_index = process_id
    # else:
    #     raise NotImplementedError
    cfg.server_index = -1
    cfg.client_index = -1
    seed = cfg.seed
    process_id = 0
    # show ultimate config
    logging.info(dict(cfg))

    #-------------------customize the process name-------------------
    str_process_name = cfg.algorithm + " (standalone):" + str(process_id)
    #setproctitle.setproctitle(str_process_name)

    logging_config(args=cfg, process_id=process_id)

    # loggcing.info("In Fed Con model construction, model is {}, {}, {}".format(
    #     cfg.model, type(cfg.model), cfg.model == 'simple-cnn'
    # ))

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if cfg.wandb_offline:
        os.environ['WANDB_MODE'] = 'dryrun'

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    set_random_seed(seed)
    torch.backends.cudnn.deterministic =True

    device = 'cpu'# torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    if cfg.record_tool == 'wandb':
        import wandb
        wandb.init(config=cfg, name=cfg.wandb_name,
                   project=cfg.wandb_project)
    if cfg.algorithm == 'FedAvg':
        fedavg_manager = FedAVGManager(device, cfg)
        fedavg_manager.train()
    elif cfg.algorithm == 'FedIRM':
        fedirm_manager = FedIRMManager(device, cfg)
        fedirm_manager.train()
    elif cfg.algorithm == 'RSCFed':
        rscfed_manager = RSCFedManager(device, cfg)
        rscfed_manager.train()
    else:
        raise NotImplementedError
    wandb.finish()
    