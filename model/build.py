from data_preprocessing.utils.stats import get_dataset_image_size
import logging

import torch
import torchvision.models as models

from model.linear.lr import LogisticRegression
from model.SSFL_ResNet18 import ResNet18

from ssl_model.BYOL import *
from ssl_model.SimCLR import *
from model.SemiFed import *
from model.SVHN_model import SVHN_model
from ssl_model.SimSiam import *


CV_MODEL_LIST = []
RNN_MODEL_LIST = ["rnn"]


def create_model(args, model_name, output_dim, pretrained=False, device=None, **kwargs):
    model = None
    logging.info(f"model name: {model_name}")

    if model_name in RNN_MODEL_LIST:
        pass
    else:
        image_size = get_dataset_image_size(args.dataset)


    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "resnet18":
        logging.info("ResNet18_v2")
        model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
    elif model_name == 'SimCLR':
        net = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
        model = SimCLRModel(net,image_size=image_size)
    elif model_name == 'SemiFed_SimCLR':
        logging.info("SemiFed_SimCLR")
        net = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
        unsup_model = SimCLRModel(net, image_size=image_size)
        sup_model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
        logging.info("SemiFed")
        model = SemiFed(unsup_model, sup_model, args=args, image_size=image_size, input_channel=args.model_input_channels, num_class=output_dim)
    elif model_name == 'SemiFed_SimSiam':
        logging.info("SemiFed_SimSiam")

        net = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                    model_input_channels=args.model_input_channels)
        unsup_model = SimSiam(net)
        sup_model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                                model_input_channels=args.model_input_channels)
        model = SemiFed(unsup_model, sup_model, args=args, image_size=image_size,
                        input_channel=args.model_input_channels)
    elif model_name == 'SemiFed_BYOL':
        logging.info("SemiFed_BYOL")
        net = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                    model_input_channels=args.model_input_channels)
        unsup_model = BYOLmodel(net, image_size=image_size)
        sup_model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                                model_input_channels=args.model_input_channels)
        model = SemiFed(unsup_model, sup_model, args=args, image_size=image_size,
                        input_channel=args.model_input_channels)

    elif model_name == 'SVHN_model':
        logging.info("SVHN_model")
        model = SVHN_model(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
    else:
        raise NotImplementedError
    return model

