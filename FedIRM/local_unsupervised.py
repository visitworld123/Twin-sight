from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision import transforms

from algorithms_standalone.basePS.client import Client
from model.SSFL_ResNet18 import ResNet18

from typing import Dict, Any, Set, Tuple, Optional
from abc import ABC, abstractmethod
from RSCFed.rscfed_dataset import *
from model.SVHN_model import SVHN_model
from FedIRM.confuse_matrix import *
from utils.set import *
from utils.log_info import log_info


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    return sigmoid_rampup(epoch, 30)

class RampUp(ABC):
    def __init__(self, length: int, current: int = 0):
        self.current = current
        self.length = length

    @abstractmethod
    def __call__(self, current: Optional[int] = None, is_step: bool = True) -> float:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return dict(current=self.current, length=self.length)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if strict:
            is_equal, incompatible_keys = self._verify_state_dict(state_dict)
            assert is_equal, f"loaded state dict contains incompatible keys: {incompatible_keys}"

        # for attr_name, attr_value in state_dict.items():
        #     if attr_name in self.__dict__:
        #         self.__dict__[attr_name] = attr_value

        self.current = state_dict["current"]
        self.length = state_dict["length"]

    def _verify_state_dict(self, state_dict: Dict[str, Any]) -> Tuple[bool, Set[str]]:
        self_keys = set(self.__dict__.keys())
        state_dict_keys = set(state_dict.keys())
        incompatible_keys = self_keys.union(state_dict_keys) - self_keys.intersection(state_dict_keys)
        is_equal = (len(incompatible_keys) == 0)

        return is_equal, incompatible_keys

    def _update_step(self, is_step: bool):
        if is_step:
            self.current += 1

class LinearRampUp(RampUp):
    def __call__(self, current: Optional[int] = None, is_step: bool = True) -> float:
        if current is not None:
            self.current = current

        if self.current >= self.length:
            ramp_up = 1.0
        else:
            ramp_up = self.current / self.length

        self._update_step(is_step)

        return ramp_up

# alpha=0.999
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_logits-target_logits)**2
    return mse_loss


class UnsupervisedLocalUpdate(Client):
    def __init__(self,client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num)
        if self.args.model == 'SVHN_model':
            net = SVHN_model(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            net_ema = SVHN_model(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        else:
            net = ResNet18(args=args, num_classes=self.args.model_output_dim, image_size=32,
                             model_input_channels=self.args.model_input_channels)
            net_ema = ResNet18(args=args, num_classes=self.args.model_output_dim, image_size=32,
                             model_input_channels=self.args.model_input_channels)

        self.ema_model = net_ema.to(self.device)
        self.model = net.to(self.device)

        for param in self.ema_model.parameters():
            param.detach_()

        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.softmax = nn.Softmax()
        self.max_grad_norm = 5
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9,
                                             weight_decay=5e-4)

        self.max_warmup_step = round(self.local_sample_number / self.args.batch_size) * 1
        self.ramp_up = LinearRampUp(length=self.max_warmup_step)
    #     self.update_flag = True
    #     self._local_unsu_dl()
    #
    # def _local_unsu_dl(self):
    #     dl, ds = get_dataloader(self.train_ori_data, self.train_ori_targets, self.args.dataset, self.args.batch_size,
    #                             is_labeled=False)
    #     self.local_unsu_dl = dl


    def train(self, net_w, epoch, target_matrix):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.train()
        self.ema_model.eval()

        self.model.to(self.device)
        self.ema_model.to(self.device)


        if self.flag:
            self.ema_model.load_state_dict(copy.deepcopy(net_w))
            self.flag = False
            logging.info('EMA model initialized')

        print('begin training')
        loss_avg = AverageMeter()
        iter_max = len(self.local_unlabeled_dl_PC)
        logging.info('Begin unsupervised training')
        logging.info('\n=> FedIRM UNSUP Training Epoch #%d, LR=%.4f' % (0, self.optimizer.param_groups[0]['lr']))

        for i, (image_batch, ema_image_batch) in enumerate(self.local_unlabeled_dl_PC):

            image_batch, ema_image_batch = image_batch.to(self.device), ema_image_batch.to(self.device)

            ema_inputs = ema_image_batch
            inputs = image_batch
            batch_size = image_batch.size(0)
            _, outputs = self.model(inputs)

            with torch.no_grad():
                ema_activations, ema_output = self.ema_model(ema_inputs)
            T = 1

            with torch.no_grad():
                _, logits_sum = self.model(inputs)
                for i in range(T):
                    _, logits = self.model(inputs)
                    logits_sum = logits_sum + logits
                logits = logits_sum / (T + 1)
                preds = F.softmax(logits, dim=1)
                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1)
                uncertainty_mask = (uncertainty < 2.2)


            with torch.no_grad():
                activations = F.softmax(outputs, dim=1)
                confidence, _ = torch.max(activations, dim=1)
                confidence_mask = (confidence >= 0.15)

            mask = confidence_mask * uncertainty_mask
            if sum(mask) <= 0:
                continue
            pseudo_labels = torch.argmax(activations[mask], dim=1)
            pseudo_labels = F.one_hot(pseudo_labels, num_classes=self.args.num_classes)
            source_matrix = get_confuse_matrix(outputs[mask], pseudo_labels, self.args.num_classes, self.device)

            consistency_weight = get_current_consistency_weight(self.epoch)
            consistency_dist = torch.sum(softmax_mse_loss(outputs, ema_output)) / batch_size
            consistency_loss = consistency_dist

            loss = 15 * consistency_weight * consistency_loss + 15 * consistency_weight * torch.sum(
                kd_loss(source_matrix, target_matrix))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            update_ema_variables(self.model, self.ema_model, 0.99, self.iter_num)

            self.iter_num = self.iter_num + 1
            loss_avg.update(loss.item(), batch_size)

            # timestamp = get_timestamp()

        self.epoch = self.epoch + 1

        logging.info("client {} updated".format(self.client_index))
        log_info('scalar', 'FedIRM_{role}_{index}_train_loss'.format(role='client', index=self.client_index),
                 loss_avg.avg, 0, self.args.record_tool)
        return self.model.state_dict(), loss_avg.avg, copy.deepcopy(self.optimizer.state_dict())
