from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from RSCFed.utils_SimPLE import label_guessing, sharpen
import logging

from algorithms_standalone.basePS.client import Client
from model.SSFL_ResNet18 import ResNet18

from typing import Dict, Any, Set, Tuple, Optional
from abc import ABC, abstractmethod
from RSCFed.rscfed_dataset import *
from utils.set import *
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
                local_unlabeled_ds_PD, local_ds_pesudolabel_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets,  train_data_num,
                local_labeled_ds_PC, local_unlabeled_ds_PC, local_labeled_ds_PD,
                local_unlabeled_ds_PD, local_ds_pesudolabel_PD, train_cls_counts_dict, device, args, model_trainer, dataset_num)
        
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
        self.unsup_lr = self.args.lr
        self.softmax = nn.Softmax()
        self.max_grad_norm = 5

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9,
                                             weight_decay=5e-4)

        self.max_warmup_step = round(self.local_sample_number / self.args.batch_size) * 1
        self.ramp_up = LinearRampUp(length=self.max_warmup_step)


    def train(self, net_w,  unlabeled_idx):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.train()
        self.ema_model.eval()

        self.model.to(self.device)
        self.ema_model.to(self.device)


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.unsup_lr

        if self.flag:
            self.ema_model.load_state_dict(copy.deepcopy(net_w))
            self.flag = False
            logging.info('EMA model initialized')

        epoch_loss = AverageMeter()
        logging.info('Unlabeled client %d begin unsupervised training' % unlabeled_idx)
        all_pseu = 0
        test_right = 0
        test_right_ema = 0
        train_right = 0
        same_total = 0
        for epoch in range(self.args.global_epochs_per_round):
            batch_loss = []
            loss_avg = AverageMeter()
            ramp_up_value = self.ramp_up(current=self.epoch)
            for i, (weak_aug_batch1, weak_aug_batch2) in enumerate(self.local_unlabeled_dl_PC):
                weak_aug_batch1, weak_aug_batch2 = weak_aug_batch1.to(self.device), weak_aug_batch2.to(self.device)
                bs = weak_aug_batch1.size(0)
                with torch.no_grad():
                    guessed = label_guessing(self.ema_model, [weak_aug_batch1])
                    sharpened = sharpen(guessed).to(self.device)

                pseu = torch.argmax(sharpened, dim=1).to(self.device)


                all_pseu += len(pseu[torch.max(sharpened, dim=1)[0] > 0.9])

                _, logits = self.model(weak_aug_batch2)
                probs_str = F.softmax(logits, dim=1)
                pred_label = torch.argmax(logits, dim=1)

                same_total += sum([pred_label[sam] == pseu[sam] for sam in range(logits.shape[0])])

                loss_u = torch.sum(softmax_mse_loss(probs_str, sharpened)) / self.args.batch_size

                loss = ramp_up_value * 0.01 * loss_u
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()

                update_ema_variables(self.model, self.ema_model, 0.999, self.iter_num)

                batch_loss.append(loss.item())
                loss_avg.update(loss.item(),bs)
                self.iter_num = self.iter_num + 1

            epoch_loss.update(loss_avg.avg,1)
            self.epoch = self.epoch + 1
        self.model.cpu()
        self.ema_model.cpu()
        return self.model.state_dict(), self.ema_model.state_dict(), epoch_loss.avg, copy.deepcopy( \
            self.optimizer.state_dict()), ramp_up_value, all_pseu, test_right, test_right_ema, same_total
