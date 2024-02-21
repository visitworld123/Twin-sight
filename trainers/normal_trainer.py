import copy
from collections import deque

import torch
from torch import nn



from utils.data_utils import (
    get_all_bn_params
)
from pseduo_dataset import *


from utils.set import *
from utils.log_info import *
from utils.tool import *
from ssl_model.ssl_utils import *

class NormalTrainer(object):
    def __init__(self, model, device, criterion, optimizer, args, **kwargs):
        # kwargs 可以包含该Trainer的role:'server'/'client'，server_index or client_index

        self.trainer_update_epoch = 0

        self.role = kwargs['role']
        self.trainer_name = None # the name describe the utilization of this trainer
        if self.role == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif self.role == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError


        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer


        # For future use
        self.param_groups = self.optimizer.param_groups  
        


        self.sup_feature_bank = deque(maxlen=3)
        self.unsup_feature_bank = deque(maxlen=3)

    def _set_optimizer(self):
        if self.args.model == 'SemiFed_BYOL':
            params_to_optimizer = [
                    {'params': self.model.sup_classifier.parameters()},
                    {'params': self.model.unsup_model.online_encoder.parameters()},
                    {'params': self.model.unsup_model.online_predictor.parameters()}
                ]
        else:
            params_to_optimizers = self.model.parameters()
        
        self.optimizer = torch.optim.SGD(params_to_optimizer,
                lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.momentum, nesterov=self.args.nesterov)
        
        
    def get_model_named_modules(self):
        return dict(self.model.cpu().named_modules())

    def get_train_batch_data(self, train_local):
        try:
            train_batch_data = next(self.train_local_iter)
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_local)
            train_batch_data = next(self.train_local_iter)
        return train_batch_data
    
    def get_model(self):
        return copy.deepcopy(self.model.cpu())

    def set_SemiFed_BYOL_model_for_gloabl(self,online_encoder, online_predictor, sup_model):
        self.model.unsup_model.online_encoder.load_state_dict(online_encoder)
        self.model.unsup_model.online_predictor.load_state_dict(online_predictor)
        self.model.sup_classifier.load_state_dict(sup_model)

    def get_SemiFed_BYOL_params(self):
        params = {}
        params['UNSUP_ONLINE_ENCODER'] = self.model.unsup_model.online_encoder.cpu().state_dict()
        params['UNSUP_ONLINE_PREDICTOR'] = self.model.unsup_model.online_predictor.cpu().state_dict()
        params['SUP_MODEL'] = self.model.sup_classifier.cpu().state_dict()
        return params

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, params):
        '''
        params: Download from cloud or other place
        '''
        if self.args.model == 'SemiFed_BYOL':
            logging.info("Use aggregated encoder and predictor in SemiFed_BYOL")
            self.model.unsup_model.online_encoder.load_state_dict(params['UNSUP_ONLINE_ENCODER'])
            self.model.unsup_model.online_predictor.load_state_dict(params['UNSUP_ONLINE_PREDICTOR'])
            self.model.sup_classifier.load_state_dict(params['SUP_MODEL'])
        else:
            self.model.load_state_dict(params)

    def set_model_for_local(self, model):
        logging.info("Updata local model")
        self.model = copy.deepcopy(model)


    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

# got it Batch_Normalization set
    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"]
                module.bias.data = all_bn_params[module_name+".bias"]
                module.running_mean = all_bn_params[module_name+".running_mean"]
                module.running_var = all_bn_params[module_name+".running_var"]
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"]


    def get_BYOL_params(self):
        params = {}
        params['ONLINE_ENCODER'] = self.model.online_encoder.cpu().state_dict()
        params['ONLINE_PREDICTOR'] = self.model.online_predictor.cpu().state_dict()
        params['TARGET_ENCODER'] = self.model.target_encoder.cpu().state_dict()
        params['CLASS_PREDICTOR'] = self.model.class_predictor.cpu().state_dict()

        return params


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        return self.optimizer.state



    
    @torch.no_grad()
    def cal_time_p_and_p_model(self,logits_x_ulb_w, time_p, p_model, label_hist):
        prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
        max_probs, max_idx = torch.max(prob_w, dim=-1)
        if time_p is None:
            time_p = max_probs.mean()
        else:
            time_p = time_p * 0.999 +  max_probs.mean() * 0.001
        if p_model is None:
            p_model = torch.mean(prob_w, dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
        if label_hist is None:
            label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist / label_hist.sum()
        else:
            hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
        return time_p,p_model,label_hist

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1) # hard_pseudo_label means real label rather than sofmax
        mask = max_p.ge(self.args.pseudo_label_threshold)
        return hard_pseudo_label, mask



    def test_on_server_for_round(self, round, testloader, device):
        self.model.to(device)
        self.model.eval()

        test_acc_avg = AverageMeter()
        test_loss_avg = AverageMeter()

        total_loss_avg = 0
        total_acc_avg = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                # distribute data to device
                x, y = x.to(device), y.to(device).view(-1, )
                batch_size = x.size(0)
                if self.args.algorithm == 'RSCFed':
                    f, logits = self.model(x)
                elif self.args.algorithm == 'FedSiam':
                    f, logits = self.model(x)
                else:
                    if self.args.model in ['SemiFed_SimCLR', 'SemiFed_SimSiam', 'SemiFed_BYOL']:
                        f, logits= self.model.inference(x)
                    else:
                        f, logits= self.model(x)

                loss = self.criterion(logits, y)
                prec1, _ = accuracy(logits.data, y)

                n_iter = (round - 1) * len(testloader) + batch_idx
                test_acc_avg.update(prec1.data.item(), batch_size)
                test_loss_avg.update(loss.data.item(), batch_size)

                log_info('scalar', '{role}_{index}_test_acc_epoch'.format(role=self.role, index=self.index),
                         test_acc_avg.avg, step=n_iter,record_tool=self.args.record_tool)
                total_loss_avg += test_loss_avg.avg
                total_acc_avg += test_acc_avg.avg
            total_acc_avg /= len(testloader)
            total_loss_avg /= len(testloader)
            log_info('scalar', '{role}_{index}_total_acc_epoch'.format(role=self.role, index=self.index),
                     test_acc_avg.avg, step=round,record_tool=self.args.record_tool)
            log_info('scalar', '{role}_{index}_total_loss_epoch'.format(role=self.role, index=self.index),
                     test_loss_avg.avg, step=round, record_tool=self.args.record_tool)
            logging.info("\n| Server Testing Round #%d\t\tTest Acc: %.4f Test Loss: %.4f" % (round, test_acc_avg.avg, test_loss_avg.avg))
            return total_acc_avg




    def train_semiFed_model_labeled_client_PC(self, round, epoch, dataloader,  device):
        self.model.to(device)
        self.model.train()
        self.model.training = True

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()

        total_loss_avg = AverageMeter()
        instanceDis_loss_avg = AverageMeter()
        sim_loss_avg = AverageMeter()
        ce_loss_avg = AverageMeter()
        train_acc_avg = AverageMeter()

        self.sup_feature_bank.clear()
        self.unsup_feature_bank.clear()
        
        logging.info('\n=> Labeled Client SemiFed Training Epoch #%d, LR=%.4f' % (self.trainer_update_epoch, self.optimizer.param_groups[0]['lr']))
        for batch_idx, (x_w , x_s, y) in enumerate(dataloader):
            real_bs = x_w.size(0)

            x_w, x_s, y = x_w.to(device), x_s.to(device), y.to(device)

            # y = torch.cat((y, y), dim=0)

            self.optimizer.zero_grad()

            # instance discrimination loss
            if self.args.model == 'SemiFed_SimSiam':
                unsup_f, sup_f, unsup_loss, sup_logits = self.model(x_w, x_s)
                instanceDis_loss = unsup_loss

            if self.args.model == 'SemiFed_SimCLR':
                unsup_f, sup_f, unsup_logits, sup_logits = self.model(x_w, x_s)

                # [2*bs], [2*bs], [2*bs], [2*bs]
                logits, labels = info_nce_loss(unsup_logits, real_bs, device)
                instanceDis_loss = criterion1(logits, labels)
            
            if self.args.model == 'SemiFed_BYOL':
                unsup_f_one, unsup_f_two, sup_f, unsup_loss, sup_logits = self.model(x_w, x_s)
                instanceDis_loss = unsup_loss
                unsup_f = unsup_f_one

            ce_loss = criterion2(sup_logits[:real_bs], y)


            unsup_sim_f = unsup_f[:real_bs]
            sup_sim_f = sup_f[:real_bs]
            unsup_sim_f = unsup_sim_f / torch.norm(unsup_sim_f, dim=1, keepdim=True)
            sup_sim_f = sup_sim_f / torch.norm(sup_sim_f, dim=1, keepdim=True)

            unsup_gram_matrix = torch.mm(unsup_sim_f, unsup_sim_f.t())
            sup_gram_matrix = torch.mm(sup_sim_f, sup_sim_f.t())
            gram_loss = torch.sum(torch.pow((unsup_gram_matrix - sup_gram_matrix), 2) / (real_bs ^ 2))
            sim_loss = gram_loss



            loss =  ce_loss + \
                self.args.insDis * instanceDis_loss  + self.args.sim * sim_loss

            loss.backward()
            self.optimizer.step()

            
            total_loss_avg.update(loss.data.item(), real_bs)
            instanceDis_loss_avg.update(instanceDis_loss.data.item(), real_bs)
            sim_loss_avg.update(sim_loss.data.item(), real_bs)

            
            prec1, prec5, correct, pred, _ = accuracy(sup_logits[:real_bs].data, y.data, topk=(1, 5))
            
            
            ce_loss_avg.update(ce_loss.data.item(), real_bs)
            train_acc_avg.update(prec1.data.item(), real_bs)

        self.trainer_update_epoch += 1  # epoch * round

        log_info('scalar', 'SemiFed_{role}_{index}_labeled_total_loss'.format(role=self.role, index=self.index),
                 total_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)
        log_info('scalar', 'SemiFed_{role}_{index}_labeled_instanceDis_loss'.format(role=self.role, index=self.index),
                 instanceDis_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)
        log_info('scalar', 'SemiFed_{role}_{index}_labeled_sim_loss'.format(role=self.role, index=self.index),
                   sim_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)
        log_info('scalar', 'SemiFed_{role}_{index}_labeled_ce_loss'.format(role=self.role, index=self.index),
            ce_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)
        log_info('scalar', 'SemiFed_{role}_{index}_labeled_train_acc'.format(role=self.role, index=self.index),
            train_acc_avg.avg, self.trainer_update_epoch, self.args.record_tool)
    

    def train_semiFed_model_unlabeled_client_PC(self, round, epoch, dataloader,  device):
        self.model.to(device)
        self.model.train()
        self.model.training = True

        criterion1 = nn.CrossEntropyLoss()

        total_loss_avg = AverageMeter()
        instanceDis_loss_avg = AverageMeter()
        sim_loss_avg = AverageMeter()
        ce_loss_avg = AverageMeter()


        self.sup_feature_bank.clear()
        self.unsup_feature_bank.clear()

        p_model = (torch.ones(self.args.num_classes) / self.args.num_classes).to(device)
        label_hist = (torch.ones(self.args.num_classes) / self.args.num_classes).to(device)
        time_p = p_model.mean()
        logging.info('\n=> UnLabeled Client SemiFed Training Epoch #%d, LR=%.4f' % (self.trainer_update_epoch, self.optimizer.param_groups[0]['lr']))

        for batch_idx, (x_w , x_s) in enumerate(dataloader):
            real_bs = x_w.size(0)

            x_w, x_s = x_w.to(device), x_s.to(device)

            self.optimizer.zero_grad()

            # instance discrimination loss
            if self.args.model == 'SemiFed_SimSiam':
                unsup_f, sup_f, unsup_loss, sup_logits = self.model(x_w, x_s)
                instanceDis_loss = unsup_loss

            if self.args.model == 'SemiFed_SimCLR':
                unsup_f, sup_f, unsup_logits, sup_logits = self.model(x_w, x_s)
                # [2*bs], [2*bs], [2*bs], [2*bs]
                logits, labels = info_nce_loss(unsup_logits, real_bs, device)
                instanceDis_loss = criterion1(logits, labels)

            if self.args.model == 'SemiFed_BYOL':
                unsup_f_one, unsup_f_two, sup_f, unsup_loss, sup_logits = self.model(x_w, x_s)
                instanceDis_loss = unsup_loss
                unsup_f = unsup_f_one


            unsup_sim_f = unsup_f[:real_bs]
            sup_sim_f = sup_f[:real_bs]

            unsup_sim_f = unsup_sim_f / torch.norm(unsup_sim_f, dim=1, keepdim=True)
            sup_sim_f = sup_sim_f / torch.norm(sup_sim_f, dim=1, keepdim=True)

            unsup_gram_matrix = torch.mm(unsup_sim_f, unsup_sim_f.t())
            sup_gram_matrix = torch.mm(sup_sim_f, sup_sim_f.t())
            gram_loss = torch.sum(torch.pow((unsup_gram_matrix - sup_gram_matrix), 2) / (real_bs ^ 2))
            sim_loss = gram_loss



            logits_x_ulb_w, logits_x_ulb_s = sup_logits[real_bs:], sup_logits[:real_bs]
            if self.args.SSL_method == 'fixmatch':
                unsup_loss, mask, select, pseudo_lb = consistency_loss_fixmatch(logits_x_ulb_w,
                                                                 logits_x_ulb_s,
                                                                 self.args.fixmatch_T, 
                                                                 self.args.fixmatch_threshold, 
                                                                 use_hard_labels=True)
            elif self.args.SSL_method == 'freematch':
                time_p, p_model, label_hist = self.cal_time_p_and_p_model(logits_x_ulb_w, time_p, p_model, label_hist)
                unsup_loss, mask = consistency_loss_freematch(self.args.dataset,logits_x_ulb_w,logits_x_ulb_s,
                                                    time_p,p_model,
                                                    use_hard_labels=True)
            ce_loss = unsup_loss

            if self.args.PC_use_pesudo_label: 
                loss = self.args.insDis * instanceDis_loss  + self.args.sim * sim_loss + self.args.PC_ulb_loss_ratio * ce_loss
            else:
                loss = self.args.insDis * instanceDis_loss  + self.args.sim * sim_loss

            loss.backward()
            self.optimizer.step()

            self.unsup_feature_bank.append(unsup_sim_f[:real_bs].detach().data)
            self.sup_feature_bank.append(sup_sim_f[:real_bs].detach().data)

            total_loss_avg.update(loss.data.item(), real_bs)
            instanceDis_loss_avg.update(instanceDis_loss.data.item(), real_bs)
            sim_loss_avg.update(sim_loss.data.item(), real_bs)

            ce_loss_avg.update(ce_loss.data.item(), real_bs)

        self.trainer_update_epoch += 1  # epoch * round

        log_info('scalar', 'SemiFed_{role}_{index}_unlabeled_total_loss'.format(role=self.role, index=self.index),
                 total_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)
        log_info('scalar', 'SemiFed_{role}_{index}_unlabeled_instanceDis_loss'.format(role=self.role, index=self.index),
                 instanceDis_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)
        log_info('scalar', 'SemiFed_{role}_{index}_unlabeled_sim_loss'.format(role=self.role, index=self.index),
                 sim_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)
        log_info('scalar', 'SemiFed_{role}_{index}_unlabeled_ce_loss'.format(role=self.role, index=self.index),
                ce_loss_avg.avg, self.trainer_update_epoch, self.args.record_tool)

    def train_semiFed_model_PD(self, round, epoch, global_params, 
                                     labeled_dataloader, unlabled_dataloader,
                                     device):
        self.model.to(device)
        self.model.train()
        self.model.training = True

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        criterion3 = nn.CrossEntropyLoss()

        total_loss_avg = AverageMeter()
        instanceDis_loss_avg = AverageMeter()
        sim_loss_avg = AverageMeter()
        sup_ce_loss_avg = AverageMeter()
        train_acc_avg = AverageMeter()

        p_model = (torch.ones(self.args.num_classes) / self.args.num_classes).to(device)
        label_hist = (torch.ones(self.args.num_classes) / self.args.num_classes).to(device) 
        time_p = p_model.mean()

        logging.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, self.optimizer.param_groups[0]['lr']))
        for batch_idx, (x_ulb_w, x_ulb_s) in enumerate(unlabled_dataloader):
            labeled_data = self.get_train_batch_data(labeled_dataloader)
            x_lb_w, x_lb_s, y = labeled_data
            x_lb_w, x_lb_s, y = x_lb_w.to(device), x_lb_s.to(device), y.to(device)
            x_ulb_w, x_ulb_s = x_ulb_w.to(device), x_ulb_s.to(device)

            lb_data_bs = x_lb_w.size(0)
            ulb_data_bs = x_ulb_w.size(0)
            inputs_bs = lb_data_bs + ulb_data_bs

            self.optimizer.zero_grad()
            inputs_w = torch.cat((x_lb_w, x_ulb_w))
            inputs_s = torch.cat((x_lb_s, x_ulb_s))
            
            if self.args.model == 'SemiFed_SimCLR':
                unsup_f, sup_f, unsup_logits, sup_logits = self.model(inputs_w, inputs_s)

                instance_logits, instance_labels = info_nce_loss(unsup_logits, inputs_bs, device)
                instanceDis_loss = criterion1(instance_logits, instance_labels)


            logits_x_lb = sup_logits[:lb_data_bs]

            sup_loss = criterion2(logits_x_lb, y)
            logits_x_ulb_w, logits_x_ulb_s = sup_logits[lb_data_bs:inputs_bs], \
                                            sup_logits[inputs_bs + lb_data_bs:]
            
            if self.args.SSL_method == 'fixmatch':
                unsup_loss, mask, select, pseudo_lb = consistency_loss_fixmatch(logits_x_ulb_w,
                                                                 logits_x_ulb_s,
                                                                 self.args.fixmatch_T, 
                                                                   self.args.fixmatch_threshold, 
                                                                   use_hard_labels=True)
            elif self.args.SSL_method == 'freematch':
                time_p, p_model, label_hist = self.cal_time_p_and_p_model(logits_x_ulb_w, time_p, p_model, label_hist)
                unsup_loss, mask = consistency_loss_freematch(self.args.dataset,logits_x_ulb_w,logits_x_ulb_s,
                                                    time_p,p_model,
                                                    use_hard_labels=True)
            
            unsup_sim_f = unsup_f[:inputs_bs]
            sup_sim_f = sup_f[:inputs_bs]

            unsup_sim_f = unsup_sim_f / torch.norm(unsup_sim_f, dim=1, keepdim=True)
            sup_sim_f = sup_sim_f / torch.norm(sup_sim_f, dim=1, keepdim=True)
            unsup_gram_matrix = torch.mm(unsup_sim_f, unsup_sim_f.t())
            sup_gram_matrix = torch.mm(sup_sim_f, sup_sim_f.t())
            gram_loss = torch.sum(torch.pow((unsup_gram_matrix - sup_gram_matrix), 2) / (inputs_bs ^ 2))
            sim_loss = gram_loss
            loss = sup_loss  + self.args.PD_ulb_loss_ratio * unsup_loss + \
                self.args.insDis * instanceDis_loss  + self.args.sim * sim_loss
            # ========================FedProx=====================#
            if self.args.fedprox:
                fed_prox_reg = 0.0
                for name, param in self.model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu ) * \
                        torch.norm((param - global_params[name].data.to(device)))**2)
                loss += fed_prox_reg
            # ========================FedProx=====================#
            loss.backward()
            self.optimizer.step()

            prec1, prec5, correct, pred, _ = accuracy(sup_logits[:lb_data_bs].data, y.data, topk=(1, 5))
            # unsup_prec1, unsup_prec5, unsup_correct, unsup_pred, _ = accuracy(unsup_cls_logits[:lb_data_bs].data, y.data, topk=(1, 5))
            total_loss_avg.update(loss.data.item(), inputs_bs)
            instanceDis_loss_avg.update(instanceDis_loss.data.item(), inputs_bs)
            sim_loss_avg.update(sim_loss.data.item(), inputs_bs)
            sup_ce_loss_avg.update(sup_loss.data.item(), lb_data_bs)
            train_acc_avg.update(prec1.data.item(), lb_data_bs)

            # loss_avg.update(loss.data.item(), batch_size)
            # acc.update(prec1.data.item(), batch_size)

            n_iter = (epoch - 1) * len(unlabled_dataloader) + batch_idx

            log_info('scalar','{role}_{index}_train_total_loss_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     total_loss_avg.avg, n_iter, self.args.record_tool)
            log_info('scalar','{role}_{index}_train_sup_loss_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     sup_ce_loss_avg.avg, n_iter, self.args.record_tool)
            log_info('scalar','{role}_{index}_train_instanceDis_loss_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     instanceDis_loss_avg.avg, n_iter, self.args.record_tool)
            log_info('scalar','{role}_{index}_train_acc_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     train_acc_avg.avg, n_iter, self.args.record_tool)
            log_info('scalar','{role}_{index}_sim_loss_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     sim_loss_avg.avg, n_iter, self.args.record_tool)



def consistency_loss_fixmatch(logits_w, logits_s, T=1.0, p_cutoff=0.0, use_hard_labels=True):
    
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    select = max_probs.ge(p_cutoff).long()

    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask

    return masked_loss.mean(), mask.mean(), select, max_idx.long()

def consistency_loss_freematch(dataset, logits_w, logits_s, time_p, p_model, use_hard_labels=True):

    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    p_cutoff = time_p
    p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]
    threshold = p_cutoff * p_model_cutoff[max_idx]
    if dataset == 'SVHN':
        threshold = torch.clamp(threshold, min=0.9, max=0.95)
    mask = max_probs.ge(threshold)
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()
    else:
        pseudo_label = torch.softmax(logits_w / 0.5, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
    return masked_loss.mean(), mask

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def compute_lr(current_round, rounds=100, eta_min=0, eta_max=0.3):
    """Compute learning rate as cosine decay"""
    pi = np.pi
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * current_round / rounds) + 1)
    return eta_t


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = 0.01
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    lr = linear_rampup(epoch, 0.0) * (0.01 - 0.0) + 0.0
    if 500:
        lr *= cosine_rampdown(epoch, 1000)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def get_current_consistency_weight(epoch):
    return sigmoid_rampup(epoch, 10)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def network_add(diff_w_ema_1, diff_w_ema_2):
    diff_w_ema = {}
    for key in diff_w_ema_1.keys():
        diff_w_ema[key] = torch.sqrt(diff_w_ema_1[key] + diff_w_ema_1[key]) / torch.sqrt(diff_w_ema_2[key] + diff_w_ema_2[key])

    return diff_w_ema
