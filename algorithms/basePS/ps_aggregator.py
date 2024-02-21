
import logging
import time
from collections import defaultdict

import matplotlib
import platform

import torch
from utils.set import *
from utils.log_info import *
from utils.tool import *

sysstr = platform.system()
if ("Windows" in sysstr):
    matplotlib.use("TkAgg")
    logging.info("On Windows, matplotlib use TkAgg")
else:
    matplotlib.use("Agg")
    logging.info("On Linux, matplotlib use Agg")

from utils.data_utils import (
    average_named_params,
    get_avg_num_iterations
)
from transform_lib import *
from algorithms_standalone.basePS import strategies

class PSAggregator(object):
    def __init__(self, train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device, args, model_trainer):

        self.upload_info = {}

        self.trainer = model_trainer
        # preparation for global data
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.pre_model_parms = self.get_global_model_params()

        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.grad_dict = dict()
        self.sample_num_dict = dict()
        self.client_label_flag = dict()
        self.added_idx_list = []

        self.class_centroids = {i:np.zeros(2048) for i in range(self.args.num_classes)}
        # Saving the client_other_params of clients
        self.client_other_params_dict = dict()


        # this flag_client_model_uploaded_dict flag dict is commonly used by gradient and model params
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):  # 标记所有client是否更新过model
            self.flag_client_model_uploaded_dict[idx] = False

        self.selected_clients = None
        # ====================================
        # self.perf_timer = perf_timer
        # ====================================
        self.global_num_iterations = self.get_num_iterations()  # 平均需要的iterations数 = 总数据量/work数量/batch_size=每个work平均数据量/batch_size

        if self.args.pretrained:  # 加载与训练模型
            if self.args.model == "inceptionresnetv2":
                pass
            else:
                ckt = torch.load(self.args.pretrained_dir)
                if "model_state_dict" in ckt:
                    self.trainer.model.load_state_dict(ckt["model_state_dict"])
                else:
                    logging.info(f"ckt.keys: {list(ckt.keys())}")
                    self.trainer.model.load_state_dict(ckt)
        # ================================================
    def compressed_global_model_param(self):
        # TODO compression algorithm
        if self.args.model == 'SemiFed_BYOL':
            return self.trainer.get_SemiFed_BYOL_params()
        else:
           return self.trainer.get_model_params()


    def _reset_queue(self):  # clear all queue like dict and list after one round Aggregation
        self.model_dict = dict()
        self.grad_dict = dict()
        self.sample_num_dict = dict()
        self.client_label_flag = dict()
        self.added_idx_list = []

        # Saving the client_other_params of clients
        self.client_other_params_dict = dict()


        # this flag_client_model_uploaded_dict flag dict is commonly used by gradient and model params
        self.flag_client_model_uploaded_dict = dict()

    # got it
    def get_num_iterations(self):
        # return 20
        return get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)


    def feature_inference(self, loader, model, device):
        feature_vector = []
        labels_vector = []
        model.eval()
        model.to(device)
        for step, (x, y) in enumerate(loader):
            x = x.to(device)

            # get encoding
            with torch.no_grad():
                h = model(x)

            h = h.squeeze()
            h = h.detach()

            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(y.numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        print("Features shape {}".format(feature_vector.shape))
        return feature_vector, labels_vector

    def get_features(self, model, train_loader, test_loader, device):
        train_X, train_y = self.feature_inference(train_loader, model, device)
        test_X, test_y = self.feature_inference(test_loader, model, device)
        return train_X, train_y, test_X, test_y

    def create_data_loaders_from_arrays(self, X_train, y_train, X_test, y_test, batch_size):
        train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=False
        )

        test = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    def test_result(self, test_loader, logreg, device):
        # Test fine-tuned model
        print("### Calculating final testing performance ###")
        logreg.eval()
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(test_loader):
            h = h.to(device)
            y = y.to(device)

            outputs = logreg(h)

            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Accuracy/test"].append(accuracy)


        for k, v in metrics.items():
            print(f"{k}: {np.array(v).mean():.4f}")
        logging.info(np.array(metrics["Accuracy/test"]).mean())



    # got it
    def get_global_model_params(self):
        return self.trainer.get_model_params()

    # got it
    def get_global_generator(self):
        return self.trainer.get_generator()

    # got it 设置global模型的参数
    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)


    def clear_grad_params(self):
        self.trainer.clear_grad_params()

    def update_model_with_grad(self):
        self.trainer.update_model_with_grad()


    # got it index means client_index-----------------------using----------------#
    def add_local_trained_result(self, index, model_params, model_indexes, sample_num, label_flag,
                                 client_other_params=None):
        logging.info("add_model. index = %d" % index)
        self.added_idx_list.append(index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.client_label_flag[index] = label_flag
        self.client_other_params_dict[index] = client_other_params
        self.flag_client_model_uploaded_dict[index] = True

    def get_global_model(self):
        return self.trainer.get_model()

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):  # 更新过的话全部重置
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    # got it sample client
    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        # logging.info("Client Sample Probability: %s "%(str(p)))
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            # make sure for each comparison, we are selecting the same clients each round
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            # else:
            #     raise NotImplementedError

        
        self.selected_clients = client_indexes
        return client_indexes

    # got it
    def test_on_server_for_all_clients(self, epoch, tracker=None, metrics=None):
        logging.info("################test_on_server_for_all_clients : {}".format(epoch))
        avg_acc = self.trainer.test(epoch, self.test_dataloader, self.device)
        return avg_acc

    def test_on_server_for_round(self, round,test_dataloader):
        logging.info("################test_on_server_for_all_clients : {}".format(round))
        avg_acc = self.trainer.test_on_server_for_round(round, test_dataloader, self.device)
        return avg_acc


    def get_average_weight_dict(self, sample_num_list):
        '''
        Args:
            sample_num_list:

        Returns:
            average_weights_list
        '''
        sum = 0
        average_weights = []
        for i in range(len(sample_num_list)):
            sum += sample_num_list[i]
        for i in range(len(sample_num_list)):
            average_weights.append(sample_num_list[i] / sum)
        return average_weights

    def download_info_from_selected_clients(self,upload_info):
        # upload_info = {'MODELS_PARAMS':{clinet_index:params},
        #                'CLIENTS_WEIGHTS':{client_index:sample_num}        }

        self.upload_info = upload_info

    def aggregation(self):

        weights = self.upload_info['CLIENTS_WEIGHTS']
        models_params = self.upload_info['MODELS_PARAMS']

        if len(weights) == 0:
            return

        if self.args.model == 'SemiFed_BYOL':

            logging.info("updata global model in SemiFed_BYOL")
            online_encoders = [param_dict['UNSUP_ONLINE_ENCODER'] for param_dict in models_params]
            online_encoder = self._federated_averaging_by_params(online_encoders, weights)

            online_predictors = [param_dict['UNSUP_ONLINE_PREDICTOR'] for param_dict in models_params]
            online_predictor = self._federated_averaging_by_params(online_predictors, weights)


            sup_models = [param_dict['SUP_MODEL'] for param_dict in models_params]
            sup_model = self._federated_averaging_by_params(sup_models, weights)

            self.trainer.set_SemiFed_BYOL_model_for_gloabl(online_encoder, online_predictor, sup_model)
        else:
            logging.info("updata global model")
            model_params = self._federated_averaging_by_params(models_params, weights)
            self.trainer.set_model_params(model_params)

        # centroid_weights = self.upload_info['CLASS_CENTROID_WEIGHTS']
        # centroids = self.upload_info['CLASS_CENTROIDS']
        #
        # self.update_centroids(centroids, centroid_weights)



    def _federated_averaging_by_models(self, model, weights):
        pass

    def _federated_averaging_by_params(self, models_params, weights):
        model_params = strategies.federated_averaging_by_params(models_params, weights)
        return model_params

    def aggregate(self):
        '''
        return:
        @averaged_params:
        @global_other_params:
        @shared_params_for_simulation:
        '''
        start_time = time.time()
        model_list = []
        training_num = 0

        global_other_params = {}
        shared_params_for_simulation = {}

        # --------FedAvg   using  this process------------#
        logging.info("Server is averaging model or adding grads!!")
        # for idx in range(self.worker_num):
        sample_num_list = []
        client_other_params_list = []
        logging.info("Agg client idx in this round is %s" % str(self.added_idx_list))
        for idx in self.added_idx_list:
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))  # (training_data_num, model)
            sample_num_list.append((self.sample_num_dict[idx], self.client_label_flag[idx]))
            if idx in self.client_other_params_dict:
                client_other_params = self.client_other_params_dict[idx]
            else:
                client_other_params = {}
            client_other_params_list.append(client_other_params)
            training_num += self.sample_num_dict[idx]

        logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))
        logging.info("Aggregator: using average type: {} ".format(
            self.args.fedavg_avg_weight_type
        ))

        average_weights_dict_list, homo_weights_list = self.get_average_weight_dict(
            sample_num_list=sample_num_list)

        averaged_params = average_named_params(
            model_list,  # from sampled client model_lis t  [(sample_number, model_params)]
            average_weights_dict_list
        )

    def save_server_model(self,path):
        torch.save(self.trainer.model.SimCLR.online_encoder.cpu().state_dict(), path)

    def update_centroids(self, client_centroids, weights):
        for i in range(self.args.num_classes):
            class_i_centroid = torch.zeros(self.class_centroids[i].shape)
            sum = 0
            for i_client, client_centroid in enumerate(client_centroids):
                if i in client_centroid.keys():
                    sum += weights[i_client]
                    class_i_centroid += client_centroid[i_client] * weights[i_client]

            class_i_centroid /= sum
            print(class_i_centroid)

            self.class_centroids[i] = class_i_centroid


    def get_class_centroid(self):
        return self.class_centroids


