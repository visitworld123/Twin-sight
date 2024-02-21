
from torch import nn
from model.SSFL_ResNet18 import ResNet18
from ssl_model.BYOL import MLP
import torch



def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class SemiFed(nn.Module):
    def __init__(self, unsup_model,
                 sup_model, args,
                 image_size=32, num_class=10, input_channel=3):
        super().__init__()
        self.unsup_model = unsup_model
        self.sup_classifier = sup_model

        self.args = args
        self.image_size = image_size

        self.output_dim  = num_class
        self.num_class = num_class
        self.input_channel = input_channel

    def forward(self, image1, image2):
        images = torch.cat((image1, image2), dim=0)
        if self.args.model == 'SemiFed_SimSiam':
            unsup_loss, unsup_f = self.unsup_model(image1, image2)
            sup_f, sup_logits = self.sup_classifier(images)

            return unsup_f, sup_f, unsup_loss, sup_logits
        
        if self.args.model == 'SemiFed_SimCLR':
            unsup_f, unsup_logits = self.unsup_model(images)
            sup_f, sup_logits = self.sup_classifier(images)
            return unsup_f, sup_f, unsup_logits, sup_logits
        
        if self.args.model == 'SemiFed_BYOL':
            unsup_f_one, unsup_f_two, unsup_loss = self.unsup_model(image1, image2)

            sup_f, sup_logits = self.sup_classifier(images)

            return unsup_f_one, unsup_f_two, sup_f, unsup_loss, sup_logits
        


        return

    def inference(self, x):

        f, logits = self.sup_classifier(x)

        return f, logits


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new