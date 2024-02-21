
from torch import nn
from model.SSFL_ResNet18 import ResNet18
from ssl_model.BYOL import MLP


class SimCLRModel(nn.Module):
    def __init__(self, net,
                 image_size=32,
                 projection_size=2048,
                 projection_hidden_size=4096):
        super().__init__()

        self.online_encoder = net
        self.online_encoder.fc = MLP(net.feature_dim, projection_size, projection_hidden_size)  # projector

    def forward(self, image):
        return self.online_encoder(image)

