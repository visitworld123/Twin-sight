import torch.nn as nn
import torch.nn.functional as F



class SVHN_model(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SVHN_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l1 = nn.Linear(84, 84)
        self.dropout = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(84, 256)

        # last layer
        self.l3 = nn.Linear(256, output_dim)
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.squeeze()
        x = self.l1(x)

        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        # x = self.fc3(x)
        return x, y