''' 
The MLP model for the synthetic check assumes the input to be mnist of 28*28 and is a model with 2 hidden layers one of 512 follwed by a 256 unit hidden layer.
'''

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)