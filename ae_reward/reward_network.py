import torch
import torch.nn as nn

import pdb

class RNet(nn.Module):
    # init
    def __init__(self, gpu=False):
        super(RNet, self).__init__()

        self.gpu = gpu
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)
        )
        self.predictor = nn.Linear(64, 1)

    # process
    def forward(self, x):
        x = self.net(x)
        r = self.predictor(x)
        return r