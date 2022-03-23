"""
ODEF samples file
"""

from re import M
import torch
import torch.nn as nn

class ODEF(nn.Module):

    def __init__(self, inp, n_layer, func):

        super().__init__()

        net = [nn.Linear(inp+2, 16)]

        for i in range(n_layer):
            net.append(func())
            net.append(nn.Linear(2**(i+4), 2**(i+5)))

        net.append(func())
        net.append(nn.Linear(2**(i+5), 2))

        self.net = nn.Sequential(*net)

    def forward(self, t, x):
        return self.net(x)
