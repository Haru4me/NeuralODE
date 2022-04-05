"""
ODEF samples file
"""

import enum
from re import M
import torch
import torch.nn as nn
import numpy as np


class ODEF(nn.Module):

    def __init__(self, layers, embs, func):

        super().__init__()

        def init_weights(m):

            if isinstance(m, nn.Linear):
                a = 1/np.sqrt(m.in_features)
                m.weight.data.uniform_(-a, a)
                m.bias.data.fill_(0)

        self.input = nn.Sequential(nn.Linear(10+sum(embs), layers[0]), func()).apply(init_weights)
        self.soil_emb = nn.Embedding(100, embs[0])
        self.cover_emb = nn.Embedding(16, embs[1])

        net = []

        if len(layers) > 1:
            for i in range(1, len(layers)):
                net.append(nn.Linear(layers[i-1], layers[i]))
                net.append(func())

        self.hiden = nn.Sequential(*net).apply(init_weights)

        self.output = nn.Linear(layers[i], 2).apply(init_weights)
        self.output.weight.data *= 0.2

    def forward(self, t, x):

        e1 = x[:,-2].long()
        e2 = x[:,-1].long()
        x = x[:,:-2]

        e1 = self.soil_emb(e1)
        e2 = self.cover_emb(e2)

        x = torch.concat((x, e1, e2), dim=-1)

        x = self.input(x)
        x = self.hiden(x)

        return self.output(x)
