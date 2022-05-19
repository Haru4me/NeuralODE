"""
ODEF samples file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearODEF(nn.Module):

    def __init__(self, layers, embs, func):

        super().__init__()

        def init_weights(m):

            if isinstance(m, nn.Linear):
                a = 1/np.sqrt(m.in_features)
                m.weight.data.uniform_(-a, a)
                m.bias.data.fill_(0)

        self.norm = nn.BatchNorm1d(13, affine=False)
        self.reg = nn.Linear(13, 2).apply(init_weights)

    def forward(self, t, x):
        return self.reg(self.norm(x))


class Embedding(nn.Module):

    def __init__(self, num, weight):

        super().__init__()
        self.weight = torch.tensor(weight)
        self.num = num

    def forward(self, x):
        return torch.matmul(F.one_hot(x.long(), num_classes=self.num).float(), self.weight)

class EmbededLinearODEF(nn.Module):

    def __init__(self, layers, embs, func, mean, std):

        super().__init__()

        def init_weights(m):

            if isinstance(m, nn.Linear):
                a = 1/np.sqrt(m.in_features)
                m.weight.data.uniform_(-a, a)
                m.bias.data.fill_(0)

        self.cult_emb = nn.Embedding(7, embs[0])
        self.soil_emb = nn.Embedding(7, embs[1])
        self.cover_emb = nn.Embedding(8, embs[2])
        self.mean = mean
        self.std = std
        self.reg = nn.Linear(10+10, 2).apply(init_weights)

    def forward(self, t, x):

        e1 = x[:, -3].long()
        e2 = x[:, -2].long()
        e3 = x[:, -1].long()
        x = x[:, :-3]

        e1 = self.cult_emb(e1)
        e2 = self.soil_emb(e2)
        e3 = self.cover_emb(e3)

        x = (x-self.mean)/self.std
        x = torch.cat((x, e1, e2, e3), dim=-1)

        return self.reg(x)

class Layer(nn.Module):

    def __init__(self, inp, outp, func, output=False):

        super().__init__()

        layer = [nn.Linear(inp, outp)]

        if not output: layer.append(func())

        self.layer = nn.Sequential(*layer)


    def forward(self, x):
        return self.layer(x)


class MultyLayerODEF(nn.Module):

    def __init__(self, layers, embs, func, mean, std):

        super().__init__()

        net = [Layer(10+10, layers[0], func)]

        if len(layers) > 1:
            for i in range(1, len(layers)):
                net.append(Layer(layers[i-1], layers[i], func))
        else:
            i = 0

        net.append(Layer(layers[i], 2, func, output=True))

        self.net = nn.Sequential(*net)
        self.cult_emb = nn.Embedding(7, embs[0])
        self.soil_emb = nn.Embedding(7, embs[1])
        self.cover_emb = nn.Embedding(8, embs[2])
        self.mean = mean
        self.std = std

    def forward(self, t, x):

        e1 = x[:, -3].long()
        e2 = x[:, -2].long()
        e3 = x[:, -1].long()
        x = x[:, :-3]

        e1 = self.cult_emb(e1)
        e2 = self.soil_emb(e2)
        e3 = self.cover_emb(e3)

        x = (x-self.mean)/self.std
        x = torch.cat((x, e1, e2, e3), dim=-1)
        x = self.net(x)

        return x
