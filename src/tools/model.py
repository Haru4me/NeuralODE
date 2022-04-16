"""
ODEF samples file
"""

import enum
from re import M
from turtle import clear, forward
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

        x = torch.cat((x, e1, e2), dim=-1)

        x = self.input(x)
        x = self.hiden(x)

        return self.output(x)


class LinearODEF(nn.Module):

    def __init__(self):

        super().__init__()

        def init_weights(m):

            if isinstance(m, nn.Linear):
                a = 1/np.sqrt(m.in_features)
                m.weight.data.uniform_(-a, a)
                m.bias.data.fill_(0)

        self.norm = nn.BatchNorm1d(12, affine=False)
        self.reg = nn.Linear(12, 2).apply(init_weights)

    def forward(self, t, x):
        return self.reg(self.norm(x))


class EmbededLinearODEF(nn.Module):

    def __init__(self):

        super().__init__()

        def init_weights(m):

            if isinstance(m, nn.Linear):
                a = 1/np.sqrt(m.in_features)
                m.weight.data.uniform_(-a, a)
                m.bias.data.fill_(0)

        self.soil_emb = nn.Embedding(12, 6)
        self.cover_emb = nn.Embedding(16, 6)
        self.norm = nn.BatchNorm1d(22, affine=False)
        self.reg = nn.Linear(22, 2).apply(init_weights)

    def forward(self, t, x):

        e1 = x[:, -2].long()
        e2 = x[:, -1].long()
        x = x[:, :-2]

        e1 = self.soil_emb(e1)
        e2 = self.cover_emb(e2)

        x = torch.cat((x, e1, e2), dim=-1)
        x = self.normalize(x)

        return self.reg(x)


class Embeddings(nn.Module):

    def __init__(self, embs):

        super().__init__()

        self.soil_emb = nn.Embedding(12, embs[0])
        self.cover_emb = nn.Embedding(16, embs[1])
        self.normalize = nn.BatchNorm1d(sum(embs))

    def forward(self, e1, e2):

        e1 = self.soil_emb(e1)
        e2 = self.cover_emb(e2)
        e = torch.cat((e1, e2), dim=-1)
        e = self.normalize(e)

        return e

class Layer(nn.Module):

    def __init__(self, inp, outp, func, output=False):

        super().__init__()

        layer = [nn.Linear(inp, outp)]

        if not output: layer.append(func())

        self.layer = nn.Sequential(*layer)


    def forward(self, x):
        return self.layer(x)

class NewODEF(nn.Module):

    def __init__(self, layers, embs, func):

        super().__init__()

        self.embeddings = Embeddings(embs)
        net = [Layer(10+sum(embs), layers[0], func)]

        if len(layers) > 1:
            for i in range(1, len(layers)):
                net.append(Layer(layers[i-1], layers[i], func))
        else:
            i = 0

        net.append(Layer(layers[i], 2, func, output=True))
        self.net = nn.Sequential(*net)

    def forward(self, t, x):

        e1 = x[:, -2].long()
        e2 = x[:, -1].long()
        x = x[:, :-2]

        e = self.embeddings(e1,e2)
        x = torch.cat((x, e), dim=-1)
        x = self.net(x)

        return x
