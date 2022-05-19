from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from pathlib import Path

from src.tools.model import MultyLayerODEF, LinearODEF, EmbededLinearODEF
from src.tools.data import DataNPZ
from src.torchdiffeq import odeint
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

warnings.filterwarnings("ignore")
plt.style.use('classic')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

paths = {}
paths['stats'] = list(Path('assets').rglob('*.csv'))
paths['models'] = list(Path('assets').rglob('*.pt'))

def draw_losses(conf: pd.DataFrame, mod: str, metric: str):

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)

    for name in cnf.index:

        stat = pd.read_csv(list(Path(f'assets/{name}').rglob('*.csv'))[0])

        if cnf.loc[name].lf == 'MSE':
            ax[0].plot(stat.val_loss_1**0.5, label=cnf.loc[name].lf)
            ax[1].plot(stat.val_loss_2**0.5, label=cnf.loc[name].lf)
        else:
            ax[0].plot(stat.val_loss_1, label=cnf.loc[name].lf)
            ax[1].plot(stat.val_loss_2, label=cnf.loc[name].lf)

        for i,axx in enumerate(ax):

            axx.legend()
            axx.grid(True)
            axx.set_xlabel('Epoch', size=20)
            axx.set_ylabel(f'val_{i+1} {metric}', size=20)

    plt.savefig(f'assets/results/loss_{mod}_{metric}.png', bbox_inches='tight')
    plt.clf()


def draw_metric(conf: pd.DataFrame, mod: str):

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)

    for name in cnf.index:

        stat = pd.read_csv(list(Path(f'assets/{name}').rglob('*.csv'))[0])
        ax[0].plot(stat.train_metric, label=cnf.loc[name].lf)
        ax[0].set_title('Обучающая выборка', size=20)
        ax[1].plot(stat.val_metric, label=cnf.loc[name].lf)
        ax[1].set_title('Валидационная выборка', size=20)

    for i,axx in enumerate(ax):

        axx.legend()
        axx.grid(True)
        axx.set_xlabel('Эпоха', size=20)
        axx.set_ylabel('Метрика', size=20)
        axx.set_ylim(0.5,1)

    plt.savefig(f'assets/results/metric_{mod}.png', bbox_inches='tight')
    plt.clf()


def draw_samples(name: str, path: str, model: nn.Module):

    fig = plt.figure(figsize=(10, 5))
    data = DataLoader(DataNPZ('sample'), batch_size=14)
    v, z0, z1 = next(iter(data))
    t = torch.linspace(0, 5, 80)

    func = model
    func.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
    func.eval()

    pred = odeint(func, z0, v, t, method='rk4')

    p1 = plt.plot(np.linspace(0, 14, 80*14).reshape((14, 80)).T,
             pred.detach().numpy()[:, :, 0], color='b', label='ЗПВ на 10мм')
    p2 = plt.plot(np.linspace(0, 14, 80*14).reshape((14, 80)).T,
             pred.detach().numpy()[:, :, 1], color='g', label='ЗПВ на 20мм')

    y = list(zip(z0[:,0].data.numpy(), z1[:,0].data.numpy()))
    t = np.linspace(0,14,15)
    t = list(zip(t[:-1], t[1:]))
    plt.scatter(t, y, c='b')

    y = list(zip(z0[:,1].data.numpy(), z1[:,1].data.numpy()))
    t = np.linspace(0,14,15)
    t = list(zip(t[:-1], t[1:]))
    plt.scatter(t, y, c='g')

    plt.xlabel('t', size=20)
    plt.ylabel('ЗПВ', size=20)
    plt.grid()

    plt.savefig(f'assets/results/samples/{name}.png', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    with open('exp_config_1.json') as f:
        config = json.load(f)
    config = pd.DataFrame().from_dict(config).T

    for mod in config['mod'].unique():

        cnf = config[(config['mod'] == mod) & (config.lf.isin(['MAE','MSE','SmoothMAE'])) & (config.af == 'Tanh')][['lf']]
        draw_losses(cnf, mod, 'MAE')
        cnf = config[(config['mod'] == mod) & (config.lf.isin(['WAPE','SMAPE'])) & (config.af == 'Tanh')][['lf']]
        draw_losses(cnf, mod, 'MAPE')
        cnf = config[(config['mod'] == mod) & (config.af == 'Tanh')][['lf']]
        draw_metric(cnf, mod)

        cnf = config[(config['mod'] == mod) & (config.af == 'Tanh')]
        for name in cnf.index:
            l = [int(i) for i in cnf.loc[name, 'l'].split(' ')]
            emb = [int(i) for i in cnf.loc[name, 'emb'].split(' ')]
            path = f'assets/{name}/model.pt'
            if mod == 'Linear':
                model = LinearODEF(l,emb, nn.Tanh)
            if mod == 'EmbededLinear':
                model = EmbededLinearODEF(l,emb, nn.Tanh)
            if mod == 'MultyLayer':
                model = MultyLayerODEF(l,emb, nn.Tanh)

            draw_samples(name, path, model)
