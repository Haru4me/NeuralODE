"""
    EXPERIMENT
"""

from tools.model import ODEF
from tools.data import Data
import argparse
import logging.config
from traceback import format_exc

import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset

from tools.settings import LOGGING_CONFIG
from tools.tools import experiment
from tools.metrics import *

import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('seaborn')


SOLVERS = ['euler', 'rk4']

CRITERION = {
    'MSE': nn.MSELoss,
    'MAE': nn.L1Loss,
    'Smooth MAE': nn.SmoothL1Loss
}

OPTIM = {
    'Adam': torch.optim.Adam,
    'SGD': None
}

METRICS = {
    'R2Score': R2Score,
    'MAPE': MAPE,
    'WAPE': WAPE
}

ACTIVATION = {
    'Tanh': nn.Tanh,
    'Tanhshrink': nn.Tanhshrink,
    'Sigmoid': nn.Sigmoid,
    'LogSigmoid': nn.LogSigmoid,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'CELU': nn.CELU,
    'GELU': nn.GELU
}

parser = argparse.ArgumentParser(prog='NeuralODE soil experiment',
                                 description="""Скрипт запускает эксперимент №1""",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-adjoint', action='store_true', help='Использовать adjoint_odeint или дефолтный')
parser.add_argument('-lr', type=float, default=0.01, help='Скорость обучения')
parser.add_argument('-batch_size', type=int, default=32, help='Размер батча')
parser.add_argument('-interval', type=int, default=1, help='Интервал для валидации и чекпоинта')
parser.add_argument('-n', '--name', type=str, required=True, help='Название эксперимента')
parser.add_argument('-m', '--method', type=str, choices=SOLVERS, default='euler', help='Выбор метода решения ОДУ')
parser.add_argument('-lf', '--loss', type=str, choices=CRITERION.keys(), default='MSE', help='Выбор функции потерь')
parser.add_argument('-mx', '--metric', type=str, choices=METRICS.keys(), default='R2Score', help='Выбор метрики')
parser.add_argument('-opt', '--optim', type=str, choices=OPTIM.keys(), default='Adam', help='Выбор меотда оптимизации')
parser.add_argument('-e', '--num_epoch', type=int, default=250, help='Количество эпох')
parser.add_argument('--n_layes', type=int, default=1, help='Кол-во скрытых весов')
parser.add_argument('--act_fun', type=str, choices=ACTIVATION.keys(), default='Tanh', help='Функция активации')
opt = parser.parse_args()

Path(f'logs/').mkdir(exist_ok=True)

LOGGING_CONFIG['handlers']['file_handler']['filename'] = f'logs/{opt.name}.log'

if opt.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    try:

        logger.info(f'Start {opt.name}')

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        Path(f'logs/').mkdir(exist_ok=True)
        Path(f"assets/{opt.name}").mkdir(exist_ok=True)
        Path(f"assets/{opt.name}/imgs").mkdir(exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        criterion = CRITERION[opt.loss]().to(device)
        metric = METRICS[opt.metric]()
        n_layers = opt.n_layes
        act_fun = ACTIVATION[opt.act_fun]
        func = ODEF(8, n_layers, act_fun).to(device)
        optimizer = OPTIM[opt.optim](func.parameters(), lr=opt.lr)
        dataloader = DataLoader(Data(), batch_size=opt.batch_size)
        val = DataLoader(Data(val=True), batch_size=opt.batch_size)

        experiment(odeint, func, dataloader, val, optimizer, criterion, metric, opt, LOGGING_CONFIG, streamlit=False)

    except Exception as exp:

        err = format_exc()
        logger.error(err)
        raise(exp)

    logger.info(f'End {opt.name}')
