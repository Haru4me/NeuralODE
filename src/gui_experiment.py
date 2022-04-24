"""
Эксперимент по обучению NeuralODE на прогнозировании ЗПВ
с использованием odeint
"""

import logging.config
from traceback import format_exc
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.settings import LOGGING_CONFIG
from tools.tools import experiment
from tools.metrics import *

import numpy as np
import pandas as pd
from pathlib import Path

from tools.data import DataNPZ
from tools.model import ODEF


SOLVERS = ['euler', 'rk4']
CRITERION = {
    'MSE': nn.MSELoss,
    'MAE': nn.L1Loss,
    'Smooth MAE': nn.SmoothL1Loss
}

OPTIM = {
    'AMSGrad': torch.optim.Adam
}

METRICS = {
    'MyMetric': MyMetric,
    'R2Score': R2Score,
    'MAPE'   : MAPE,
    'WAPE'   : WAPE
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


if __name__ == '__main__':

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    st.title('Experiment NeuralODE for moisor')

    SETTINGS = dict()

    st.sidebar.title('Settings')
    SETTINGS['name'] = st.sidebar.text_input(label='Name of experiment', value='exp1', max_chars=10, placeholder='text name of experiment')
    SETTINGS['method'] = st.sidebar.selectbox(label='ODE solver', options=SOLVERS)
    SETTINGS['optim'] = st.sidebar.selectbox(label='Optimizer', options=OPTIM.keys())
    SETTINGS['loss'] = st.sidebar.selectbox(label='Loss func', options=CRITERION.keys())
    SETTINGS['metric'] = st.sidebar.selectbox(label='Metric func', options=METRICS.keys())
    SETTINGS['lr'] = st.sidebar.number_input(label='Learning rate', min_value=0., max_value=0.1, value=1e-3, step=1e-6, format='%.6f')
    SETTINGS['interval'] = st.sidebar.number_input(label='Checkpoint interval', min_value=1, max_value=100, value=1, step=1)
    SETTINGS['batch_size'] = st.sidebar.slider(label='Batch size', min_value=6, max_value=128, value=32, step=2)
    SETTINGS['num_epoch'] = st.sidebar.slider(label='Number of epoches', min_value=10, max_value=200, value=30, step=10)
    SETTINGS['layers'] = st.sidebar.text_input(label='Weights number', value='16 32 16', help='Number of weights for each hiden layer')
    SETTINGS['embedings'] = st.sidebar.text_input(label='Weights number', value='16 3', help='Length of embeding vector')
    SETTINGS['act_fun'] = st.sidebar.selectbox(label='Activation function', options=ACTIVATION.keys())
    SETTINGS['adjoint'] = st.sidebar.checkbox(label='Use abjoint method', value=False, help='https://arxiv.org/abs/1806.07366')
    SETTINGS['clicked'] = st.sidebar.button('Start runnig', disabled=False)

    if SETTINGS['clicked']:

        try:

            Path(f'logs/').mkdir(exist_ok=True)
            Path(f"assets/{SETTINGS['name']}").mkdir(exist_ok=True)
            Path(f"assets/{SETTINGS['name']}/imgs").mkdir(exist_ok=True)

            LOGGING_CONFIG['handlers']['file_handler']['filename'] = f"logs/{SETTINGS['name']}.log"
            logging.config.dictConfig(LOGGING_CONFIG)
            logger = logging.getLogger(__name__)

            logger.info(f"Start {SETTINGS['name']}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if SETTINGS['adjoint']:
                from torchdiffeq import odeint_adjoint as odeint
            else:
                from torchdiffeq import odeint

            criterion = CRITERION[SETTINGS['loss']]().to(device)
            metric = METRICS[SETTINGS['metric']]()
            layers = [int(i) for i in SETTINGS['layers'].split(' ')]
            embedings = [int(i) for i in SETTINGS['embedings'].split(' ')]
            act_fun = ACTIVATION[SETTINGS['act_fun']]
            func = ODEF(layers, embedings, act_fun).to(device)
            optimizer = OPTIM[SETTINGS['optim']](func.parameters(), lr=SETTINGS['lr'], amsgrad=True)
            with st.spinner('Load data...'):
                dataloader = DataLoader(DataNPZ('train'), batch_size=SETTINGS['batch_size'], shuffle=True)
                val = DataLoader(DataNPZ('val'), batch_size=SETTINGS['batch_size'], shuffle=True)
                sample = DataLoader(DataNPZ('sample'), batch_size=11)
            st.success('Data is loaded!')
            st.info('Start experiment model')

            experiment(odeint, func, dataloader, val, sample, optimizer, criterion, metric, SETTINGS, LOGGING_CONFIG, streamlit=True)

            st.success('Experiment ended!')

            st.stop()

        except Exception as exp:

            err = format_exc()
            logger.error(err)
            raise(exp)

        logger.info(f"End {SETTINGS['name']}")
