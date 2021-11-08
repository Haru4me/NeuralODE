import argparse
import logging.config
from traceback import format_exc

import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset

from tools.settings import LOGGING_CONFIG

import numpy as np
import imageio
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('seaborn')



parser = argparse.ArgumentParser(prog='Тест NeuralODE',
                                 description="""
    Скрипт тестирует реализованный NeuralODE
    """,formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-adjoint', action='store_true',
                    help='Использовать adjoint_odeint или дефолтный')
parser.add_argument('-lr', type=float, default=0.01,
                    help='Скорость обучения')
parser.add_argument('-batch_size', type=int, default=32,
                    help='Размер батча')
parser.add_argument('-n','--exp_name', type=str, required=True,
                    help='Название эксперимента')
parser.add_argument('-m', '--method', type=str, choices=['euler', 'rk4', 'dopri5', 'fixed_adams'], default='euler',
                    help='Выбор метода решения ОДУ')
parser.add_argument('-e','--num_epoch', type=int, default=250, 
                    help='Количество эпох')
opt = parser.parse_args()

Path(f'logs/').mkdir(exist_ok=True)

LOGGING_CONFIG['handlers']['file_handler']['filename'] = f'logs/{opt.exp_name}.log'
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

if opt.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class ODEF(nn.Module):

    def __init__(self, w=None):
        super().__init__()
        self.net = nn.Linear(2, 2, bias=False)
        if w is not None:
            self.net.weight = nn.Parameter(w)

    def forward(self, t, z):
        return self.net(z)



class Data(Dataset):
    
    def __init__(self, func, z0, t, noise=0.01, method='dopri5', options={}):

        self.z = odeint(func, z0, t, method=method, options=options)
        self.z += torch.randn_like(self.z) * noise
        self.t = t
    
    def __len__(self):
        return self.t.size(0)
    
    def __getitem__(self, index):
        index %= self.__len__()
        return self.z[index], self.t[index]


def draw_sample(obs, time, model, name, epoch):

    z0 = torch.Tensor([[0.6, 0.3]])
    t = torch.linspace(time.min(), time.max(), 400)
    model.eval()
    traj = odeint(model, z0, t, method=opt.method)


    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].scatter(obs[:, 0, 0].detach().numpy(), 
                  obs[:, 0, 1].detach().numpy(), 
                  c=time, s=15, cmap='magma')
    ax[0].plot(traj[:, 0, 0].detach().numpy(),
               traj[:, 0, 1].detach().numpy())
    ax[0].set_xlim(-0.7, 0.7)
    ax[0].set_ylim(-0.7, 0.7)
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')

    ax[1].plot(time.detach().numpy(), 
               obs[:, 0, 0].detach().numpy())
    ax[1].plot(time.detach().numpy(),
               obs[:, 0, 1].detach().numpy())
    ax[1].plot(t.detach().numpy(),
               traj[:, 0, 0].detach().numpy(),'--')
    ax[1].plot(t.detach().numpy(),
               traj[:, 0, 1].detach().numpy(),'--')
    ax[1].set_xlim(time.min(), time.max())
    ax[1].set_ylim(-0.7, 0.7)
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$x,y$')


    plt.savefig(f'assets/{name}/imgs/{epoch}.png')
    plt.cla()
    plt.close(fig)


if __name__ == '__main__':
    
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        logger.info(f'START test model with '+', '.join('{}: {}'.format(key, val) \
                        for key, val in opt._get_kwargs()))

        Path(f'assets/{opt.exp_name}').mkdir(parents=True, exist_ok=True)
        Path(f'assets/{opt.exp_name}/imgs').mkdir(parents=True, exist_ok=True)
        Path(f'assets/{opt.exp_name}/model').mkdir(parents=True, exist_ok=True)

        w = torch.Tensor([[-0.1, -1.], [1., -0.1]])
        datafunc = ODEF(w)
        z0 = torch.Tensor([[0.6, 0.3]])
        t = torch.linspace(0, 6.29*5, 200)

        with torch.no_grad():
            loader = DataLoader(Data(datafunc, z0, t),
                                    batch_size=opt.batch_size, shuffle=False, drop_last=True)

            obs = odeint(datafunc, z0, t)
            obs += torch.randn_like(obs) * 0.01
        
        loss_func = nn.MSELoss()
        func = ODEF()
        optimizer = torch.optim.Adam(func.parameters(), lr=opt.lr)

        pbar = tqdm(total=opt.num_epoch, desc='Training model')

        with pbar:

            for epoch in range(opt.num_epoch):
                func.train()
                mean_loss = []
                for batch in loader:

                    optimizer.zero_grad()

                    target, time = batch
                    z0 = target[0]

                    preds = odeint(func, z0, time, method=opt.method, options={'step_size':0.05})
                    loss = loss_func(preds, target)
                    loss.backward()
                    optimizer.step()

                    mean_loss.append(loss.item())

                pbar.update(1)

                if epoch % 10 == 0:
                    draw_sample(obs, t, func, opt.exp_name, epoch)
                    logger.info('Epoch %i\t–\tLoss: %.6f' % (epoch, np.mean(mean_loss)))
                    state_dict = {
                        'epoch': epoch,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }
                    torch.save(state_dict, f'assets/{opt.exp_name}/model/model')
    
        images = []
        for filename in [f'assets/{opt.exp_name}/imgs/{i}.png' for i in range(0, opt.num_epoch, 10)]:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'assets/{opt.exp_name}/{opt.exp_name}.gif', images)
                    
    except Exception as exp:
        err = format_exc()
        logger.error(err)
        raise(exp)

    logger.info('END test model')

