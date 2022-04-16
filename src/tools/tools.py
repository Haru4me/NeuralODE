from cProfile import label
from click import option
import streamlit as st
from tqdm import tqdm
import logging.config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.optim.lr_scheduler import LambdaLR
import torch

plt.style.use('seaborn')


def draw_sample(stat, name, epoch, streamlit, smpl, gt):

    if torch.cuda.is_available():
        smpl = smpl.cpu().detach()
        gt = gt.cpu().detach()

    epoches = np.arange(stat.shape[0])
    fig = plt.figure(figsize=(14, 15))

    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title('val_1 loss')
    ax1.plot(epoches, stat['train_loss_1'].values, label='train')
    ax1.plot(epoches, stat['val_loss_1'].values, label='val')
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_title('val_2 loss')
    ax2.plot(epoches, stat['train_loss_2'].values, label='train')
    ax2.plot(epoches, stat['val_loss_2'].values, label='val')
    ax2.legend()


    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_title('Output sample')
    x1 = smpl[-1, :, 0].detach().numpy()
    x2 = smpl[-1, :, 1].detach().numpy()
    t = np.linspace(0, 1, x1.shape[0])
    ax3.plot(t, x1, c='b', label='val_1 pred')
    ax3.plot(t, x2, c='g', label='val_2 pred')
    ax3.scatter(t, gt[:, 0], color='b', label='val_1 ground truth')
    ax3.scatter(t, gt[:, 1], color='g', label='val_2 ground truth')
    """
    for i in range(11):
        x1 = smpl[:, i, 0].detach().numpy()
        x2 = smpl[:, i, 1].detach().numpy()
        t = np.linspace(i, i+1, x1.shape[0])
        #bar = ax3.bar(t[1:], r[i].detach().numpy(), width=np.diff(t).mean(), color='b', alpha=0.3)
        line1, = ax3.plot(t, x1, c='b')
        line2, = ax3.plot(t, x2, c='g')
        point1 = ax3.scatter(t[[0, -1]], [start[i, 0].item(), gt[i, 0].item()], color='b')
        point2 = ax3.scatter(t[[0, -1]], [start[i, 1].item(), gt[i, 1].item()], color='g')

    line1.set_label('predict val_1')
    line2.set_label('predict val_2')
    point1.set_label('truth val_1')
    point2.set_label('truth val_2')
    #bar.set_label('rainfall')
    """

    ax3.legend(ncol=2)
    ax3.set_ylim(ymax=42)

    ax4 = fig.add_subplot(gs[0, 0])
    ax4.plot(epoches, stat['train_loss'].values, label='train')
    ax4.plot(epoches, stat['val_loss'].values, label='val')
    ax4.set_title('Loss')
    ax4.legend()

    ax5 = fig.add_subplot(gs[0, 1])
    ax5.plot(epoches, stat['train_metric'].values, label='train')
    ax5.plot(epoches, stat['val_metric'].values, label='val')
    ax5.set_title('Metric')
    ax5.legend()

    plt.savefig(f'assets/{name}/imgs/{epoch}.png')

    if streamlit:
        return fig
    else:
        plt.cla()
        plt.close(fig)


def experiment(odeint,
               func,
               dataloader,
               val,
               sample,
               optimizer,
               criterion,
               metric,
               settings,
               log_config,
               streamlit=False):

    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__name__)

    logger.info('Start experiment')
    logger.info(settings)

    if streamlit:
        method = settings['method']
        name = settings['name']
        interval = settings['interval']
        num_epoch = settings['num_epoch']
        st.header('Progress bar')
        pbar = st.progress(0)
        placeholder = st.empty()
    else:
        method = settings.method
        name = settings.name
        interval = settings.interval
        num_epoch = settings.num_epoch
        name = settings.name
        pbar = tqdm(range(num_epoch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_lr = lambda epoch: 0.1 ** (epoch / 150)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    stats = pd.DataFrame([], columns=['train_loss', 'val_loss', 'train_metric', 'val_metric', 'val_loss_1', 'val_loss_2', 'train_loss_1', 'train_loss_2'])

    with pbar:

        for epoch in range(num_epoch):

            run_loss = []
            run_loss_1 = []
            run_loss_2 = []
            run_metric = []
            val_loss = []
            val_loss_1 = []
            val_loss_2 = []
            val_metric = []

            func.train()

            for batch in dataloader:

                optimizer.zero_grad()
                v, z = batch
                v, z = v.to(device), z.to(device)
                z0, z1, t = z[:, 0], z[:, -1], torch.linspace(0, 25, 81, device=device)

                first, preds = odeint(func, z0, v, t, method=method)[[0,-1]]
                loss = criterion(preds, z1)
                loss.backward()
                optimizer.step()

                run_loss.append(loss.item())
                run_loss_1.append(criterion(preds[:, 0], z1[:, 0]).item())
                run_loss_2.append(criterion(preds[:, 1], z1[:, 1]).item())
                run_metric.append(metric([first, preds], z1).item())

            """
                Validation
            """


            if not epoch % interval:

                func.eval()

                for batch in val:

                    optimizer.zero_grad()
                    v, z = batch
                    v, z = v.to(device), z.to(device)
                    z0, z1, t = z[:, 0], z[:, -1], torch.linspace(0, 25, 81, device=device)

                    sample_v, sample_z = next(iter(sample))
                    sample_v, sample_z = sample_v.to(device), sample_z.to(device)
                    sample_z0, sample_z1 = sample_z[:, 0], sample_z[:, 1]

                    first, preds = odeint(func, z0, v, t, method=method)[[0,-1]]
                    sample_pred = odeint(func, sample_z0, sample_v, t, method=method)

                    loss = criterion(preds, z1)

                    val_loss.append(loss.item())
                    val_loss_1.append(criterion(preds[:, 0], z1[:, 0]).item())
                    val_loss_2.append(criterion(preds[:, 1], z1[:, 1]).item())
                    val_metric.append(metric([first, preds], z1).item())

                scheduler.step()

                logger.info('Epoch %i\t–\tTrain loss: %.6f\t–\tTrain metric: %.6f\t–\tVal loss: %.6f\t–\tVal metric: %.6f' % (
                    epoch, np.mean(run_loss), np.mean(run_metric), np.mean(val_loss), np.mean(val_metric)))

                state_dict = {
                    'epoch': epoch,
                    'model_state_dict': func.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'settings': settings
                }
                torch.save(state_dict, f'assets/{name}/model.pt')

                stats.loc[epoch, 'train_loss'] = np.mean(run_loss)
                stats.loc[epoch, 'val_loss'] = np.mean(val_loss)
                stats.loc[epoch, 'train_metric'] = np.mean(run_metric)
                stats.loc[epoch, 'val_metric'] = np.mean(val_metric)
                stats.loc[epoch, 'val_loss_1'] = np.mean(val_loss_1)
                stats.loc[epoch, 'val_loss_2'] = np.mean(val_loss_2)
                stats.loc[epoch, 'train_loss_1'] = np.mean(run_loss_1)
                stats.loc[epoch, 'train_loss_2'] = np.mean(run_loss_2)

            if streamlit:

                if epoch == 0:
                    dm1, dm2, dm3, dm4 = np.zeros(4)
                else:
                    dm1, dm2, dm3, dm4 = np.round(stats.iloc[-1, :4].fillna(0).to_numpy()-stats.iloc[-2, :4].fillna(0).to_numpy(), 3)
                    plt.close(fig)

                placeholder.empty()
                pr_point = round((epoch+1)/num_epoch*100)
                pbar.progress(pr_point)

                with placeholder.container():

                    st.header('Metrix')
                    col1, col2, col3, col4 = st.columns(4)

                    m1, m2, m3, m4 = np.round(stats.iloc[-1, :4].fillna(0).to_numpy(), 3)

                    col1.metric(f"{settings['loss']} train", m1, dm1, delta_color="inverse")
                    col2.metric(f"{settings['loss']} val", m2, dm2, delta_color="inverse")
                    col3.metric(f"{settings['metric']} train", m3, dm3, delta_color="off")
                    col4.metric(f"{settings['metric']} val", m4, dm4, delta_color="off")

                    st.header('Plots')
                    fig = draw_sample(stats, name, epoch, streamlit, sample_pred, sample_z1)
                    st.pyplot(fig, clear_figure=True)

            else:
                pbar.update(1)
                draw_sample(stats, name, epoch, streamlit, sample_pred, sample_z1)

            stats.to_csv(f'assets/{name}/stats.csv')

            state_dict = {
                'epoch': epoch,
                'model_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'settings': settings,
            }
            torch.save(state_dict, f'assets/{name}/model.pt')


    if streamlit:
        st.success('Experiment finished!')
        st.balloons()

    logger.info('END experiment')
